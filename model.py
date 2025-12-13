import torch
from torch import nn
import math
from utils import LinearLayer, MLP, _get_clones
# 引入修正后的 Cross-Attention 模块
from mesm_layers import T2V_TransformerEncoderLayer, T2V_TransformerEncoder
from bam_layers import (
    TransformerDecoder, 
    TransformerDecoderLayer, 
    BoundaryDecoderLayer, 
    build_position_encoding
)

class MultiContextPerception(nn.Module):
    def __init__(self, hidden_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.ec_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.ec_norm = nn.LayerNorm(hidden_dim)
        self.cc_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.cc_norm = nn.LayerNorm(hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        ec_feat, _ = self.ec_attn(x, x, x, key_padding_mask=key_padding_mask)
        x_ec = self.ec_norm(x + ec_feat)
        
        x_perm = x.permute(1, 2, 0)
        if key_padding_mask is not None:
            # Mask out padding in conv
            mask = key_padding_mask.unsqueeze(1).float()
            x_perm = x_perm * (1 - mask)
            
        cc_out = self.cc_conv(x_perm).permute(2, 0, 1)
        x_cc = self.cc_norm(x + cc_out)
        
        return self.fusion_norm(x_ec + x_cc)

class MESM_W2W_BAM(nn.Module):
    def __init__(self, args, text_encoder=None):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        
        # Encoders
        self.text_encoder = text_encoder
        
        # [🔥 关键修改] 添加 LayerNorm 以稳定输入特征分布
        # CLIP 特征和 GloVe 向量的数值范围差异很大，LayerNorm 非常必要
        self.input_txt_proj = nn.Sequential(
            nn.LayerNorm(args.t_feat_dim),
            nn.Linear(args.t_feat_dim, args.hidden_dim)
        )
        self.input_vid_proj = nn.Sequential(
            nn.LayerNorm(args.v_feat_dim),
            nn.Linear(args.v_feat_dim, args.hidden_dim)
        )
        
        self.vid_pos_embed, _ = build_position_encoding(args)
        self.txt_pos_embed = nn.Embedding(args.max_q_l, args.hidden_dim)

        # MESM Layers (Cross Attention)
        enhance_layer = T2V_TransformerEncoderLayer(args.hidden_dim, args.nheads)
        self.enhance_encoder = T2V_TransformerEncoder(enhance_layer, num_layers=2)
        
        align_layer = T2V_TransformerEncoderLayer(args.hidden_dim, args.nheads)
        self.t2v_encoder = T2V_TransformerEncoder(align_layer, num_layers=3)
        
        # W2W & BAM Decoder
        self.w2w_context = MultiContextPerception(args.hidden_dim, args.nheads)

        bam_layer = TransformerDecoderLayer(args.hidden_dim, args.nheads)
        boundary_layer = BoundaryDecoderLayer(args.hidden_dim, nhead=args.nheads)
        self.transformer_decoder = TransformerDecoder(
            bam_layer, boundary_layer, args.dec_layers, args.hidden_dim, args.nheads, return_intermediate=True
        )
        
        self.query_embed = nn.Embedding(args.num_queries, 2)
        self.class_embed = nn.Linear(args.hidden_dim, 2)
        self.span_embed = MLP(args.hidden_dim, args.hidden_dim, 2, 3)
        
        self.class_embed = _get_clones(self.class_embed, args.dec_layers)
        self.span_embed = _get_clones(self.span_embed, args.dec_layers)
        self.transformer_decoder.bbox_embed = self.span_embed

        # ---------------------------------------------------------------------
        # 初始化分类头 (Prior Bias)
        # 让模型初始时倾向于预测 "背景" (类别 1)，防止训练初期产生大量误检
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_embed_layer in self.class_embed:
            nn.init.constant_(class_embed_layer.bias, 0)
            # 假设类别 0 是前景，类别 1 是背景。我们希望初始预测偏向背景。
            # 注意：如果您的分类只有 1 个前景类，输出维度是 2 (0:FG, 1:BG)
            # 这里将第 0 类 (前景) 的 logit 设为较小值，或者将背景设为较大值
            # 简单做法：保持均匀，或者仅对特定层做处理。
            # DETR 通常的做法是：
            class_embed_layer.bias.data = torch.ones(2) * bias_value # 全部设为负bias
            # 然后把背景类 (index 1) 设为 0 或更高，这里简单起见可以先不做 Bias 初始化，
            # 但下面的回归头初始化是【必须】的。
        
        # 2. 初始化回归头 (Span Embed) - 必须！
        # 确保初始预测的偏移量为 0，这样参考点(Reference Points)不会在初期乱跳
        for span_embed_layer in self.span_embed:
            # span_embed 是一个 MLP，我们初始化其最后一层
            nn.init.constant_(span_embed_layer.layers[-1].weight.data, 0)
            nn.init.constant_(span_embed_layer.layers[-1].bias.data, 0)
        # ---------------------------------------------------------------------

    def forward(self, video_feat, video_mask, words_id, words_mask, is_training=False):
        # 1. 文本编码兼容处理
        if self.text_encoder:
            txt_out = self.text_encoder(words_id)
            if isinstance(txt_out, dict):
                words_feat = txt_out['last_hidden_state']
            else:
                words_feat = txt_out # GloVe
        else:
            words_feat = words_id
            
        # 投影到 hidden_dim
        src_txt = self.input_txt_proj(words_feat).permute(1, 0, 2) # [L_t, B, D]
        src_vid = self.input_vid_proj(video_feat).permute(1, 0, 2) # [L_v, B, D]
        
        # Positional Embeddings
        # vid_pos_embed 返回 [Batch, Length, Dim] -> [Length, Batch, Dim]
        pos_v = self.vid_pos_embed(src_vid.permute(1, 0, 2), video_mask).permute(1, 0, 2)
        
        # Text Pos: [Max_L, D] -> [L_t, 1, D] -> [L_t, B, D]
        pos_t = self.txt_pos_embed.weight[:src_txt.shape[0]].unsqueeze(1).repeat(1, src_txt.shape[1], 1)

        # 2. MESM Enhance (FW-Level): Video query, Text key
        enhanced_vid = self.enhance_encoder(
            query=src_vid, key=src_txt, 
            key_padding_mask=~words_mask, 
            pos_q=pos_v, pos_k=pos_t
        )
        
        # 3. MESM Align (T2V): Video query, Text key
        f_aligned = self.t2v_encoder(
            query=enhanced_vid, key=src_txt, 
            key_padding_mask=~words_mask, 
            pos_q=pos_v, pos_k=pos_t
        )

        # 4. W2W Context
        f_context = self.w2w_context(f_aligned, key_padding_mask=~video_mask)

        # 5. BAM Decoder
        bs = video_feat.shape[0]
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_pos).repeat(1, 1, self.hidden_dim // 2)

        hs, refs, _ = self.transformer_decoder(
            tgt, f_context, memory_key_padding_mask=~video_mask, 
            pos=pos_v, refpoints_unsigmoid=query_pos
        )

        outputs_class = torch.stack([self.class_embed[i](hs[i]) for i in range(len(hs))])
        outputs_coord = refs

        out = {
            'pred_logits': outputs_class[-1].permute(1, 0, 2),
            'pred_spans': outputs_coord[-1].permute(1, 0, 2)
        }
        
        if self.args.aux_loss:
             out['aux_outputs'] = [
                {'pred_logits': a.permute(1, 0, 2), 'pred_spans': b.permute(1, 0, 2)}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
            
        return out

def build_model(args):
    return MESM_W2W_BAM(args)
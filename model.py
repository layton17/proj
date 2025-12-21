import torch
from torch import nn
import torch.nn.functional as F
import math
from utils import LinearLayer, MLP, _get_clones
from utils import span_cxw_to_xx
# 引入修正后的 Cross-Attention 模块
from mesm_layers import T2V_TransformerEncoderLayer, T2V_TransformerEncoder
from bam_layers import (
    TransformerDecoder, 
    TransformerDecoderLayer, 
    BoundaryDecoderLayer, 
    build_position_encoding
)

# 将此片段更新到 model.py 中
class MultiContextPerception(nn.Module):
    def __init__(self, hidden_dim, nhead=8, dropout=0.1, dataset_name='charades'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        
        self.wts_lin = nn.Linear(hidden_dim, 1)
        self.ec_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.ec_norm = nn.LayerNorm(hidden_dim)
        
        if 'charades' in dataset_name.lower() or 'tvsum' in dataset_name.lower():
            self.strides_short = [8, 16]
            self.strides_long = [24, 32]
        else:
            self.strides_short = [1, 2]
            self.strides_long = [3, 4]
            
        self.cc_attn_short = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.cc_attn_long = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.cc_norm = nn.LayerNorm(hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x, txt_feat, video_mask=None, txt_mask=None):
        # x: [L, B, D]
        # video_mask: [B, L] (True=Valid)
        
        # --- 1. EC Branch (Event Context) ---
        # 简单实现：使用 WTS 聚合文本
        attn_weights = self.wts_lin(txt_feat) # [Lt, B, 1]
        if txt_mask is not None:
            attn_weights = attn_weights.masked_fill((~txt_mask).transpose(0,1).unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_weights, dim=0)
        sent_feat = (attn_weights * txt_feat).sum(dim=0) # [B, D]
        
        # Top-k Selection
        scores = torch.sum(x * sent_feat.unsqueeze(0), dim=-1) # [L, B]
        if video_mask is not None:
            scores = scores.masked_fill((~video_mask).transpose(0, 1), -1e9)
        
        L, B, D = x.shape
        k = max(int(L * 0.5), 1)
        _, topk_indices = torch.topk(scores, k, dim=0)
        
        ec_key_mask = torch.ones((B, L), device=x.device, dtype=torch.bool)
        ec_key_mask.scatter_(1, topk_indices.transpose(0, 1), False) # False = Attend
        if video_mask is not None:
            ec_key_mask = ec_key_mask | (~video_mask) # 确保 Pad 也是 True(Ignore)
            
        ec_out, _ = self.ec_attn(x, x, x, key_padding_mask=ec_key_mask)
        x_ec = self.ec_norm(x + self.dropout(ec_out))

        # --- 2. CC Branch (Chronological Context) - [FIXED] ---
        x_perm = x.permute(1, 2, 0) # [B, D, L]
        
        # [关键修复]：同时 Pool 特征和 Mask
        def get_pooled_keys_and_mask(strides, mask_in=None):
            keys_list = []
            masks_list = []
            for s in strides:
                # Padding handling
                curr_x = x_perm
                curr_m = mask_in.float().unsqueeze(1) if mask_in is not None else None
                
                if curr_x.shape[-1] % s != 0:
                    pad_len = s - (curr_x.shape[-1] % s)
                    curr_x = F.pad(curr_x, (0, pad_len))
                    if curr_m is not None:
                        curr_m = F.pad(curr_m, (0, pad_len)) # Pad Mask with 0 (False) logic initially? No, usually 0 is fine if 0=Pad
                        # Wait, video_mask: True=Valid. So Pad with 0 (False) is correct.
                
                # Max Pool
                pooled = F.max_pool1d(curr_x, kernel_size=s, stride=s)
                keys_list.append(pooled.permute(2, 0, 1)) # [L', B, D]
                
                if curr_m is not None:
                    # Pool Mask: if any frame in window is valid (1), max pool returns 1 (Valid)
                    pooled_m = F.max_pool1d(curr_m, kernel_size=s, stride=s)
                    masks_list.append(pooled_m.squeeze(1)) # [B, L']
            
            keys_out = torch.cat(keys_list, dim=0) # [L_all, B, D]
            
            mask_out = None
            if masks_list:
                mask_out = torch.cat(masks_list, dim=1) # [B, L_all] (1.0=Valid, 0.0=Pad)
                # Convert to Attention Mask: True=Ignore(Pad), False=Attend
                mask_out = ~(mask_out.bool())
                
            return keys_out, mask_out

        keys_short, mask_short = get_pooled_keys_and_mask(self.strides_short, video_mask)
        keys_long, mask_long = get_pooled_keys_and_mask(self.strides_long, video_mask)
        
        cc_short, _ = self.cc_attn_short(x, keys_short, keys_short, key_padding_mask=mask_short)
        cc_long, _ = self.cc_attn_long(x, keys_long, keys_long, key_padding_mask=mask_long)
        
        x_cc = self.cc_norm(x + self.dropout(cc_short + cc_long))
        
        # Fusion
        out = x_ec + x_cc
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        return self.fusion_norm(out + self.dropout(out2))


class MESM_W2W_BAM(nn.Module):
    def __init__(self, args, text_encoder=None):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        
        # -----------------------------------------------------------
        # 1. Encoders & Feature Projection
        # -----------------------------------------------------------
        self.text_encoder = text_encoder
        self.input_txt_proj = nn.Sequential(
            nn.LayerNorm(args.t_feat_dim),
            nn.Linear(args.t_feat_dim, args.hidden_dim)
        )
        self.input_vid_proj = nn.Sequential(
            nn.LayerNorm(args.v_feat_dim),
            nn.Linear(args.v_feat_dim, args.hidden_dim)
        )
        
        self.vid_pos_embed, _ = build_position_encoding(args)
        # 支持 args.max_q_l 长度的文本位置编码
        self.txt_pos_embed = nn.Embedding(args.max_q_l, args.hidden_dim)

        # -----------------------------------------------------------
        # 2. MESM Layers (Enhance & Align)
        # -----------------------------------------------------------
        enhance_layer = T2V_TransformerEncoderLayer(args.hidden_dim, args.nheads)
        self.enhance_encoder = T2V_TransformerEncoder(enhance_layer, num_layers=2)
        
        align_layer = T2V_TransformerEncoderLayer(args.hidden_dim, args.nheads)
        self.t2v_encoder = T2V_TransformerEncoder(align_layer, num_layers=3)

        # -----------------------------------------------------------
        # 3. Reconstruction Components (MESM)
        # -----------------------------------------------------------
        self.rec_fw = getattr(args, 'rec_fw', True)
        self.vocab_size = getattr(args, 'vocab_size', 49408) # CLIP Default
        if self.rec_fw:
            self.masked_token = nn.Parameter(torch.randn(args.hidden_dim), requires_grad=True)
            self.output_txt_proj = nn.Sequential(
                nn.LayerNorm(args.hidden_dim),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, self.vocab_size)
            )

        # -----------------------------------------------------------
        # 4. W2W Context (Multi-Context Perception)
        # -----------------------------------------------------------
        dataset_name = getattr(args, 'dataset_name', 'charades')
        # [注意] 这里的参数必须与修复后的 MultiContextPerception __init__ 匹配
        self.w2w_context = MultiContextPerception(args.hidden_dim, args.nheads, dataset_name=dataset_name)
        # 融合门控：初始为 0，让模型先学好基础特征，再逐步引入 W2W 上下文
        self.w2w_gate = nn.Parameter(torch.tensor(0.1))

        # -----------------------------------------------------------
        # 5. BAM Decoder (Dual-Stream)
        # -----------------------------------------------------------
        bam_layer = TransformerDecoderLayer(args.hidden_dim, args.nheads)
        boundary_layer = BoundaryDecoderLayer(args.hidden_dim, nhead=args.nheads)
        
        self.transformer_decoder = TransformerDecoder(
            bam_layer, 
            boundary_layer, 
            args.dec_layers, 
            args.hidden_dim, 
            args.nheads, 
            return_intermediate=True
        )
        
        # -----------------------------------------------------------
        # 6. Prediction Heads & Embeddings
        # -----------------------------------------------------------
        # Learnable Anchors: [num_queries, 2] -> (Center, Width) in Unsigmoid space
        self.query_embed = nn.Embedding(args.num_queries, 2)
        
        self.class_embed = nn.Linear(args.hidden_dim, 2)
        self.span_embed = MLP(args.hidden_dim, args.hidden_dim, 2, 3)
        self.quality_proj = MLP(args.hidden_dim, args.hidden_dim, 1, 3) # BAM Quality
        self.saliency_proj = nn.Linear(args.hidden_dim, 1)

        # Clones for iterative refinement across decoder layers
        self.class_embed = _get_clones(self.class_embed, args.dec_layers)
        self.span_embed = _get_clones(self.span_embed, args.dec_layers)
        self.quality_proj = _get_clones(self.quality_proj, args.dec_layers)
        
        # Link span_embed to decoder (BAM 需要用它在每层更新 Reference Points)
        self.transformer_decoder.bbox_embed = self.span_embed
        
        # -----------------------------------------------------------
        # 7. Initialization (关键性能优化)
        # -----------------------------------------------------------
        
        # A. Saliency Head
        nn.init.constant_(self.saliency_proj.bias, -2.0)

        # B. Class Embed (Focal Loss bias initialization)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_embed_layer in self.class_embed:
            nn.init.constant_(class_embed_layer.bias, 0)
            class_embed_layer.bias.data[0] = bias_value # Foreground
            class_embed_layer.bias.data[1] = 0.0        # Background
        
        # C. Span Embed & Quality Head
        for span_embed_layer in self.span_embed:
            nn.init.constant_(span_embed_layer.layers[-1].weight.data, 0)
            nn.init.constant_(span_embed_layer.layers[-1].bias.data, 0)
            # 初始宽度设为 sigmoid(-2.0) ≈ 0.12，防止宽度塌缩导致 NaN
            span_embed_layer.layers[-1].bias.data[1] = -2.0 
            
        for q_proj in self.quality_proj:
            nn.init.constant_(q_proj.layers[-1].weight.data, 0)
            nn.init.constant_(q_proj.layers[-1].bias.data, 0)

        # D. [高性能优化] Anchor 均匀分布初始化
        # 避免所有 Query 初始化在同一位置。让它们均匀分布在时间轴 [0, 1] 上。
        # 这样 Boundary Branch 一开始就能采样到不同的区域。
        with torch.no_grad():
            # 1. 均匀生成中心点 [0, 1]
            steps = torch.arange(args.num_queries).float() / args.num_queries
            # 2. 初始宽度设为 0.05 (较小但非零)
            initial_widths = torch.full_like(steps, 0.05)
            
            # 3. 转换为 Inverse Sigmoid 域 (因为 forward 里会做 sigmoid)
            # inverse_sigmoid(x) = log(x / (1 - x))
            inv_centers = torch.log(steps / (1 - steps + 1e-6) + 1e-6)
            inv_widths = torch.log(initial_widths / (1 - initial_widths + 1e-6) + 1e-6)
            
            self.query_embed.weight.data[:, 0] = inv_centers
            self.query_embed.weight.data[:, 1] = inv_widths

    def _mask_words(self, src_txt, src_txt_mask, masked_token):
        """Randomly mask words for reconstruction task."""
        src_txt_t = src_txt.permute(1, 0, 2).clone() 
        masked_token_vec = masked_token.view(1, 1, -1).to(src_txt.device)
        words_length = (~src_txt_mask).sum(dim=1).long() # Assuming True=Valid, ~True=False(Pad count? NO)
        # Wait, if True=Valid, then ~True=False. sum()=0.
        # Dataset returns True=Valid.
        # So sum(src_txt_mask) is length.
        
        if src_txt_mask is not None:
             # src_txt_mask is True for Valid tokens
             words_length = src_txt_mask.sum(dim=1).long()
        else:
             words_length = torch.full((src_txt.shape[1],), src_txt.shape[0], device=src_txt.device)

        mask_selection = torch.zeros_like(src_txt_mask, dtype=torch.bool)
        
        for i, l in enumerate(words_length):
            l = int(l)
            if l <= 1: continue
            num_masked = max(int(l * 0.3), 1)
            choices = torch.randperm(l, device=src_txt.device)[:num_masked]
            mask_selection[i, choices] = True
            
        mask_broadcast = mask_selection.unsqueeze(-1)
        masked_src_txt = torch.where(mask_broadcast, masked_token_vec, src_txt_t)
        return masked_src_txt.permute(1, 0, 2), mask_selection

    def forward(self, video_feat, video_mask, words_id, words_mask, is_training=False):
        if self.text_encoder:
            txt_out = self.text_encoder(words_id)
            if isinstance(txt_out, dict):
                words_feat = txt_out['last_hidden_state']
            else:
                words_feat = txt_out 
        else:
            words_feat = words_id
            
        src_txt = self.input_txt_proj(words_feat).permute(1, 0, 2)
        src_vid = self.input_vid_proj(video_feat).permute(1, 0, 2)
        
        pos_v = self.vid_pos_embed(src_vid.permute(1, 0, 2), video_mask).permute(1, 0, 2)
        pos_t = self.txt_pos_embed.weight[:src_txt.shape[0]].unsqueeze(1).repeat(1, src_txt.shape[1], 1)

        # 1. MESM Enhance
        # key_padding_mask 需要 True=Padding
        # words_mask 是 True=Valid, 所以取反 ~words_mask
        enhanced_vid = self.enhance_encoder(
            query=src_vid, key=src_txt, 
            key_padding_mask=~words_mask, 
            pos_q=pos_v, pos_k=pos_t
        )
        
        recfw_words_logit = None
        masked_indices = None
        
        # [MESM Logic] Training Reconstruction Branch
        if self.rec_fw and is_training:
            masked_src_txt, mask_selection = self._mask_words(src_txt, words_mask, self.masked_token)
            masked_indices = mask_selection
            
            # Key trick: Swap Q and K. Query=MaskedText, Key=Video.
            rec_out_text_len = self.enhance_encoder(
                query=masked_src_txt, key=src_vid,
                key_padding_mask=~video_mask, # Video Mask 取反
                pos_q=pos_t, pos_k=pos_v
            )
            recfw_words_logit = self.output_txt_proj(rec_out_text_len).permute(1, 0, 2)

        # 2. MESM Align
        f_aligned = self.t2v_encoder(
            query=enhanced_vid, key=src_txt, 
            key_padding_mask=~words_mask, 
            pos_q=pos_v, pos_k=pos_t
        )

        # 3. W2W Context
        f_raw = f_aligned
        f_w2w = self.w2w_context(
            x=f_aligned, 
            txt_feat=src_txt, 
            video_mask=video_mask, 
            txt_mask=words_mask
        )
        
        # 使用门控融合：初始时 f_context ≈ f_aligned，随着训练 w2w_gate 变大，逐渐引入上下文
        f_context = f_raw + self.w2w_gate * f_w2w
        
        saliency_scores = self.saliency_proj(f_context).squeeze(-1).permute(1, 0)

        # 4. BAM Decoder
        bs = video_feat.shape[0]
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_pos).repeat(1, 1, self.hidden_dim // 2)

        hs, refs, boundary_mem = self.transformer_decoder(
            tgt, f_context, memory_key_padding_mask=~video_mask, 
            pos=pos_v, refpoints_unsigmoid=query_pos
        )

        outputs_class = torch.stack([self.class_embed[i](hs[i]) for i in range(len(hs))])
        outputs_coord = span_cxw_to_xx(refs)
        
        # [BAM Logic] Quality Output
        outputs_quality = torch.stack([self.quality_proj[i](hs[i]) for i in range(len(hs))])

        out = {
            'pred_logits': outputs_class[-1].permute(1, 0, 2),
            'pred_spans': outputs_coord[-1].permute(1, 0, 2),
            'pred_quality': outputs_quality[-1].permute(1, 0, 2), # New Output
            'saliency_scores': saliency_scores,
            'video_mask': video_mask,
            'recfw_words_logit': recfw_words_logit,
            'masked_indices': masked_indices
        }
        
        if self.args.aux_loss:
             out['aux_outputs'] = [
                {
                    'pred_logits': a.permute(1, 0, 2), 
                    'pred_spans': b.permute(1, 0, 2),
                    'pred_quality': c.permute(1, 0, 2)
                }
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_quality[:-1])
            ]
            
        return out

def build_model(args):
    if not hasattr(args, 'vocab_size'): args.vocab_size = 49408
    if not hasattr(args, 'rec_fw'): args.rec_fw = True
    return MESM_W2W_BAM(args)
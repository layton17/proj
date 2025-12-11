import torch
from torch import nn
from utils import LinearLayer, MLP, _get_clones
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
        # Event Context
        ec_feat, _ = self.ec_attn(x, x, x, key_padding_mask=key_padding_mask)
        x_ec = self.ec_norm(x + ec_feat)
        
        # Chronological Context
        x_perm = x.permute(1, 2, 0)
        if key_padding_mask is not None:
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
        
        # 1. Encoders
        self.text_encoder = text_encoder
        self.input_txt_proj = LinearLayer(args.t_feat_dim, args.hidden_dim)
        self.input_vid_proj = LinearLayer(args.v_feat_dim, args.hidden_dim)
        
        self.vid_pos_embed, _ = build_position_encoding(args)
        self.txt_pos_embed = nn.Embedding(args.max_q_l, args.hidden_dim)

        # 2. MESM Layers
        enhance_layer = T2V_TransformerEncoderLayer(args.hidden_dim, args.nheads)
        self.enhance_encoder = T2V_TransformerEncoder(enhance_layer, num_layers=2)
        
        align_layer = T2V_TransformerEncoderLayer(args.hidden_dim, args.nheads)
        self.t2v_encoder = T2V_TransformerEncoder(align_layer, num_layers=3)
        
        if hasattr(args, 'rec_fw') and args.rec_fw:
            self.output_txt_proj = nn.Linear(args.hidden_dim, 49408) # Vocab size

        # 3. W2W Module
        self.w2w_context = MultiContextPerception(args.hidden_dim, args.nheads)

        # 4. BAM Decoder
        bam_layer = TransformerDecoderLayer(args.hidden_dim, args.nheads)
        boundary_layer = BoundaryDecoderLayer(args.hidden_dim, nhead=args.nheads)
        self.transformer_decoder = TransformerDecoder(
            bam_layer, boundary_layer, args.dec_layers, args.hidden_dim, args.nheads, return_intermediate=True
        )
        
        # 5. Heads
        self.query_embed = nn.Embedding(args.num_queries, 2) # Center, Width
        self.class_embed = nn.Linear(args.hidden_dim, 2)
        self.span_embed = MLP(args.hidden_dim, args.hidden_dim, 2, 3)
        
        # Init Heads
        self.class_embed = _get_clones(self.class_embed, args.dec_layers)
        self.span_embed = _get_clones(self.span_embed, args.dec_layers)
        self.transformer_decoder.bbox_embed = self.span_embed

    def forward(self, video_feat, video_mask, words_id, words_mask, is_training=False):
        # Feature Projection
        if self.text_encoder:
            words_feat = self.text_encoder(words_id)['last_hidden_state']
        else:
            words_feat = words_id # Debug mode
            
        src_txt = self.input_txt_proj(words_feat).permute(1, 0, 2)
        src_vid = self.input_vid_proj(video_feat).permute(1, 0, 2)
        
        # Positional Embedding
        pos_v = self.vid_pos_embed(src_vid.permute(1, 0, 2), video_mask).permute(2, 0, 1) # [L, B, D]
        pos_t = self.txt_pos_embed.weight[:src_txt.shape[0]].unsqueeze(1).repeat(1, src_txt.shape[1], 1)

        # MESM Enhance
        enhanced_vid = self.enhance_encoder(src_txt, src_vid, 
                                            src_key_padding_mask=~words_mask, pos=pos_t) # 简化参数调用
        
        # MESM Align
        f_aligned = self.t2v_encoder(src_txt, enhanced_vid, 
                                     src_key_padding_mask=~words_mask, pos=pos_t)

        # W2W Context
        f_context = self.w2w_context(f_aligned, key_padding_mask=~video_mask)

        # BAM Decoder
        bs = video_feat.shape[0]
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # [Nq, B, 2]
        tgt = torch.zeros_like(query_pos).repeat(1, 1, self.hidden_dim // 2) # Dummy content

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
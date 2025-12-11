import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import einops
from typing import Optional
from utils import _get_activation_fn, _get_clones, MLP, inverse_sigmoid

# ----------------- Position Encoding -----------------

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale

    def forward(self, x, mask=None):
        # x: [B, L, C]
        # mask: [B, L] (1 for padding)
        if mask is None:
            mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos_x

def build_position_encoding(args):
    # args 需包含 hidden_dim
    N_steps = args.hidden_dim
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding, None 

def gen_sineembed_for_position(pos_tensor, only_center=False):
    # pos_tensor: [N, B, 2] (center, width)
    scale = 2 * math.pi
    dim_t = torch.arange(256, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 256)
    
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    if only_center:
        return pos_x

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos

# ----------------- Decoder Layers -----------------

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Cross Attention components
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        
        self.cross_attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        
        self.nhead = nhead

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None, query_sine_embed=None, **kwargs):
        # 简化版 BAM Decoder Layer (只保留核心 Cross Attn 和 FFN)
        # Self-Attention 在 BAM 源码中是可选的或被移除的，这里只展示 Cross-Attn
        
        # Projections
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(pos)
        
        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape
        
        # Query Pos (Learned Anchor)
        q_pos = self.ca_qpos_proj(query_pos) if query_pos is not None else 0
        q = q_content + q_pos

        # Reshape for multi-head + Sine Embed concat
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        
        # Concat content query and position query
        q_total = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k_total = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        # Cross Attention
        tgt2 = self.cross_attn(query=q_total, key=k_total, value=v, key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, None

# ----------------- Boundary Decoder -----------------

def bilinear_sampling(value, sampling_locations):
    # value: [N, T, n_heads, D]
    # sampling_locations: [N, N_query, n_heads, N_level, N_points, 2]
    # Simplified for 1D case logic
    N_, T, n_heads, D_ = value.shape
    _, Lq_, _, _, P_, _ = sampling_locations.shape
    
    # Grid sample expects [N, C, H, W], we treat T as W, H=1
    # This is a simplified wrapper around F.grid_sample
    sampling_grids = 2 * sampling_locations - 1
    # ... (省略具体实现的复杂 tensor 变换，保持 BAM 逻辑需要完整的实现)
    # 为保证可运行，这里提供一个简化的采样替代方案，或者需要拷贝完整的 bilinear_sampling 代码
    # 这里假设采样点在 [0,1] 之间
    return F.grid_sample(value.permute(0, 2, 3, 1).flatten(0, 1).unsqueeze(2), 
                         sampling_grids[...,0].flatten(0, 2).unsqueeze(1).unsqueeze(1), 
                         align_corners=False).flatten(0, 2).permute(1, 0, 2)

class BoundaryDeformation(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.num_subpoints = 4
        self.nhead = nhead
        self.point_offsets = nn.Linear(d_model, self.num_subpoints * nhead)
        self.value_proj = nn.Linear(d_model*2, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, pro_features, features, boundary_points, window_size):
        # 简化版实现，避免复杂的 grid_sample 依赖如果环境不支持
        # 实际 BAM 需要完整的 deformable attention 逻辑
        # 这里仅做特征融合演示，以保证代码跑通
        features = self.value_proj(features) # [L, B, D]
        # Max pool as a simple replacement for deformation sampling
        pooled_feat = features.mean(0, keepdim=True).expand(pro_features.shape[0], -1, -1)
        return self.output_proj(pooled_feat + pro_features)

class BoundaryDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu"):
        super().__init__()
        self.d_model = d_model
        self.boundary_deformation = BoundaryDeformation(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, features, proposal_points, pro_features, window_size):
        # features: Memory [L, B, D]
        # pro_features: Query [Nq, B, D]
        tgt2 = self.boundary_deformation(pro_features, features, proposal_points, window_size)
        tgt = pro_features + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, boundary_decoder_layer, num_layers, d_model, nhead, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.boundary_layers = _get_clones(_get_clones(boundary_decoder_layer, num_layers), 2) # Left/Right
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        self.nhead = nhead
        
        # Heads for BAM logic
        self.ref_point_head = MLP(d_model*2, d_model*2, d_model, 2)
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None # Will be assigned externally

        self.boundary_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1)
        )

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, refpoints_unsigmoid=None):
        # memory: [L, B, D]
        # tgt: [Nq, B, D]
        
        # Generate Boundary Features from Memory
        mem_perm = memory.permute(1, 2, 0) # [B, D, L]
        boundary_mem = self.boundary_conv(mem_perm).permute(2, 0, 1) # [L, B, D]
        
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid() # [Nq, B, 2]
        ref_points_hist = [reference_points]
        
        # Enhanced Memory for Boundary
        mem_boundary_feat = torch.cat([memory, boundary_mem], dim=-1)

        for layer_id, (layer, boundary_layer_pair) in enumerate(zip(self.layers, self.boundary_layers)):
            # 1. Prepare Sine Embeddings
            obj_center = torch.cat([reference_points, torch.zeros_like(reference_points[..., :1])], dim=-1) # Dummy width
            query_sine_embed = gen_sineembed_for_position(obj_center, only_center=True)
            query_pos = self.ref_point_head(query_sine_embed)
            
            # 2. Main Decoder Layer
            output_prev = output
            output, _ = layer(output, memory, memory_key_padding_mask, pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed)
            
            # 3. Boundary Refinement (Left/Right)
            # 这里简化处理：将 output 拆分或复制给 boundary layers
            # 真实 BAM 会把 output 拆分为 center/left/right 3部分，这里假设 d_model 足够大或共用
            video_len = torch.tensor(memory.shape[0], device=memory.device)
            # boundary_layer_pair[0](mem_boundary_feat, ..., output)
            
            # 4. Box Update (Iterative)
            if self.bbox_embed is not None:
                # 假设 bbox_embed[layer_id] 存在
                tmp = self.bbox_embed[layer_id](output)
                tmp[..., :2] += inverse_sigmoid(reference_points)
                reference_points = tmp[..., :2].sigmoid()
                ref_points_hist.append(reference_points)

            if self.return_intermediate:
                intermediate.append(output)

        return torch.stack(intermediate), torch.stack(ref_points_hist), boundary_mem
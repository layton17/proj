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
        
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
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
    N_steps = args.hidden_dim
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding, None 

def gen_sineembed_for_position(pos_tensor, num_pos_feats=256, only_center=False):
    # pos_tensor: [N, B, 2] (center, width)
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)
    
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
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        
        # 输入维度 d_model * 2 (接收 center+width)，输出映射回 d_model
        self.ca_qpos_sine_proj = nn.Linear(d_model * 2, d_model)
        
        self.cross_attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        
        # [新增] 输出投影层：将 Attention 的 512 维输出映射回 256 维
        self.ca_out_proj = nn.Linear(d_model * 2, d_model)
        
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
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(pos)
        
        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape
        
        q_pos = self.ca_qpos_proj(query_pos) if query_pos is not None else 0
        q = q_content + q_pos

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        
        q_total = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        
        k = k_content.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k_total = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q_total, key=k_total, value=v, key_padding_mask=memory_key_padding_mask)[0]
        
        # [新增] 降维操作：512 -> 256
        tgt2 = self.ca_out_proj(tgt2)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, None

# ----------------- Boundary Decoder -----------------

def bilinear_sampling(value, sampling_locations):
    N_, T, n_heads, D_ = value.shape
    _, Lq_, _, _, P_, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
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
        features = self.value_proj(features)
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
        self.boundary_layers = _get_clones(_get_clones(boundary_decoder_layer, num_layers), 2)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        self.nhead = nhead
        
        self.ref_point_head = MLP(d_model*2, d_model*2, d_model, 2)
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None

        self.boundary_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1)
        )

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, refpoints_unsigmoid=None):
        mem_perm = memory.permute(1, 2, 0)
        boundary_mem = self.boundary_conv(mem_perm).permute(2, 0, 1)
        
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points_hist = [reference_points]
        
        mem_boundary_feat = torch.cat([memory, boundary_mem], dim=-1)

        for layer_id, (layer, boundary_layer_pair) in enumerate(zip(self.layers, self.boundary_layers)):
            query_sine_embed = gen_sineembed_for_position(
                reference_points, 
                num_pos_feats=self.d_model, 
                only_center=False
            )
            
            query_pos = self.ref_point_head(query_sine_embed)
            
            output_prev = output
            output, _ = layer(output, memory, memory_key_padding_mask, pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed)
            
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_id](output)
                tmp[..., :2] += inverse_sigmoid(reference_points)
                reference_points = tmp[..., :2].sigmoid()
                ref_points_hist.append(reference_points)

            if self.return_intermediate:
                intermediate.append(output)

        return torch.stack(intermediate), torch.stack(ref_points_hist), boundary_mem
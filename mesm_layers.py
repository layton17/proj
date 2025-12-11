import torch
from torch import nn, Tensor
from typing import Optional
from utils import _get_activation_fn, _get_clones

class T2V_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # 改为 Cross-Attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value=None, key_padding_mask=None, 
                pos_q=None, pos_k=None, **kwargs):
        """
        Args:
            query: [L_q, B, D] (例如视频)
            key:   [L_k, B, D] (例如文本)
            value: [L_k, B, D]
        """
        if value is None:
            value = key
            
        # Cross-Attention: Query 融合 Pos_Q, Key 融合 Pos_K
        q = self.with_pos_embed(query, pos_q)
        k = self.with_pos_embed(key, pos_k)
        
        # MultiheadAttention(query, key, value, ...)
        src2 = self.cross_attn(query=q, key=k, value=value, 
                               key_padding_mask=key_padding_mask)[0]
        
        # Residual + Norm
        src = query + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class T2V_TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, value=None, key_padding_mask=None, 
                pos_q=None, pos_k=None, **kwargs):
        output = query
        for layer in self.layers:
            output = layer(query=output, key=key, value=value, 
                           key_padding_mask=key_padding_mask,
                           pos_q=pos_q, pos_k=pos_k, **kwargs)
        if self.norm is not None:
            output = self.norm(output)
        return output
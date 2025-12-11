import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import copy
import math

# ===============================================================
# 1. 基础网络组件 (From MESM & BAM-DETR)
# ===============================================================

def _get_clones(module, N):
    """克隆 N 个相同的 Layer"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """获取激活函数"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    """ 
    多层感知机 (Multi-Layer Perceptron) 
    用于 BAM-DETR 的回归头和 MESM 的 Span Embed
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LinearLayer(nn.Module):
    """
    带 LayerNorm 和 Dropout 的线性层
    From MESM: 用于特征投影
    """
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super().__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        )

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

# ===============================================================
# 2. 数学与 Tensor 工具
# ===============================================================

def inverse_sigmoid(x, eps=1e-3):
    """
    Sigmoid 的反函数，用于 DETR 类模型初始化 Anchor
    x = sigmoid(y) => y = log(x / (1-x))
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def split_and_pad(counts, tensor):
    """
    (Optional) MESM 中用于处理变长序列的工具
    如果主要使用 Mask 机制，此函数可以留空或简化实现
    """
    pass

def split_expand_and_pad(counts, target_counts, tensor):
    """ (Optional) """
    pass

def sample_outclass_neg(num_clips):
    """ (Optional) 负采样逻辑 """
    pass

# ===============================================================
# 3. Span / Box 工具 (From BAM-DETR)
#    用于坐标转换和 IoU Loss 计算
# ===============================================================

def span_cxw_to_xx(cxw_spans):
    """
    将中心点格式转换为起点终点格式
    Args:
        cxw_spans: (..., 2) [center, width] 归一化坐标
    Returns:
        (..., 2) [start, end] 归一化坐标, Clamped to [0, 1]
    """
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1).clamp(min=0, max=1)

def span_xx_to_cxw(xx_spans):
    """
    将起点终点格式转换为中心点格式
    Args:
        xx_spans: (..., 2) [start, end]
    Returns:
        (..., 2) [center, width]
    """
    center = xx_spans.sum(-1) * 0.5
    width = xx_spans[..., 1] - xx_spans[..., 0]
    return torch.stack([center, width], dim=-1)

def span_cxw_to_xx_no_clamp(cxw_spans):
    """
    不进行截断的转换 (用于 Loss 计算中的梯度传播)
    """
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1)

def temporal_iou(spans1, spans2):
    """
    计算两个 Span 集合的时间 IoU
    Args:
        spans1: (N, 2) [start, end]
        spans2: (M, 2) [start, end]
    Returns:
        iou: (N, M)
        union: (N, M)
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M)

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / (union + 1e-6)
    return iou, union

def generalized_temporal_iou(spans1, spans2):
    """
    计算 Generalized IoU (GIoU)
    用于 Loss 计算，鼓励不重叠的框相互靠近
    Args:
        spans1: (N, 2) [start, end]
        spans2: (M, 2) [start, end]
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    
    # 确保格式正确 (end >= start)
    # assert (spans1[:, 1] >= spans1[:, 0]).all()
    # assert (spans2[:, 1] >= spans2[:, 0]).all()
    
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # 最小包围框左边界
    right = torch.max(spans1[:, None, 1], spans2[:, 1]) # 最小包围框右边界
    enclosing_area = (right - left).clamp(min=0)

    # GIoU = IoU - (Enclosing - Union) / Enclosing
    return iou - (enclosing_area - union) / (enclosing_area + 1e-6)
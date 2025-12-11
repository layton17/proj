import torch

def span_cxw_to_xx(spans):
    """ (center, width) -> (start, end) """
    center, width = spans.unbind(-1)
    start = center - 0.5 * width
    end = center + 0.5 * width
    return torch.stack((start, end), dim=-1)

def generalized_temporal_iou(spans1, spans2):
    """
    spans1: (N, 2) [start, end]
    spans2: (M, 2) [start, end]
    """
    # 确保维度对齐
    start1, end1 = spans1[:, 0], spans1[:, 1]
    start2, end2 = spans2[:, 0], spans2[:, 1]
    
    start1 = start1.unsqueeze(1) # [N, 1]
    end1 = end1.unsqueeze(1)
    start2 = start2.unsqueeze(0) # [1, M]
    end2 = end2.unsqueeze(0)

    intersection_min = torch.max(start1, start2)
    intersection_max = torch.min(end1, end2)
    intersection = (intersection_max - intersection_min).clamp(min=0)

    union_min = torch.min(start1, start2)
    union_max = torch.max(end1, end2)
    union = (union_max - union_min).clamp(min=0) # 实际上是最小包围框

    area1 = (end1 - start1).clamp(min=0)
    area2 = (end2 - start2).clamp(min=0)
    
    real_union = area1 + area2 - intersection
    iou = intersection / (real_union + 1e-6)
    
    giou = iou - (union - real_union) / (union + 1e-6)
    return giou

# 占位函数
def split_and_pad(counts, tensor):
    pass 
def split_expand_and_pad(counts, target_counts, tensor):
    pass
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
def sample_outclass_neg(num_clips):
    pass
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_span=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 展平 batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_span = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # 连接所有 GT
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_span = torch.cat([v["spans"] for v in targets])

        # 计算 Cost Matrix
        # 1. Classification Cost
        cost_class = -out_prob[:, tgt_ids]

        # 2. Span L1 Cost
        cost_span = torch.cdist(out_span, tgt_span, p=1)

        # 3. GIoU Cost (这里需要引入 generalized_temporal_iou 函数)
        # 简单起见，这里假设有该函数，见下文 utils.py
        from utils import generalized_temporal_iou, span_cxw_to_xx
        cost_giou = -generalized_temporal_iou(span_cxw_to_xx(out_span), span_cxw_to_xx(tgt_span))

        # 总 Cost
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from utils import generalized_temporal_iou, span_cxw_to_xx

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_span=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_span = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2] (Start, End)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_span = torch.cat([v["spans"] for v in targets]) # [N_gt, 2] (Center, Width)

        if len(tgt_ids) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]

        # 1. Classification Cost
        cost_class = -out_prob[:, tgt_ids]

        # 2. Span L1 Cost
        # [修正] 必须将 GT (cx, w) 转为 (st, ed) 才能和 model output (st, ed) 计算距离
        tgt_span_xx = span_cxw_to_xx(tgt_span)
        cost_span = torch.cdist(out_span, tgt_span_xx, p=1)

        # 3. GIoU Cost
        # [修正] out_span 已经是 xx，不需要转；tgt_span 需要转为 xx
        cost_giou = -generalized_temporal_iou(out_span, tgt_span_xx)

        # Total Cost
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        if not torch.all(torch.isfinite(C)):
            C = torch.nan_to_num(C, nan=100.0, posinf=1000.0, neginf=-1000.0)

        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i].detach().numpy()) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
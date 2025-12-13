import torch
import torch.nn.functional as F
from torch import nn
from utils import span_cxw_to_xx, generalized_temporal_iou

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses, eos_coef=0.1):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef
        
        # 定义类别权重: 前景 1.0, 背景 eos_coef (0.1)
        empty_weight = torch.ones(2)
        empty_weight[1] = self.eos_coef 
        self.register_buffer('empty_weight', empty_weight)

    # -----------------------------------------------------------------------
    # [新增] 显著性损失计算函数 (参照 BAM-DETR 思路，但适配您的 BCE 逻辑)
    # -----------------------------------------------------------------------
    def loss_saliency(self, outputs, targets, indices, num_spans):
        """
        计算显著性损失 (Binary Cross Entropy)
        由于 Dataset 没有直接提供每帧的 0/1 标签，我们需要根据 GT Spans 动态生成。
        """
        # [注意] model.py 必须在 out 中输出 'saliency_scores' 和 'video_mask'
        if 'saliency_scores' not in outputs:
            return {'loss_saliency': 0}
            
        src_logits = outputs['saliency_scores']  # [B, T]
        video_mask = outputs['video_mask']       # [B, T]
        
        # 动态生成 Target Labels: 判断每一帧是否落在任意一个 GT Span 内
        B, T = src_logits.shape
        target_labels = torch.zeros_like(src_logits) # [B, T]
        
        for idx, t in enumerate(targets):
            # t['spans'] 是归一化的 [center, width]
            # 转换为 [start, end] 并映射到 [0, T] 索引
            spans_cxw = t['spans']
            
            # 还原到帧索引
            starts = (spans_cxw[:, 0] - spans_cxw[:, 1] / 2) * T
            ends = (spans_cxw[:, 0] + spans_cxw[:, 1] / 2) * T
            
            for s, e in zip(starts, ends):
                s_idx = max(0, int(s.floor()))
                e_idx = min(T, int(e.ceil()))
                if e_idx > s_idx:
                    target_labels[idx, s_idx:e_idx] = 1.0
        
        # 计算 BCE Loss (带 Logits，数值更稳定)
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_labels, reduction='none')
        
        # 仅计算有效帧 (Mask 为 True 的部分)
        loss_ce = (loss_ce * video_mask.float()).sum() / (video_mask.sum() + 1e-6)
        
        return {'loss_saliency': loss_ce}

    def loss_labels(self, outputs, targets, indices, num_spans):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # 1 is background
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device) 
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {'loss_label': loss_ce}

    def loss_spans(self, outputs, targets, indices, num_spans):
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
        loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))

        return {
            'loss_span': loss_span.sum() / num_spans,
            'loss_giou': loss_giou.sum() / num_spans
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # -----------------------------------------------------------------------
    # [修改] Forward 方法，跳过辅助层的 Saliency 计算
    # -----------------------------------------------------------------------
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_spans = sum(len(t["labels"]) for t in targets)
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)

        losses = {}
        for loss in self.losses:
            # 动态调用 loss 函数
            losses.update(self.__getattribute__(f'loss_{loss}')(outputs, targets, indices, num_spans))
            
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # [关键] 跳过 saliency，因为 aux_outputs 通常没有 saliency_scores
                    if loss == 'saliency': continue 
                    
                    l_dict = self.__getattribute__(f'loss_{loss}')(aux_outputs, targets, indices, num_spans)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
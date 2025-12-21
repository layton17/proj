import torch
import torch.nn.functional as F
from torch import nn
from utils import span_cxw_to_xx, generalized_temporal_iou

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses, eos_coef, span_loss_type="l1"):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_spans, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_labels': loss_ce}

    def loss_spans(self, outputs, targets, indices, num_spans):
        assert 'pred_spans' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # [修正] src_spans 已经是 (Start, End)
        src_spans = outputs['pred_spans'][idx]
        
        # [修正] target_spans 是 (Center, Width)，需要转换
        target_spans_cw = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_spans_xx = span_cxw_to_xx(target_spans_cw)
        
        # 计算 L1 Loss (Start vs Start, End vs End)
        loss_span = F.l1_loss(src_spans, target_spans_xx, reduction='none')
        
        # 计算 GIoU (直接传入两个 xx 格式)
        loss_giou = 1 - torch.diag(generalized_temporal_iou(src_spans, target_spans_xx))
        
        losses = {}
        losses['loss_span'] = loss_span.sum() / num_spans
        losses['loss_giou'] = loss_giou.sum() / num_spans
        return losses

    def loss_quality(self, outputs, targets, indices, num_spans):
        assert 'pred_quality' in outputs
        src_quality = outputs['pred_quality']
        src_spans = outputs['pred_spans']
        idx = self._get_src_permutation_idx(indices)
        
        # [修正] 这里的 IoU 计算也需要注意格式
        matched_spans = src_spans[idx] # (St, Ed)
        
        target_spans_cw = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_spans_xx = span_cxw_to_xx(target_spans_cw) # 转为 (St, Ed)
        
        # 手动计算 IoU 用于 Quality Target
        inter_min = torch.max(matched_spans[:, 0], target_spans_xx[:, 0])
        inter_max = torch.min(matched_spans[:, 1], target_spans_xx[:, 1])
        inter_len = (inter_max - inter_min).clamp(min=0)
        union_len = (matched_spans[:, 1] - matched_spans[:, 0]) + \
                    (target_spans_xx[:, 1] - target_spans_xx[:, 0]) - inter_len
        gt_iou = inter_len / (union_len + 1e-6)

        matched_quality = src_quality[idx].squeeze(-1)
        loss_quality = F.l1_loss(matched_quality.sigmoid(), gt_iou.detach(), reduction='sum')
        
        return {'loss_quality': loss_quality / num_spans}

    def loss_recfw(self, outputs, targets, indices, num_spans):
        if 'recfw_words_logit' not in outputs or outputs['recfw_words_logit'] is None:
            return {}
        logits = outputs['recfw_words_logit']
        mask_indices = outputs['masked_indices']
        if mask_indices.sum() == 0:
            return {'loss_recfw': torch.tensor(0.0, device=logits.device)}
        gt_words = torch.stack([t['words_id'] for t in targets])
        masked_logits = logits[mask_indices]
        masked_labels = gt_words[mask_indices]
        loss = F.cross_entropy(masked_logits, masked_labels)
        return {'loss_recfw': loss}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_spans, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'spans': self.loss_spans,
            'quality': self.loss_quality,
            'recfw': self.loss_recfw,
        }
        if loss not in loss_map: return {}
        return loss_map[loss](outputs, targets, indices, num_spans, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_spans = sum(len(t["labels"]) for t in targets)
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_spans = torch.clamp(num_spans, min=1).item()

        losses = {}
        current_losses = self.losses + ['quality', 'recfw']
        for loss in current_losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_spans))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'recfw': continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_spans)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
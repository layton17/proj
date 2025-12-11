import torch
import torch.nn.functional as F
from torch import nn
from utils import span_cxw_to_xx, generalized_temporal_iou

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_spans):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device) # 1 is background
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
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

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_spans = sum(len(t["labels"]) for t in targets)
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)

        losses = {}
        for loss in self.losses:
            losses.update(self.__getattribute__(f'loss_{loss}')(outputs, targets, indices, num_spans))
            
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.__getattribute__(f'loss_{loss}')(aux_outputs, targets, indices, num_spans)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
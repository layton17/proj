import torch
import numpy as np
import logging
from tqdm import tqdm
from utils import span_cxw_to_xx

logger = logging.getLogger(__name__)

def compute_temporal_iou(pred_spans, gt_spans):
    """
    计算预测片段与真实片段的 IoU
    pred_spans: [N, 2] (start, end)
    gt_spans: [M, 2] (start, end)
    """
    # 统一转为 Tensor 计算，避免类型混淆
    if not isinstance(pred_spans, torch.Tensor):
        pred_spans = torch.tensor(pred_spans)
    if not isinstance(gt_spans, torch.Tensor):
        gt_spans = torch.tensor(gt_spans)

    if pred_spans.ndim == 1: pred_spans = pred_spans.unsqueeze(0)
    if gt_spans.ndim == 1: gt_spans = gt_spans.unsqueeze(0)

    tmin = torch.max(pred_spans[:, 0].unsqueeze(1), gt_spans[:, 0].unsqueeze(0))
    tmax = torch.min(pred_spans[:, 1].unsqueeze(1), gt_spans[:, 1].unsqueeze(0))
    
    intersection = (tmax - tmin).clamp(min=0)
    
    pred_len = pred_spans[:, 1] - pred_spans[:, 0]
    gt_len = gt_spans[:, 1] - gt_spans[:, 0]
    
    union = pred_len.unsqueeze(1) + gt_len.unsqueeze(0) - intersection
    
    return intersection / (union + 1e-8)

def calculate_stats(predictions, iou_thresholds):
    """
    同时计算 Recall@1, Recall@5, mIoU
    """
    r1_results = {th: 0.0 for th in iou_thresholds}
    r5_results = {th: 0.0 for th in iou_thresholds}
    miou_sum = 0.0
    n_samples = len(predictions)
    
    if n_samples == 0:
        return {}

    for item in predictions:
        pred_scores = item['pred_scores']
        pred_spans = item['pred_spans']
        gt_spans = item['gt_spans']
        
        # 1. 获取 Top-K 预测
        # 排序
        sorted_idx = torch.argsort(pred_scores, descending=True)
        pred_spans_sorted = pred_spans[sorted_idx]
        
        # Top-1
        top1_span = pred_spans_sorted[0].unsqueeze(0)
        iou_top1 = compute_temporal_iou(top1_span, gt_spans) # [1, M]
        max_iou_r1 = iou_top1.max().item() if iou_top1.numel() > 0 else 0
        
        # Top-5
        k = min(5, len(pred_spans_sorted))
        top5_spans = pred_spans_sorted[:k]
        iou_top5 = compute_temporal_iou(top5_spans, gt_spans) # [K, M]
        max_iou_r5 = iou_top5.max().item() if iou_top5.numel() > 0 else 0
        
        # mIoU (Top-1 的 IoU)
        miou_sum += max_iou_r1
        
        # 统计 Recall
        for th in iou_thresholds:
            if max_iou_r1 >= th:
                r1_results[th] += 1.0
            if max_iou_r5 >= th:
                r5_results[th] += 1.0
                
    metrics = {}
    for th in iou_thresholds:
        metrics[f"R1@{th}"] = (r1_results[th] / n_samples) * 100
        metrics[f"R5@{th}"] = (r5_results[th] / n_samples) * 100
        
    metrics["mIoU"] = (miou_sum / n_samples) * 100
    return metrics

def calculate_mAP(predictions, iou_thresholds):
    """
    计算标准的 mAP (修正版: 剔除重复匹配)
    """
    ap_results = {th: [] for th in iou_thresholds}
    
    for item in predictions:
        pred_spans = item['pred_spans'].numpy() if isinstance(item['pred_spans'], torch.Tensor) else item['pred_spans']
        pred_scores = item['pred_scores'].numpy() if isinstance(item['pred_scores'], torch.Tensor) else item['pred_scores']
        gt_spans = item['gt_spans'].numpy() if isinstance(item['gt_spans'], torch.Tensor) else item['gt_spans']
        
        # 1. 按分数从高到低排序
        sorted_idx = np.argsort(pred_scores)[::-1]
        pred_spans_sorted = pred_spans[sorted_idx]
        
        num_gt = len(gt_spans)
        if num_gt == 0:
            for th in iou_thresholds:
                ap_results[th].append(0.0)
            continue

        # 计算所有预测框与所有 GT 的 IoU 矩阵
        iou_mat = compute_temporal_iou(pred_spans_sorted, gt_spans).numpy()
        
        for th in iou_thresholds:
            gt_matched = np.zeros(num_gt, dtype=bool)
            tp = np.zeros(len(pred_spans_sorted))
            fp = np.zeros(len(pred_spans_sorted))
            
            for i in range(len(pred_spans_sorted)):
                # 找到该预测框重叠最大的那个 GT
                if iou_mat.shape[1] > 0:
                    max_iou_idx = np.argmax(iou_mat[i])
                    max_iou = iou_mat[i][max_iou_idx]
                else:
                    max_iou = 0
                
                if max_iou >= th:
                    if not gt_matched[max_iou_idx]:
                        # True Positive: 首次匹配到该 GT
                        tp[i] = 1.0
                        gt_matched[max_iou_idx] = True
                    else:
                        # False Positive: 该 GT 已经被分更高的框匹配了
                        fp[i] = 1.0
                else:
                    # False Positive: IoU 不够
                    fp[i] = 1.0
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            recall = tp_cumsum / num_gt
            
            # 简单的 AP 计算: Sum(Precision * Hit) / Num_GT
            ap = np.sum(precision * tp) / num_gt
            ap_results[th].append(ap)
            
    mAP_scores = {f"mAP@{th}": np.mean(vals) * 100 for th, vals in ap_results.items()}
    
    avg_map = np.mean([mAP_scores[f"mAP@{th}"] for th in iou_thresholds if th >= 0.5])
    mAP_scores["mAP@Avg"] = avg_map
    
    return mAP_scores

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    results = []
    
    logger.info("Evaluating...")
    for batch in tqdm(data_loader, desc="Inference"):
        video_feat = batch['video_feat'].to(device)
        video_mask = batch['video_mask'].to(device)
        words_id = batch['words_id'].to(device)
        words_mask = batch['words_mask'].to(device)
        
        outputs = model(video_feat, video_mask, words_id, words_mask, is_training=False)
        
        pred_spans = span_cxw_to_xx(outputs['pred_spans'].cpu()) # [B, Nq, 2]
        pred_logits = outputs['pred_logits'].cpu()
        pred_scores = torch.softmax(pred_logits, dim=-1)[..., 0] 
        
        targets = batch['targets']
        for i, target in enumerate(targets):
            duration = target['duration']
            
            # [关键修复] GT 也要转换格式
            gt_spans_xx = span_cxw_to_xx(target['spans'])
            
            results.append({
                "video_id": target['video_id'],
                "pred_spans": pred_spans[i] * duration,
                "pred_scores": pred_scores[i],
                "gt_spans": gt_spans_xx * duration # 使用转换后的 GT
            })
            
    if not results:
        return {}

    # 定义阈值
    recall_thds = [0.5, 0.7]
    map_thds = [0.5, 0.75]
    map_thds_full = [round(x * 0.05, 2) for x in range(10, 20)] # 0.5 - 0.95
    
    # 计算所有指标
    metrics = {}
    metrics.update(calculate_stats(results, recall_thds)) # 计算 R1, R5, mIoU
    metrics.update(calculate_mAP(results, map_thds_full)) # 计算 mAP
    
    # 打印日志 (增加 R5 和 mIoU 的显示)
    logger.info("------------------------------------------------")
    logger.info(f"R1@0.5: {metrics['R1@0.5']:.2f} | R1@0.7: {metrics['R1@0.7']:.2f}")
    logger.info(f"R5@0.5: {metrics['R5@0.5']:.2f} | R5@0.7: {metrics['R5@0.7']:.2f}")
    logger.info(f"mAP@0.5: {metrics['mAP@0.5']:.2f} | mAP@0.75: {metrics['mAP@0.75']:.2f} | mAP@Avg: {metrics['mAP@Avg']:.2f}")
    logger.info(f"mIoU: {metrics['mIoU']:.2f}")
    logger.info("------------------------------------------------")
    
    return metrics
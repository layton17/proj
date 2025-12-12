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
    if pred_spans.ndim == 1: pred_spans = pred_spans.unsqueeze(0)
    if gt_spans.ndim == 1: gt_spans = gt_spans.unsqueeze(0)

    tmin = torch.max(pred_spans[:, 0].unsqueeze(1), gt_spans[:, 0].unsqueeze(0))
    tmax = torch.min(pred_spans[:, 1].unsqueeze(1), gt_spans[:, 1].unsqueeze(0))
    
    intersection = (tmax - tmin).clamp(min=0)
    
    pred_len = pred_spans[:, 1] - pred_spans[:, 0]
    gt_len = gt_spans[:, 1] - gt_spans[:, 0]
    
    union = pred_len.unsqueeze(1) + gt_len.unsqueeze(0) - intersection
    
    return intersection / (union + 1e-8)

def calculate_recall(predictions, iou_thresholds):
    """计算 Recall@1"""
    recall_results = {th: 0.0 for th in iou_thresholds}
    n_samples = len(predictions)
    
    for item in predictions:
        # 获取 Top-1 预测 (分数最高的)
        best_idx = torch.argmax(item['pred_scores'])
        top1_span = item['pred_spans'][best_idx].unsqueeze(0)
        gt_spans = item['gt_spans']
        
        # 计算与所有 GT 的 IoU
        iou = compute_temporal_iou(top1_span, gt_spans) # [1, M]
        max_iou = iou.max().item()
        
        for th in iou_thresholds:
            if max_iou >= th:
                recall_results[th] += 1.0
                
    return {f"R1@{th}": (count / n_samples) * 100 for th, count in recall_results.items()}

def calculate_mAP(predictions, iou_thresholds):
    """计算 mAP"""
    ap_results = {th: [] for th in iou_thresholds}
    
    for item in predictions:
        pred_spans = item['pred_spans']
        pred_scores = item['pred_scores']
        gt_spans = item['gt_spans']
        
        # 按分数排序
        sorted_idx = torch.argsort(pred_scores, descending=True)
        pred_spans_sorted = pred_spans[sorted_idx]
        
        iou_mat = compute_temporal_iou(pred_spans_sorted, gt_spans)
        
        for th in iou_thresholds:
            # 只要命中任意一个 GT 就算 True Positive
            is_hit = (iou_mat > th).any(dim=1).numpy()
            
            if not is_hit.any():
                ap = 0.0
            else:
                # 计算 Average Precision
                tp = np.cumsum(is_hit)
                precision = tp / (np.arange(len(is_hit)) + 1)
                ap = precision[is_hit].sum() / len(gt_spans)
            
            ap_results[th].append(ap)
            
    mAP_scores = {f"mAP@{th}": np.mean(vals) * 100 for th, vals in ap_results.items()}
    
    # 计算平均 mAP (Avg)
    avg_map = np.mean([mAP_scores[f"mAP@{th}"] for th in iou_thresholds if th >= 0.5])
    mAP_scores["mAP@Avg"] = avg_map
    
    return mAP_scores

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    results = []
    
    logger.info("Evaluating...")
    for batch in tqdm(data_loader, desc="Inference"):
        # 1. 数据准备
        video_feat = batch['video_feat'].to(device)
        video_mask = batch['video_mask'].to(device)
        words_id = batch['words_id'].to(device)
        words_mask = batch['words_mask'].to(device)
        
        # 2. 推理
        outputs = model(video_feat, video_mask, words_id, words_mask, is_training=False)
        
        # 3. 解析结果
        pred_spans = span_cxw_to_xx(outputs['pred_spans'].cpu()) # [B, Nq, 2]
        pred_logits = outputs['pred_logits'].cpu()
        # 假设第 0 类是前景类 (Foreground)
        pred_scores = torch.softmax(pred_logits, dim=-1)[..., 0] 
        
        targets = batch['targets']
        for i, target in enumerate(targets):
            duration = target['duration']
            
            results.append({
                "video_id": target['video_id'],
                "pred_spans": pred_spans[i] * duration, # 还原为真实时间
                "pred_scores": pred_scores[i],
                "gt_spans": target['spans'] * duration
            })
            
    # 4. 计算指标
    if not results:
        return {}

    recall_thds = [0.5, 0.7]
    map_thds = [0.5, 0.75]
    map_thds_full = [round(x * 0.05, 2) for x in range(10, 20)] # 0.5 - 0.95
    
    metrics = {}
    metrics.update(calculate_recall(results, recall_thds))
    metrics.update(calculate_mAP(results, map_thds_full))
    
    # 打印日志
    logger.info("------------------------------------------------")
    logger.info(f"R1@0.5: {metrics['R1@0.5']:.2f} | R1@0.7: {metrics['R1@0.7']:.2f}")
    logger.info(f"mAP@0.5: {metrics['mAP@0.5']:.2f} | mAP@0.75: {metrics['mAP@0.75']:.2f} | mAP@Avg: {metrics['mAP@Avg']:.2f}")
    logger.info("------------------------------------------------")
    
    return metrics
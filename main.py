import torch
import os
import time
import random
import numpy as np
import logging
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入自定义模块
from config import get_args_parser
from dataset import VideoDataset, collate_fn
from model import build_model
from text_encoder import CLIPTextEncoder, GloveTextEncoder
from matcher import HungarianMatcher
from criterion import SetCriterion
from engine import evaluate

# 配置基础 Logger (控制台输出)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    criterion.train()
    
    total_loss = 0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} Train")
    
    for i, batch in pbar:
        video_feat = batch['video_feat'].to(device)
        video_mask = batch['video_mask'].to(device)
        words_id = batch['words_id'].to(device) # 模型输入 (可能是 Feature 也可能是 ID)
        txt_ids = batch['txt_ids'].to(device)   # Loss 标签 (一定是 ID)
        words_mask = batch['words_mask'].to(device)
        
        # 构建 targets 列表
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch['targets']]

        # [关键修复] 将 txt_ids (整数) 注入到 targets 中供 Loss 使用
        # 覆盖 targets 原有的 key (如果有的话)，确保 criterion 拿到的是 ID
        for t, t_id in zip(targets, txt_ids):
            t['words_id'] = t_id

        outputs = model(video_feat, video_mask, words_id, words_mask, is_training=True)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # 处理 Fallback Loss (如果 criterion 没算)
        if 'loss_recfw' in loss_dict and 'loss_recfw' not in weight_dict:
             losses += 0.5 * loss_dict['loss_recfw']
        elif "recfw_words_logit" in outputs and 'loss_recfw' not in loss_dict:
            pred_logits = outputs["recfw_words_logit"]
            gt_ids = txt_ids # 使用 txt_ids
            loss_rec = torch.nn.functional.cross_entropy(
                pred_logits.transpose(1, 2), 
                gt_ids, 
                ignore_index=0 
            )
            losses += 0.5 * loss_rec

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses.item()
        if i % 10 == 0:
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

def main(args):
    device = torch.device(args.device)
    set_seed(args.seed)
    
    # 1. 创建 Checkpoint 保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 日志分离逻辑
    exp_name = os.path.basename(os.path.normpath(args.save_dir))
    log_dir = os.path.join("logs", exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger('')
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    root_logger.addHandler(file_handler)
    
    logger.info(f"✅ Log file created at: {log_path}")
    logger.info(f"✅ Checkpoints will be saved to: {args.save_dir}")
    logger.info(f"Initializing Dataset: {args.dataset_name}")
    
    # -----------------------------------------------------------
    # 2. 加载数据集
    # -----------------------------------------------------------
    dataset_train = VideoDataset(args, is_training=True)
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Train dataset size: {len(dataset_train)}")

    test_anno_path = args.annotation_path.replace("train.txt", "test.txt")
    dataloader_val = None
    if os.path.exists(test_anno_path):
        logger.info(f"Loading Validation Dataset from: {test_anno_path}")
        args_val = type(args)(**vars(args)) 
        args_val.annotation_path = test_anno_path
        dataset_val = VideoDataset(args_val, is_training=False)
        
        if args.text_encoder_type == 'glove':
            dataset_val.word2idx = dataset_train.word2idx
            dataset_val.vocab = dataset_train.vocab
            
        dataloader_val = DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False, 
            collate_fn=collate_fn, num_workers=4, pin_memory=True
        )
    else:
        logger.warning(f"Validation file not found at {test_anno_path}")

    # -----------------------------------------------------------
    # 3. 初始化 Text Encoder
    # -----------------------------------------------------------
    text_encoder = None
    if args.text_encoder_type == 'clip':
        logger.info("Building CLIP Text Encoder...")
        text_encoder = CLIPTextEncoder(
            embed_dim=args.t_feat_dim,
            context_length=args.max_q_l,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12
        )
        
        if hasattr(args, 'clip_weight_path') and args.clip_weight_path:
            logger.info(f"Loading CLIP weights from {args.clip_weight_path}")
            try:
                loaded_obj = torch.load(args.clip_weight_path, map_location='cpu')
            except Exception:
                loaded_obj = torch.jit.load(args.clip_weight_path, map_location='cpu')
                
            if hasattr(loaded_obj, 'state_dict'):
                state_dict = loaded_obj.state_dict()
            else:
                state_dict = loaded_obj

            if 'positional_embedding' in state_dict:
                ckpt_len = state_dict['positional_embedding'].shape[0]
                model_len = text_encoder.positional_embedding.shape[0]
                if ckpt_len > model_len:
                    logger.info(f"⚠️ Truncating positional embedding from {ckpt_len} to {model_len}")
                    state_dict['positional_embedding'] = state_dict['positional_embedding'][:model_len, :]

            text_encoder.load_state_dict(state_dict, strict=False)

    elif args.text_encoder_type == 'glove':
        logger.info("Building GloVe Text Encoder...")
        if not hasattr(dataset_train, 'vocab'):
            raise AttributeError("Dataset needs 'vocab' attribute for GloVe mode.")
        
        text_encoder = GloveTextEncoder(
            vocab_list=dataset_train.vocab, 
            glove_path=args.glove_path
        )
    
    if text_encoder is not None:
        text_encoder.to(device)
    elif args.text_encoder_type != 'precomputed': 
        raise ValueError("Text Encoder failed to initialize.")

    # -----------------------------------------------------------
    # 4. 构建模型
    # -----------------------------------------------------------
    logger.info("Building Model...")
    model = build_model(args)
    # 如果 build_model 内部没有赋值 text_encoder，这里手动赋一次
    if hasattr(model, 'text_encoder') and model.text_encoder is None:
        model.text_encoder = text_encoder
    model.to(device)

    # -----------------------------------------------------------
    # 5. 匹配器和损失
    # -----------------------------------------------------------
    matcher = HungarianMatcher(cost_class=args.label_loss_coef, 
                               cost_span=args.span_loss_coef, 
                               cost_giou=args.giou_loss_coef)
    
    weight_dict = {'loss_label': args.label_loss_coef, 
                   'loss_span': args.span_loss_coef, 
                   'loss_giou': args.giou_loss_coef}
    
    # 增加 Quality Loss 和 Recfw Loss 的默认权重（如果 config 里没设置）
    if 'quality' not in weight_dict:
        weight_dict['loss_quality'] = 1.0 # 建议权重
    if 'recfw' not in weight_dict:
        weight_dict['loss_recfw'] = 0.25   # 建议权重

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 注册所有需要的 loss
    losses = ['labels', 'spans', 'quality', 'recfw', 'saliency'] 
    # 注意：saliency 暂时没用到核心逻辑中，可以根据需要加回
    
    criterion = SetCriterion(matcher, weight_dict, losses=losses, eos_coef=args.eos_coef)
    criterion.to(device)

    # -----------------------------------------------------------
    # 6. 优化器
    # -----------------------------------------------------------
    # 将 quality_proj 和 masked_token 等新参数加入优化器
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad], "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # 余弦退火调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,     # 自动适配总轮次
        eta_min=args.lr * 0.01 # 最小学习率 (例如 1e-6)
    )
    # -----------------------------------------------------------
    # 7. 训练循环
    # -----------------------------------------------------------
    logger.info(f"Start training for {args.epochs} epochs.")
    
    best_r1_07 = 0.0 
    
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch)
        
        # 验证与最佳模型保存
        if dataloader_val is not None:
            metrics = evaluate(model, dataloader_val, device)
            
            # 使用更严格的指标 R1@0.7 作为保存标准
            current_metric = metrics.get('R1@0.7', 0)
            if current_metric > best_r1_07:
                best_r1_07 = current_metric
                best_path = os.path.join(args.save_dir, "checkpoint_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, best_path)
                logger.info(f"⭐ New Best Model! R1@0.7: {best_r1_07:.2f}%")

        ckpt_path = os.path.join(args.save_dir, "checkpoint_last.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }, ckpt_path)

if __name__ == '__main__':
    parser = get_args_parser()
    if not any(action.dest == 'text_encoder_type' for action in parser._actions):
        parser.add_argument('--text_encoder_type', default='clip', choices=['clip', 'glove'], help='Type of text encoder')
    if not any(action.dest == 'glove_path' for action in parser._actions):
        parser.add_argument('--glove_path', default='', type=str, help='Path to glove vectors')
    if not any(action.dest == 'clip_weight_path' for action in parser._actions):
        parser.add_argument('--clip_weight_path', default='', type=str, help='Path to pretrained CLIP weights')
    
    args = parser.parse_args()
    
    # [关键修复] 在 args 定义后，main 执行前设置默认参数
    if not hasattr(args, 'vocab_size'):
        args.vocab_size = 49408
    if not hasattr(args, 'rec_fw'):
        args.rec_fw = True
        
    main(args)
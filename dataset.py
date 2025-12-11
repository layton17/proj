import os
import time
import json
import random
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入自定义模块
from config import get_args_parser
from dataset import VideoDataset, collate_fn
from model import build_model
from text_encoder import CLIPTextEncoder, GloveTextEncoder
from matcher import HungarianMatcher
from criterion import SetCriterion

# 设置日志
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
    # 使用 tqdm 显示进度条
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}")
    
    for i, batch in pbar:
        # 1. 数据移至 GPU
        video_feat = batch['video_feat'].to(device)
        video_mask = batch['video_mask'].to(device)
        words_id = batch['words_id'].to(device)
        words_mask = batch['words_mask'].to(device)
        
        # Target 是 list of dicts，需要逐个处理 tensor
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch['targets']]

        # 2. Forward
        # 开启 is_training=True 以启用 MESM 的重构 Loss (如果模型中有定义)
        outputs = model(video_feat, video_mask, words_id, words_mask, is_training=True)
        
        # 3. Loss 计算
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # 加权求和
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # 4. Backward
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = losses.item()
        total_loss += loss_val
        
        # 更新进度条上的 Loss 显示
        if i % 10 == 0:
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

def main(args):
    # 1. 环境设置
    device = torch.device(args.device)
    set_seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger.info(f"Initializing Dataset: {args.dataset_name}")
    
    # 2. 数据加载 (Dataset & Dataloader)
    # 必须先加载 Dataset，因为 GloVe 需要词表信息
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

    # 3. 初始化文本编码器 (Text Encoder)
    text_encoder = None
    if args.text_encoder_type == 'clip':
        logger.info("Building CLIP Text Encoder...")
        text_encoder = CLIPTextEncoder(
            embed_dim=args.t_feat_dim,      # e.g. 512
            context_length=args.max_q_l,    # e.g. 77
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12
        )
        # TODO: 加载预训练权重
        if hasattr(args, 'clip_weight_path') and args.clip_weight_path:
             logger.info(f"Loading CLIP weights from {args.clip_weight_path}")
             state_dict = torch.load(args.clip_weight_path, map_location='cpu')
             text_encoder.load_state_dict(state_dict, strict=False)

    elif args.text_encoder_type == 'glove':
        logger.info("Building GloVe Text Encoder...")
        # 确保 Dataset 有 vocab 属性 (在 dataset.py 中实现构建逻辑)
        if not hasattr(dataset_train, 'vocab'):
            raise AttributeError("Dataset needs 'vocab' attribute for GloVe.")
        
        vocab_list = dataset_train.vocab 
        text_encoder = GloveTextEncoder(vocab_list, args.glove_path)
    
    if text_encoder is not None:
        text_encoder.to(device)

    # 4. 构建模型 (Model)
    logger.info("Building Model...")
    model = build_model(args)
    
    # 手动注入文本编码器
    model.text_encoder = text_encoder 
    model.to(device)

    # 5. 匹配器与损失函数 (Matcher & Criterion)
    matcher = HungarianMatcher(
        cost_class=args.label_loss_coef, 
        cost_span=args.span_loss_coef, 
        cost_giou=args.giou_loss_coef
    )
    
    weight_dict = {
        'loss_label': args.label_loss_coef, 
        'loss_span': args.span_loss_coef, 
        'loss_giou': args.giou_loss_coef
    }
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher, weight_dict, losses=['labels', 'spans'])
    criterion.to(device)

    # 6. 优化器 (Optimizer)
    # 区分参数组：通常不训练 Text Encoder (除非微调)
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad],
            "lr": args.lr
        },
        # 如果需要微调 Text Encoder，取消下面注释
        # {
        #     "params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad],
        #     "lr": args.lr * 0.1
        # },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # 7. 训练循环
    logger.info(f"Start training for {args.epochs} epochs.")
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, ckpt_path)
            logger.info(f"Model saved to {ckpt_path}")

if __name__ == '__main__':
    parser = get_args_parser()
    
    # 补充必要的命令行参数定义，如果 config.py 中没有包含它们
    parser.add_argument('--text_encoder_type', default='clip', choices=['clip', 'glove'], help='Type of text encoder')
    parser.add_argument('--glove_path', default='', type=str, help='Path to glove vectors if using glove')
    parser.add_argument('--clip_weight_path', default='', type=str, help='Path to pretrained CLIP weights')
    
    args = parser.parse_args()
    main(args)
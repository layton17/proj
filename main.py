import torch
import os
import time
import random
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
# [新增] 引入验证函数
from engine import evaluate

# 配置 Logger
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
        words_id = batch['words_id'].to(device)
        words_mask = batch['words_mask'].to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch['targets']]

        # Forward
        outputs = model(video_feat, video_mask, words_id, words_mask, is_training=True)
        
        # Loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Backward
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
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger.info(f"Initializing Dataset: {args.dataset_name}")
    
    # -----------------------------------------------------------
    # 1. 加载数据集 (Train & Val)
    # -----------------------------------------------------------
    # 训练集
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

    # [新增] 验证集 (自动寻找 test.txt)
    # 假设 train 路径是 .../charades_sta_train.txt，则 test 路径推断为 .../charades_sta_test.txt
    test_anno_path = args.annotation_path.replace("train.txt", "test.txt")
    dataloader_val = None
    
    if os.path.exists(test_anno_path):
        logger.info(f"Loading Validation Dataset from: {test_anno_path}")
        # 临时修改 args 里的路径来初始化验证集 Dataset
        args_val = type(args)(**vars(args)) # 浅拷贝 args
        args_val.annotation_path = test_anno_path
        
        dataset_val = VideoDataset(args_val, is_training=False)
        
        # 关键：如果是 Glove，验证集必须使用训练集的 vocab
        if args.text_encoder_type == 'glove':
            dataset_val.word2idx = dataset_train.word2idx
            dataset_val.vocab = dataset_train.vocab
            
        dataloader_val = DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False, 
            collate_fn=collate_fn, num_workers=4, pin_memory=True
        )
        logger.info(f"Val dataset size: {len(dataset_val)}")
    else:
        logger.warning(f"Validation file not found at {test_anno_path}. Skipping validation.")

    # -----------------------------------------------------------
    # 2. 初始化 Text Encoder
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
             state_dict = torch.load(args.clip_weight_path, map_location='cpu')
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
    else:
        raise ValueError("Text Encoder failed to initialize.")

    # -----------------------------------------------------------
    # 3. 构建模型
    # -----------------------------------------------------------
    logger.info("Building Model...")
    model = build_model(args)
    model.text_encoder = text_encoder
    model.to(device)

    # -----------------------------------------------------------
    # 4. 匹配器和损失
    # -----------------------------------------------------------
    matcher = HungarianMatcher(cost_class=args.label_loss_coef, 
                               cost_span=args.span_loss_coef, 
                               cost_giou=args.giou_loss_coef)
    
    weight_dict = {'loss_label': args.label_loss_coef, 
                   'loss_span': args.span_loss_coef, 
                   'loss_giou': args.giou_loss_coef}
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher, weight_dict, losses=['labels', 'spans'])
    criterion.to(device)

    # -----------------------------------------------------------
    # 5. 优化器
    # -----------------------------------------------------------
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad], "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------------------------------------------
    # 6. 训练循环
    # -----------------------------------------------------------
    logger.info(f"Start training for {args.epochs} epochs.")
    
    best_r1_07 = 0.0 # 用于保存最佳模型
    
    for epoch in range(args.epochs):
        # 1. 训练
        train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch)
        
        # 2. 保存 Checkpoint (定期)
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, ckpt_path)
        
        # 3. 验证 (每个 Epoch 结束)
        if dataloader_val is not None:
            logger.info(f"Running validation at epoch {epoch}...")
            metrics = evaluate(model, dataloader_val, device)
            
            # 记录最佳模型 (以 R1@0.7 为标准)
            current_r1 = metrics.get('R1@0.7', 0.0)
            if current_r1 > best_r1_07:
                best_r1_07 = current_r1
                best_path = os.path.join(args.save_dir, "checkpoint_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, best_path)
                logger.info(f"🔥🔥 Best Model Saved! R1@0.7: {best_r1_07:.2f}%")

if __name__ == '__main__':
    parser = get_args_parser()
    # 补充命令行参数
    if not any(action.dest == 'text_encoder_type' for action in parser._actions):
        parser.add_argument('--text_encoder_type', default='clip', choices=['clip', 'glove'], help='Type of text encoder')
    if not any(action.dest == 'glove_path' for action in parser._actions):
        parser.add_argument('--glove_path', default='', type=str, help='Path to glove vectors')
    if not any(action.dest == 'clip_weight_path' for action in parser._actions):
        parser.add_argument('--clip_weight_path', default='', type=str, help='Path to pretrained CLIP weights')
    
    args = parser.parse_args()
    main(args)
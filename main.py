import torch
from torch.utils.data import DataLoader
from config import get_args_parser
from dataset import VideoDataset, collate_fn
from model import build_model # 从你之前生成的model.py导入
from text_encoder import CLIPTextEncoder, GloveTextEncoder
from matcher import HungarianMatcher
from criterion import SetCriterion
import time

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    criterion.train()
    
    total_loss = 0
    for i, batch in enumerate(data_loader):
        # 数据移至 GPU
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
            print(f"Epoch [{epoch}][{i}/{len(data_loader)}] Loss: {losses.item():.4f}")

def main(args):
    device = torch.device(args.device)

    # 1. 文本编码器 (如果是独立的)
    # 假设使用简单 Embedding, 实际可以用 model 内部的 text_encoder
    # model.py 里 build_model 需要一个 text_encoder
    # 根据配置选择编码器
    if args.text_encoder_type == 'clip':
        # 初始化 CLIP (参数需与预训练模型一致)
        text_encoder = CLIPTextEncoder(
            embed_dim=512,
            context_length=args.max_q_l, # 或 77
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12
        )
        # TODO: 这里需要加载预训练权重，例如从 'ViT-B/32'
        # state_dict = torch.load("path/to/clip_weights.pth")
        # text_encoder.load_state_dict(state_dict)
        
    elif args.text_encoder_type == 'glove':
        # 假设你有一个 vocab 列表和 glove 路径
        # vocab_list = dataset.vocab 
        # text_encoder = GloveTextEncoder(vocab_list, args.glove_path)
        pass 
    
    text_encoder.to(device)
    
    # 2. 构建模型
    model = build_model(args)
    model.text_encoder = text_encoder # 手动注入或在 build_model 里处理
    model.to(device)

    # 3. 匹配器和损失
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

    # 4. 数据加载
    dataset_train = VideoDataset(args, is_training=True)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                                  shuffle=True, collate_fn=collate_fn)

    # 5. 优化器
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": args.lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # 6. 训练循环
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch)
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{args.save_dir}/checkpoint_{epoch}.pth")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
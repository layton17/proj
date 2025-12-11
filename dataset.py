import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random

# 尝试导入 CLIP，如果用户没装，需要提示安装
try:
    import clip
except ImportError:
    clip = None

class VideoDataset(Dataset):
    def __init__(self, args, is_training=True):
        self.args = args
        self.is_training = is_training
        self.feature_dir = args.feature_dir
        self.max_v_l = args.max_v_l
        self.max_q_l = args.max_q_l
        
        # 1. 加载标注文件
        anno_path = args.annotation_path
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found at {anno_path}")
            
        print(f"Loading annotations from {anno_path}...")
        with open(anno_path, 'r') as f:
            self.annotations = json.load(f)
            
        # 2. 文本处理准备 (Tokenizer / Vocab)
        self.text_encoder_type = args.text_encoder_type
        self.vocab = None
        
        if self.text_encoder_type == 'glove':
            # GloVe 模式需要构建词表
            self.word2idx = self._build_vocab()
            self.vocab = list(self.word2idx.keys()) # 传给 GloveTextEncoder 用
            print(f"Vocab size: {len(self.vocab)}")
        elif self.text_encoder_type == 'clip':
            if clip is None:
                print("Warning: 'clip' module not found. Please install it via: pip install git+https://github.com/openai/CLIP.git")
                # 这里为了演示，如果不装CLIP，会报错。实际使用必须装。

    def _build_vocab(self):
        """仅用于 GloVe 模式：构建简单的词表"""
        word_counts = {}
        for sample in self.annotations:
            sentence = sample['sentence'].lower().replace('.', '').replace(',', '')
            for w in sentence.split():
                word_counts[w] = word_counts.get(w, 0) + 1
        
        # 构建映射: <PAD>=0, <UNK>=1
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for w, count in word_counts.items():
            if count >= 1: # 可以在这里过滤低频词
                word2idx[w] = len(word2idx)
        return word2idx

    def _load_video_feature(self, vid_id):
        """读取 .npy 视频特征并进行采样/填充"""
        feat_path = os.path.join(self.feature_dir, f"{vid_id}.npy")
        
        if not os.path.exists(feat_path):
            # 如果特征不存在，生成一个随机的代替 (仅用于调试代码流程，实际应报错)
            # print(f"Warning: Feature for {vid_id} not found, using random.")
            # return torch.randn(self.max_v_l, self.args.v_feat_dim), torch.ones(self.max_v_l, dtype=torch.bool)
            raise FileNotFoundError(f"Video feature not found: {feat_path}")

        feat = np.load(feat_path) # [T, D]
        # 也可以在这里把 float16 转 float32
        feat = torch.from_numpy(feat).float()
        
        seq_len = feat.shape[0]
        target_len = self.max_v_l
        
        if seq_len >= target_len:
            # 采样: 简单均匀采样
            idxs = torch.linspace(0, seq_len - 1, target_len).long()
            feat = feat[idxs]
            mask = torch.ones(target_len, dtype=torch.bool)
        else:
            # 填充: 补零
            pad_len = target_len - seq_len
            padding = torch.zeros(pad_len, feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
            mask = torch.cat([torch.ones(seq_len, dtype=torch.bool), 
                              torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            
        return feat, mask

    def _process_text(self, sentence):
        """将句子转换为 Token IDs"""
        if self.text_encoder_type == 'clip':
            # CLIP 的 Tokenizer 会自动处理截断和特殊字符
            # context_length=77 是 CLIP 默认，但我们参数里可能有 max_q_l
            # 这里我们强制截断到 max_q_l
            try:
                tokens = clip.tokenize(sentence, truncate=True).squeeze(0) # [77]
            except NameError:
                raise RuntimeError("Please install openai-clip to use CLIP tokenizer.")
            
            if self.max_q_l < 77:
                tokens = tokens[:self.max_q_l]
            
            # 创建 mask (CLIP 的 padding id 通常是 0，但也可能是其他，这里简化处理)
            # CLIP tokenize 后，结束符之后都是 0
            mask = (tokens != 0)
            return tokens, mask

        elif self.text_encoder_type == 'glove':
            # 简单的空格分词
            words = sentence.lower().replace('.', '').replace(',', '').split()
            words = words[:self.max_q_l]
            
            ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
            # Padding
            pad_len = self.max_q_l - len(ids)
            mask = [True] * len(ids) + [False] * pad_len
            ids = ids + [self.word2idx["<PAD>"]] * pad_len
            
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        vid_id = sample['video_id']
        duration = sample['duration']
        sentence = sample['sentence']
        
        # 1. Video
        video_feat, video_mask = self._load_video_feature(vid_id)
        
        # 2. Text
        words_id, words_mask = self._process_text(sentence)
        
        # 3. Target (Spans)
        # 将 [start, end] 转换为 [center, width] 并归一化
        gt_spans = []
        if 'timestamps' in sample:
            for st, ed in sample['timestamps']:
                # 确保时间在 duration 范围内
                st = max(0, min(st, duration))
                ed = max(0, min(ed, duration))
                if ed < st: continue 

                # 归一化
                center = (st + ed) / 2.0 / duration
                width = (ed - st) / duration
                
                # 这里的格式必须是 [center, width]
                gt_spans.append([center, width])
        
        if len(gt_spans) == 0:
            # 防止空数据报错，添加一个 dummy span (例如整个视频)
            gt_spans.append([0.5, 1.0])
            
        target = {
            "spans": torch.tensor(gt_spans, dtype=torch.float32),
            "labels": torch.zeros(len(gt_spans), dtype=torch.long), # 0 为前景
            "video_id": vid_id,
            "duration": duration,
            "raw_txt": sentence 
        }

        return {
            "video_feat": video_feat,
            "video_mask": video_mask,
            "words_id": words_id,
            "words_mask": words_mask,
            "target": target
        }

def collate_fn(batch):
    batch_data = {
        "video_feat": torch.stack([b['video_feat'] for b in batch]),
        "video_mask": torch.stack([b['video_mask'] for b in batch]),
        "words_id": torch.stack([b['words_id'] for b in batch]),
        "words_mask": torch.stack([b['words_mask'] for b in batch]),
        "targets": [b['target'] for b in batch] # List of dicts
    }
    return batch_data
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import logging

try:
    import clip
except ImportError:
    clip = None

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(self, args, is_training=True):
        self.args = args
        self.is_training = is_training
        self.feature_dir = args.feature_dir
        self.max_v_l = args.max_v_l
        self.max_q_l = args.max_q_l
        
        # [修改点 1] 支持 .txt 格式标注
        anno_path = args.annotation_path
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found at {anno_path}")
            
        logger.info(f"Loading annotations from {anno_path}...")
        if anno_path.endswith('.txt'):
            self.annotations = self._load_txt(anno_path)
        else:
            with open(anno_path, 'r') as f:
                self.annotations = json.load(f)
            
        # 文本处理准备 (Tokenizer / Vocab)
        self.text_encoder_type = args.text_encoder_type
        self.vocab = None
        
        if self.text_encoder_type == 'glove':
            self.word2idx = self._build_vocab()
            self.vocab = list(self.word2idx.keys())
            logger.info(f"Vocab size: {len(self.vocab)}")
        elif self.text_encoder_type == 'clip':
            if clip is None:
                logger.warning("CLIP module not found. Please install openai-clip.")

    def _load_txt(self, path):
        """解析 Charades-STA 原始 txt 格式"""
        data = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # 格式: AO8RW 0.0 6.9##a person is putting a book on a shelf.
                try:
                    meta, sentence = line.split('##')
                    parts = meta.split()
                    vid_id = parts[0]
                    start = float(parts[1])
                    end = float(parts[2])
                    
                    data.append({
                        "video_id": vid_id,
                        # 注意：TXT 中没有 duration，我们将在 __getitem__ 中根据特征长度计算
                        "timestamps": [[start, end]],
                        "sentence": sentence.strip()
                    })
                except Exception as e:
                    logger.warning(f"Skipping line: {line}. Error: {e}")
        return data

    def _build_vocab(self):
        """构建简单的词表"""
        word_counts = {}
        for sample in self.annotations:
            sentence = sample['sentence'].lower().replace('.', '').replace(',', '')
            for w in sentence.split():
                word_counts[w] = word_counts.get(w, 0) + 1
        
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for w, count in word_counts.items():
            if count >= 1:
                word2idx[w] = len(word2idx)
        return word2idx

    def _load_video_feature(self, vid_id):
        """
        [修改点 2] 读取视频特征
        优先读取 .npz (用户数据), 兼容 .npy, 保留原有采样逻辑
        """
        p_npz = os.path.join(self.feature_dir, f"{vid_id}.npz")
        p_npy = os.path.join(self.feature_dir, f"{vid_id}.npy")
        
        feat = None
        if os.path.exists(p_npz):
            try:
                data = np.load(p_npz)
                # 处理 npz 可能的 key: 通常是 'features', 'embedding' 或 'arr_0'
                if isinstance(data, np.lib.npyio.NpzFile):
                    # 尝试常见键名
                    for key in ['features', 'arr_0', 'embedding']:
                        if key in data:
                            feat = data[key]
                            break
                    if feat is None:
                        # 如果没找到常见键，取第一个
                        feat = data[list(data.keys())[0]]
                else:
                    feat = data
            except Exception as e:
                logger.error(f"Error loading {p_npz}: {e}")
        elif os.path.exists(p_npy):
            feat = np.load(p_npy)
            
        if feat is None:
            # 这里的 raise 替代了您代码中潜在的随机生成逻辑，保证训练数据的真实性
            # 如果确实需要随机特征调试，可在此处 catch 并生成 torch.randn
            raise FileNotFoundError(f"Feature not found for {vid_id} in {self.feature_dir}")

        feat = torch.from_numpy(feat).float()
        if feat.dim() == 1: feat = feat.unsqueeze(0) # 修正维度

        # --- 保留原有的采样/填充逻辑 ---
        seq_len = feat.shape[0]
        target_len = self.max_v_l
        
        if seq_len >= target_len:
            idxs = torch.linspace(0, seq_len - 1, target_len).long()
            feat = feat[idxs]
            mask = torch.ones(target_len, dtype=torch.bool)
        else:
            pad_len = target_len - seq_len
            padding = torch.zeros(pad_len, feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
            mask = torch.cat([torch.ones(seq_len, dtype=torch.bool), 
                              torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            
        # 返回原始 seq_len 用于计算 duration
        return feat, mask, seq_len

    def _process_text(self, sentence):
        """将句子转换为 Token IDs"""
        if self.text_encoder_type == 'clip':
            try:
                tokens = clip.tokenize(sentence, truncate=True).squeeze(0)
            except NameError:
                raise RuntimeError("Please install openai-clip to use CLIP tokenizer.")
            
            if self.max_q_l < 77:
                tokens = tokens[:self.max_q_l]
            mask = (tokens != 0)
            return tokens, mask

        elif self.text_encoder_type == 'glove':
            words = sentence.lower().replace('.', '').replace(',', '').split()
            words = words[:self.max_q_l]
            ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
            pad_len = self.max_q_l - len(ids)
            mask = [True] * len(ids) + [False] * pad_len
            ids = ids + [self.word2idx["<PAD>"]] * pad_len
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        vid_id = sample['video_id']
        sentence = sample['sentence']
        
        # 1. Video
        video_feat, video_mask, raw_seq_len = self._load_video_feature(vid_id)
        
        # [修改点 3] 计算 Duration 和 归一化时间戳
        # 用户参数: 24fps, stride=4.  即每帧代表 4/24 = 1/6 秒
        time_unit = 4.0 / 24.0 
        
        # 如果 sample 中没有 duration (txt 读取情况)，则根据特征长度估算
        duration = sample.get('duration', raw_seq_len * time_unit)
        
        # 防止 duration 为 0
        if duration <= 0: duration = 1.0

        # 2. Text
        words_id, words_mask = self._process_text(sentence)
        
        # 3. Target (Spans)
        gt_spans = []
        if 'timestamps' in sample:
            for st, ed in sample['timestamps']:
                st = max(0, min(st, duration))
                ed = max(0, min(ed, duration))
                if ed <= st: continue 

                # 归一化到 [0, 1]
                center = (st + ed) / 2.0 / duration
                width = (ed - st) / duration
                gt_spans.append([center, width])
        
        if len(gt_spans) == 0:
            gt_spans.append([0.5, 0.1]) # Dummy target if empty
            
        target = {
            "spans": torch.tensor(gt_spans, dtype=torch.float32),
            "labels": torch.zeros(len(gt_spans), dtype=torch.long),
            "video_id": vid_id,
            "duration": duration,
            "raw_txt": sentence 
        }

        return {
            "video_feat": video_feat,
            "video_mask": video_mask,
            "words_id": words_id,
            "words_mask": words_mask,
            "targets": target # 注意：Collator 中这里对应 list of dicts
        }

def collate_fn(batch):
    batch_data = {
        "video_feat": torch.stack([b['video_feat'] for b in batch]),
        "video_mask": torch.stack([b['video_mask'] for b in batch]),
        "words_id": torch.stack([b['words_id'] for b in batch]),
        "words_mask": torch.stack([b['words_mask'] for b in batch]),
        "targets": [b['targets'] for b in batch] 
    }
    return batch_data
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import logging

# 尝试导入 CLIP
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
        
        # 1. 加载标注文件
        anno_path = args.annotation_path
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found at {anno_path}")
            
        logger.info(f"Loading annotations from {anno_path}...")
        
        # 根据后缀判断加载方式
        if anno_path.endswith('.txt'):
            self.annotations = self._load_txt(anno_path)
        else:
            try:
                with open(anno_path, 'r') as f:
                    self.annotations = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to load JSON from {anno_path}.")
                raise

        # 2. 文本处理准备 (Tokenizer / Vocab)
        self.text_encoder_type = args.text_encoder_type
        self.vocab = None
        
        if self.text_encoder_type == 'glove':
            self.word2idx = self._build_vocab()
            self.vocab = list(self.word2idx.keys())
            logger.info(f"Vocab size: {len(self.vocab)}")
        elif self.text_encoder_type == 'clip':
            if clip is None:
                logger.warning("Warning: 'clip' module not found. Please install openai-clip.")

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
                        "timestamps": [[start, end]],
                        "sentence": sentence.strip()
                        # 注意：txt没有duration，将在 __getitem__ 中计算
                    })
                except Exception as e:
                    logger.warning(f"Skipping line: {line}. Error: {e}")
        logger.info(f"Loaded {len(data)} samples from txt.")
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
        读取视频特征并进行 [均匀降采样] 或 [填充]
        """
        p_npz = os.path.join(self.feature_dir, f"{vid_id}.npz")
        p_npy = os.path.join(self.feature_dir, f"{vid_id}.npy")
        
        feat = None
        if os.path.exists(p_npz):
            try:
                data = np.load(p_npz)
                if isinstance(data, np.lib.npyio.NpzFile):
                    for key in ['features', 'arr_0', 'embedding']:
                        if key in data:
                            feat = data[key]
                            break
                    if feat is None and len(data.files) > 0:
                        feat = data[data.files[0]]
                else:
                    feat = data
            except Exception as e:
                logger.error(f"Error loading {p_npz}: {e}")
        elif os.path.exists(p_npy):
            feat = np.load(p_npy)
            
        if feat is None:
            raise FileNotFoundError(f"Feature not found for {vid_id}")

        feat = torch.from_numpy(feat).float()
        
        # [健壮性检查] 确保维度是 [Time, Dim]
        # 有些提取脚本保存的是 [Dim, Time]，需要转置
        if feat.shape[0] == self.args.v_feat_dim and feat.shape[1] != self.args.v_feat_dim:
             # 如果第0维是512(特征维)，第1维是时间，则转置
             feat = feat.transpose(0, 1)
        
        # 如果是一维数据 [Dim]，扩展为 [1, Dim]
        if feat.dim() == 1: 
            feat = feat.unsqueeze(0)
            
        raw_seq_len = feat.shape[0]
        target_len = self.max_v_l
        
        # --- 核心采样逻辑 ---
        if raw_seq_len >= target_len:
            # [降采样] 均匀选取 target_len 个帧，覆盖全视频
            # 例如: [0, 2, 4, 6...] 
            idxs = torch.linspace(0, raw_seq_len - 1, target_len).long()
            feat = feat[idxs]
            mask = torch.ones(target_len, dtype=torch.bool)
        else:
            # [填充] 补零
            pad_len = target_len - raw_seq_len
            padding = torch.zeros(pad_len, feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
            mask = torch.cat([torch.ones(raw_seq_len, dtype=torch.bool), 
                              torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            
        return feat, mask, raw_seq_len

    def _process_text(self, sentence):
        """将句子转换为 Token IDs"""
        if self.text_encoder_type == 'clip':
            try:
                # CLIP Tokenizer 自带截断和填充
                tokens = clip.tokenize(sentence, truncate=True).squeeze(0)
            except NameError:
                raise RuntimeError("Please install openai-clip to use CLIP tokenizer.")
            
            # 如果配置的 max_q_l 小于 CLIP 默认的 77，手动截断
            if self.max_q_l < 77:
                tokens = tokens[:self.max_q_l]
            
            # 构建 mask (非 0 为 True)
            mask = (tokens != 0)
            return tokens, mask

        elif self.text_encoder_type == 'glove':
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
        sentence = sample['sentence']
        
        # 1. 加载视频特征
        video_feat, video_mask, raw_seq_len = self._load_video_feature(vid_id)
        
        # 2. 计算视频真实时长
        # 您的特征提取参数: 24fps, stride=4
        # 时间分辨率 = stride / fps = 4 / 24 = 1/6 秒
        time_unit = 4.0 / 24.0 
        
        # 优先使用 json 中的 duration，如果没有则根据特征长度估算
        # raw_seq_len 是未采样前的原始帧数
        duration = sample.get('duration', raw_seq_len * time_unit)
        
        # 3. 处理文本
        words_id, words_mask = self._process_text(sentence)
        
        # 4. 处理 Target (归一化 Span)
        gt_spans = []
        if 'timestamps' in sample:
            for st, ed in sample['timestamps']:
                # 边界保护
                st = max(0, min(st, duration))
                ed = max(0, min(ed, duration))
                if ed <= st: continue 

                # 归一化到 [0, 1]
                center = (st + ed) / 2.0 / duration
                width = (ed - st) / duration
                gt_spans.append([center, width])
        
        if len(gt_spans) == 0:
            # 异常数据保护
            gt_spans.append([0.5, 0.1]) 
            
        target = {
            "spans": torch.tensor(gt_spans, dtype=torch.float32),
            "labels": torch.zeros(len(gt_spans), dtype=torch.long),
            "video_id": vid_id,
            "duration": duration,
            "raw_txt": sentence 
        }

        return {
            "video_feat": video_feat, # [max_v_l, Dim]
            "video_mask": video_mask, # [max_v_l]
            "words_id": words_id,     # [max_q_l]
            "words_mask": words_mask, # [max_q_l]
            "targets": target 
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
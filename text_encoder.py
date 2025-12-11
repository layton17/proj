import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

# =================================================================
# 1. 基础组件 (LayerNorm, QuickGELU, ResidualAttentionBlock)
#    用于构建 CLIP 的 Transformer 结构
# =================================================================

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# =================================================================
# 2. CLIP Text Encoder
#    手动实现的 CLIP 文本编码器，加载权重后可提取 Token 级特征
# =================================================================

class CLIPTextEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12
                 ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # causal attention mask
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        # 默认使用 FP32，如果加载了 FP16 权重会自动转换
        return self.token_embedding.weight.dtype

    def forward(self, text):
        """
        Args:
            text: [batch_size, n_ctx] token ids
        Returns:
            dict with 'last_hidden_state' [batch_size, n_ctx, embed_dim]
        """
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # 投影到联合空间 (Optional, depending on usage)
        # x = x @ self.text_projection

        # last_hidden_state 保留了词级信息，用于 FW-MESM
        # pooler_output 用于句子级信息，用于 SS-MESM
        eos_x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] # @ self.text_projection

        return dict(last_hidden_state=x, pooler_output=eos_x)


# =================================================================
# 3. GloVe Text Encoder
#    用于旧数据集 (Charades-STA 等)
# =================================================================

class GloVe(object):
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.dim = 300
        self.glove = self._load()
        # 初始化特殊 token
        if "<PAD>" not in self.glove:
            self.glove["<PAD>"] = torch.zeros(self.dim)
        if "<UNK>" not in self.glove:
            self.glove["<UNK>"] = torch.randn(self.dim)

    def get(self, word):
        return self.glove.get(word, self.glove["<UNK>"])

    def _load(self):
        glove = dict()
        # 假设 GloVe 文件格式为: word val1 val2 ...
        try:
            with open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Loading GloVe from {self.glove_path}"):
                    split_line = line.split()
                    word = " ".join(split_line[:-self.dim])
                    embedding = torch.tensor([float(val) for val in split_line[-self.dim:]], dtype=torch.float32)
                    glove[word] = embedding
        except FileNotFoundError:
            print(f"Warning: GloVe file not found at {self.glove_path}. Initializing empty dictionary.")
        return glove

class GloveTextEncoder(nn.Module):
    def __init__(self, vocab_list, glove_path):
        """
        Args:
            vocab_list: list of words in vocabulary
            glove_path: path to .txt glove file
        """
        super(GloveTextEncoder, self).__init__()
        self.glove_loader = GloVe(glove_path)
        dim = self.glove_loader.dim
        
        self.emb = nn.Embedding(num_embeddings=len(vocab_list), embedding_dim=dim)
        
        # 初始化 Embedding 权重
        weight_matrix = torch.zeros((len(vocab_list), dim))
        for i, word in enumerate(vocab_list):
            weight_matrix[i] = self.glove_loader.get(word)
            
        self.emb.weight.data.copy_(weight_matrix)
        
        # Freeze parameters
        for param in self.emb.parameters():
            param.requires_grad = False

    def forward(self, word_ids):
        """
        Args:
            word_ids: (B, L)
        Returns:
            (B, L, out_dim)
        """
        return self.emb(word_ids)

# 辅助函数：加载预训练权重到 CLIPTextEncoder
def load_clip_weights(model, state_dict):
    # 需要根据实际 state_dict 的 key 进行匹配
    # 这里只是一个占位符，实际使用时建议直接用 openai/CLIP 库加载权重后提取参数
    model.load_state_dict(state_dict, strict=False)
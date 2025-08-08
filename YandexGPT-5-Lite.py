# python >=3.9, torch >=2.1
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Config (matches your spec)
# ---------------------------
@dataclass
class YandexGPT5LiteConfig:
    vocab_size: int = 129_024              # llama.vocab_size
    context_length: int = 32_768           # llama.context_length
    d_model: int = 4_096                   # llama.embedding_length
    n_layers: int = 32                     # llama.block_count
    n_heads: int = 32                      # llama.attention.head_count
    n_kv_heads: int = 8                    # llama.attention.head_count_kv (GQA)
    rope_dim: int = 128                    # llama.rope.dimension_count
    rope_base: float = 500_000.0           # llama.rope.freq_base
    rms_eps: float = 1e-6                  # llama.attention.layer_norm_rms_epsilon
    ffn_hidden: int = 14_336               # llama.feed_forward_length

    # Tokenizer hints (not required for the module itself)
    add_bos_token: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 2  # typical; adjust if needed


# ---------------------------
# RMSNorm (pre-norm)
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


# ---------------------------
# RoPE helpers
# ---------------------------
def _rope_inv_freq(dim: int, base: float, device, dtype):
    # dim must be even
    return 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))

def apply_rope(q: torch.Tensor, k: torch.Tensor, rope_dim: int, base: float):
    """
    q, k: [B, n, T, H]; rotate first rope_dim dims (<= H).
    """
    b, n, t, h = q.shape
    rope_dim = min(rope_dim, h)
    device = q.device

    inv = _rope_inv_freq(rope_dim, base, device, torch.float32)
    pos = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", pos, inv)  # [T, rope_dim/2]
    cos = torch.cos(freqs)[None, None, :, :, None]
    sin = torch.sin(freqs)[None, None, :, :, None]

    def rope(x):
        x1 = x[..., :rope_dim].view(b, n, t, rope_dim // 2, 2)
        x2 = x[..., rope_dim:]
        a, b2 = x1[..., 0], x1[..., 1]
        a2 = a * cos.squeeze(-1) - b2 * sin.squeeze(-1)
        b3 = a * sin.squeeze(-1) + b2 * cos.squeeze(-1)
        x1_out = torch.stack([a2, b3], dim=-1).view(b, n, t, rope_dim)
        return torch.cat([x1_out, x2], dim=-1)

    return rope(q), rope(k)


# ---------------------------
# Multi-Head Attention with GQA
# ---------------------------
class LlamaAttentionGQA(nn.Module):
    """
    Matches your per-tensor shapes and names:
      - attn_q.weight : [4096, 4096]   (Linear: in=4096, out=4096)
      - attn_k.weight : [1024, 4096]   (Linear: in=4096, out=1024)   # 8 KV heads * 128
      - attn_v.weight : [1024, 4096]   (Linear: in=4096, out=1024)
      - attn_output.weight : [4096, 4096]
    GGML lists matrices transposed; PyTorch stores [out, in].
    """
    def __init__(self, cfg: YandexGPT5LiteConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads  # 4096/32 = 128
        assert self.head_dim * self.n_heads == self.d_model

        self.attn_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_k = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.attn_v = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.attn_output = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope_dim = cfg.rope_dim
        self.rope_base = cfg.rope_base

    def _split_heads(self, x: torch.Tensor, n: int) -> torch.Tensor:
        # [B, T, n*H] -> [B, n, T, H]
        b, t, c = x.shape
        h = c // n
        x = x.view(b, t, n, h)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, n, T, H] -> [B, T, n*H]
        b, n, t, h = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(b, t, n * h)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # [B, n_kv, T, H] -> [B, n_heads, T, H]
        b, n_kv, t, h = x.shape
        rep = self.n_heads // n_kv
        return x.repeat_interleave(rep, dim=1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.attn_q(x)  # [B, T, 4096]
        k = self.attn_k(x)  # [B, T, 1024]
        v = self.attn_v(x)  # [B, T, 1024]

        q = self._split_heads(q, self.n_heads)      # [B, 32, T, 128]
        k = self._split_heads(k, self.n_kv_heads)   # [B,  8, T, 128]
        v = self._split_heads(v, self.n_kv_heads)   # [B,  8, T, 128]

        # RoPE on first rope_dim dims
        q, k = apply_rope(q, k, self.rope_dim, self.rope_base)

        # GQA: repeat KV to match Q heads
        k = self._repeat_kv(k)                      # [B, 32, T, 128]
        v = self._repeat_kv(v)                      # [B, 32, T, 128]

        # Scaled dot-product attention (standard 1/sqrt(head_dim) for LLaMA)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B, 32, T, T]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask   # additive mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_out = torch.matmul(attn_probs, v)      # [B, 32, T, 128]
        attn_out = self._merge_heads(attn_out)      # [B, T, 4096]
        out = self.attn_output(attn_out)            # [B, T, 4096]
        return out


# ---------------------------
# LLaMA-style FFN (gated SiLU)
# ---------------------------
class LlamaFFN(nn.Module):
    """
    Matches your tensors:
      - ffn_up.weight   : [4096, 14336] in GGML (transposed) -> PyTorch Linear(4096 -> 14336)
      - ffn_gate.weight : [4096, 14336] in GGML (transposed) -> PyTorch Linear(4096 -> 14336)
      - ffn_down.weight : [14336, 4096] in GGML (transposed) -> PyTorch Linear(14336 -> 4096)
    """
    def __init__(self, cfg: YandexGPT5LiteConfig):
        super().__init__()
        self.ffn_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.ffn_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.ffn_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.ffn_up(x)                 # [B, T, 14336]
        g = self.ffn_gate(x)               # [B, T, 14336]
        return self.ffn_down(F.silu(g) * u)


# ---------------------------
# Transformer Block
# ---------------------------
class LlamaBlock(nn.Module):
    """
    Parameter names align with your table:
      - attn_norm.weight
      - attn_q.weight, attn_k.weight, attn_v.weight, attn_output.weight
      - ffn_norm.weight
      - ffn_up.weight, ffn_gate.weight, ffn_down.weight
    """
    def __init__(self, cfg: YandexGPT5LiteConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = LlamaAttentionGQA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = LlamaFFN(cfg)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + h
        h2 = self.ffn(self.ffn_norm(x))
        x = x + h2
        return x


# ---------------------------
# Full Model
# ---------------------------
class YandexGPT5LiteModel(nn.Module):
    """
    Tensor names and shapes (PyTorch view):
      token_embd.weight          : [vocab, d_model]
      blk.{i}.attn_norm.weight   : [d_model]
      blk.{i}.attn_q.weight      : [4096, 4096]
      blk.{i}.attn_k.weight      : [1024, 4096]
      blk.{i}.attn_v.weight      : [1024, 4096]
      blk.{i}.attn_output.weight : [4096, 4096]
      blk.{i}.ffn_norm.weight    : [d_model]
      blk.{i}.ffn_up.weight      : [14336, 4096]
      blk.{i}.ffn_gate.weight    : [14336, 4096]
      blk.{i}.ffn_down.weight    : [4096, 14336]
      output_norm.weight         : [d_model]
      output.weight              : [vocab, d_model]   # GGML lists [4096, 129024] = transposed
    """
    def __init__(self, cfg: YandexGPT5LiteConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding
        self.token_embd = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Blocks
        self.blk = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layers)])

        # Final norm and output head (separate parameter per your table)
        self.output_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Token IDs for convenience
        self.bos_token_id = cfg.bos_token_id
        self.eos_token_id = cfg.eos_token_id
        self.pad_token_id = cfg.pad_token_id

    def _causal_mask(self, t: int, device, dtype=torch.float32):
        i = torch.arange(t, device=device)
        j = torch.arange(t, device=device)
        mask = (i[:, None] < j[None, :]).to(dtype) * (-1e9)
        return mask[None, None, :, :]  # [1,1,T,T]

    def forward(self,
                input_ids: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                return_logits: bool = True):
        """
        input_ids: [B, T]
        attn_mask: optional additive mask broadcastable to [B, 1, T, T]
        """
        b, t = input_ids.shape
        x = self.token_embd(input_ids)  # [B, T, 4096]

        if attn_mask is None:
            attn_mask = self._causal_mask(t, x.device, x.dtype)

        for block in self.blk:
            x = block(x, attn_mask=attn_mask)

        x = self.output_norm(x)

        if return_logits:
            logits = self.output(x)  # [B, T, vocab]
            return logits

        return x  # hidden states


# ---------------------------
# Build helper
# ---------------------------
def build_yandexgpt5_lite_from_spec() -> YandexGPT5LiteModel:
    cfg = YandexGPT5LiteConfig()
    return YandexGPT5LiteModel(cfg)


if __name__ == "__main__":
    cfg = YandexGPT5LiteConfig()
    model = YandexGPT5LiteModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    print(logits.shape)  # [2, 16, 129024]

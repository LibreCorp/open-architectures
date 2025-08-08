# python >=3.9, torch >=2.1
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Config (matches your spec)
# ---------------------------
@dataclass
class Phi3Config:
    vocab_size: int = 100_352                 # inferred from weights
    context_length: int = 16_384              # phi3.context_length
    d_model: int = 5_120                      # phi3.embedding_length
    n_layers: int = 40                        # phi3.block_count
    n_heads: int = 40                         # phi3.attention.head_count
    n_kv_heads: int = 10                      # phi3.attention.head_count_kv (GQA)
    rope_dim: int = 128                       # phi3.rope.dimension_count
    rope_base: float = 250_000.0              # phi3.rope.freq_base
    rope_orig_ctx: int = 16_384               # phi3.rope.scaling.original_context_length
    rms_eps: float = 1e-5                     # phi3.attention.layer_norm_rms_epsilon
    ffn_hidden: int = 17_920                  # phi3.feed_forward_length
    sliding_window: int = 131_072             # phi3.attention.sliding_window (>= ctx => full causal)
    tie_word_embeddings: bool = True          # typical for GPTs


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
# Rotary embeddings (RoPE)
# ---------------------------
def _rope_angles(dim: int, base: float, device, dtype):
    # dim is rope_dim (even)
    # frequencies: base^(2i/dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    return inv_freq

def apply_rope(q: torch.Tensor,
               k: torch.Tensor,
               rope_dim: int,
               base: float,
               orig_ctx: int,
               positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, n, T, H] where H=head_dim
    rope_dim: number of dims to rotate (<= head_dim), here 128 which equals head_dim
    """
    b, n, t, h = q.shape
    device, dtype = q.device, q.dtype
    rope_dim = min(rope_dim, h)
    inv_freq = _rope_angles(rope_dim, base, device, torch.float32)  # use float32 for phase stability

    if positions is None:
        pos = torch.arange(t, device=device, dtype=torch.float32)
    else:
        pos = positions.to(dtype=torch.float32)

    # Optional linear scaling for longer contexts (NTK-like). Here we keep identity scaling
    # because rope_orig_ctx == context_length. You can add scaling if you extrapolate.
    freqs = torch.einsum("t,f->tf", pos, inv_freq)  # [T, rope_dim/2]
    # Cast phases to compute sin/cos in float32, then back to dtype
    cos = torch.cos(freqs)[None, None, :, :, None].to(dtype)
    sin = torch.sin(freqs)[None, None, :, :, None].to(dtype)

    def rotate_half(x):
        x1 = x[..., :rope_dim]
        x2 = x[..., rope_dim:]
        x1 = x1.view(b, n, t, rope_dim // 2, 2)
        x1_rot = torch.stack([-x1[..., 1], x1[..., 0]], dim=-1)
        x1_rot = x1_rot.view(b, n, t, rope_dim)
        return x1_rot, x2

    def rope(x):
        x1 = x[..., :rope_dim]
        x2 = x[..., rope_dim:]
        x1 = x1.view(b, n, t, rope_dim // 2, 2)
        x1_rot = torch.stack([-x1[..., 1], x1[..., 0]], dim=-1)  # rotate pairs
        x1_rot = x1_rot.view(b, n, t, rope_dim)

        x1 = x[..., :rope_dim].view(b, n, t, rope_dim // 2, 2)
        a, b2 = x1[..., 0], x1[..., 1]  # [B,n,T,rope_dim/2]
        a2 = a * cos.squeeze(-1) - b2 * sin.squeeze(-1)
        b3 = a * sin.squeeze(-1) + b2 * cos.squeeze(-1)
        x1_out = torch.stack([a2, b3], dim=-1).view(b, n, t, rope_dim)
        return torch.cat([x1_out, x2], dim=-1)

    q = rope(q)
    k = rope(k)
    return q, k


# ---------------------------
# Multi-Head Attention w/ GQA
# ---------------------------
class MHA_GQA(nn.Module):
    """
    Fused QKV projection with Grouped-Query Attention:
      - n_heads total query heads
      - n_kv_heads shared K/V heads (each KV head serves n_heads / n_kv_heads Q heads)
    Shapes tied to your tensor inventory:
      attn_qkv.weight: [7680, 5120] in PyTorch (GGML lists transposed)
        - 5120 dims in -> 7680 out (Q=5120, K=1280, V=1280)
      attn_output.weight: [5120, 5120]
    """
    def __init__(self, cfg: Phi3Config):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads  # 5120 / 40 = 128
        assert self.head_dim * self.n_heads == self.d_model
        assert cfg.rope_dim <= self.head_dim

        # fused qkv: out = 5120 + 1280 + 1280 = 7680
        self.attn_qkv = nn.Linear(cfg.d_model, cfg.d_model + 2 * (self.n_kv_heads * self.head_dim), bias=False)
        self.attn_output = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope_dim = cfg.rope_dim
        self.rope_base = cfg.rope_base
        self.rope_orig_ctx = cfg.rope_orig_ctx
        self.sliding_window = cfg.sliding_window

    def _split_heads(self, x: torch.Tensor, n: int) -> torch.Tensor:
        # x: [B, T, n * head_dim] -> [B, n, T, head_dim]
        b, t, c = x.shape
        x = x.view(b, t, n, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n, T, head_dim] -> [B, T, n*head_dim]
        b, n, t, h = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(b, t, n * h)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match Q heads: factor = n_heads // n_kv_heads
        x: [B, n_kv, T, H] -> [B, n_heads, T, H]
        """
        b, n_kv, t, h = x.shape
        rep = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(rep, dim=1)

    def _causal_mask(self, t: int, window: int, device, dtype):
        # Causal mask with optional sliding window (local attention)
        i = torch.arange(t, device=device)
        j = torch.arange(t, device=device)
        mask = i[:, None] < j[None, :]  # True where future positions
        if window is not None and window < t:
            # disallow attending to tokens older than 'window'
            too_old = (i[:, None] - j[None, :]) > (window - 1)
            mask = mask | too_old
        return mask.to(dtype=dtype) * (-1e9)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.attn_qkv(x)  # [B, T, 7680]
        q, k, v = torch.split(qkv, [self.d_model, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=-1)

        q = self._split_heads(q, self.n_heads)        # [B, 40, T, 128]
        k = self._split_heads(k, self.n_kv_heads)     # [B, 10, T, 128]
        v = self._split_heads(v, self.n_kv_heads)     # [B, 10, T, 128]

        # RoPE on first rope_dim dims (here rope_dim == head_dim == 128)
        q, k = apply_rope(q, k, self.rope_dim, self.rope_base, self.rope_orig_ctx, positions)

        # GQA: repeat KV
        k = self._repeat_kv(k)                        # [B, 40, T, 128]
        v = self._repeat_kv(v)                        # [B, 40, T, 128]

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, 40, T, T]

        # Causal + sliding window mask
        if attn_mask is None:
            causal = self._causal_mask(t, self.sliding_window, x.device, x.dtype)
            attn_scores = attn_scores + causal
        else:
            attn_scores = attn_scores + attn_mask  # allow external mask if you pass one

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_out = torch.matmul(attn_probs, v)      # [B, 40, T, 128]
        attn_out = self._merge_heads(attn_out)      # [B, T, 5120]
        out = self.attn_output(attn_out)            # [B, T, 5120]
        return out


# ---------------------------
# SwiGLU MLP (Phi3-style)
# ---------------------------
class SwiGLU(nn.Module):
    """
    Matches your tensor list:
      ffn_up.weight   : [35840, 5120]  (d_model -> 2 * ffn_hidden)
      ffn_down.weight : [17920, 5120]  (ffn_hidden -> d_model)  NOTE: in GGML listing, dims are transposed.
    In PyTorch, nn.Linear(out, in) uses [out, in].
    """
    def __init__(self, cfg: Phi3Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)   # 5120 -> 35840
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)     # 17920 -> 5120

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)                                  # [B, T, 35840]
        x1, x2 = x_up.chunk(2, dim=-1)                     # [B, T, 17920] each
        return self.down(F.silu(x1) * x2)                  # SwiGLU


# ---------------------------
# Transformer Block
# ---------------------------
class Block(nn.Module):
    """
    Matches naming in your tensor inventory:
      - attn_norm.weight    (RMSNorm)
      - attn_qkv.weight     (Linear fused)
      - attn_output.weight  (Linear)
      - ffn_norm.weight     (RMSNorm)
      - ffn_up.weight       (Linear)
      - ffn_down.weight     (Linear)
    """
    def __init__(self, cfg: Phi3Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = MHA_GQA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attn(self.attn_norm(x), positions=positions, attn_mask=attn_mask)
        x = x + h
        h2 = self.ffn(self.ffn_norm(x))
        x = x + h2
        return x


# ---------------------------
# Full Model
# ---------------------------
class Phi3Model(nn.Module):
    """
    Tensor names and shapes (PyTorch view):
      token_embd.weight: [vocab_size, d_model]
      blk.{i}.attn_norm.weight: [d_model]
      blk.{i}.attn_qkv.weight: [7680, 5120]
      blk.{i}.attn_output.weight: [5120, 5120]
      blk.{i}.ffn_norm.weight: [5120]
      blk.{i}.ffn_up.weight: [35840, 5120]
      blk.{i}.ffn_down.weight: [5120, 17920]  (PyTorch internal; GGML listing shows transposed)
      output_norm.weight: [5120]
      output.weight: [vocab_size, 5120]
    """
    def __init__(self, cfg: Phi3Config):
        super().__init__()
        self.cfg = cfg
        self.token_embd = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blk = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.output_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            # Tie LM head to embeddings (common GPT trick)
            self.output.weight = self.token_embd.weight

        # Register BOS/EOS/PAD ids for convenience (from your tokenizer section)
        self.bos_token_id = 100_257
        self.eos_token_id = 100_257
        self.pad_token_id = 100_257

    def forward(self,
                input_ids: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                return_logits: bool = True) -> torch.Tensor:
        """
        input_ids: [B, T]
        positions: [T] or [B, T] token positions for RoPE; if None, uses arange(T)
        attn_mask: optional [B, 1, T, T] additive mask (broadcastable) in logits space
        """
        b, t = input_ids.shape
        if positions is None:
            positions = torch.arange(t, device=input_ids.device)

        x = self.token_embd(input_ids)   # [B, T, 5120]

        # If an external mask is provided, it should already include causal structure.
        # Internally, each attention layer applies causal+window if attn_mask is None.
        for block in self.blk:
            x = block(x, positions=positions, attn_mask=attn_mask)

        x = self.output_norm(x)
        if return_logits:
            logits = self.output(x)  # [B, T, vocab_size]
            return logits
        return x  # hidden states

    # Convenience for generation-time causal mask if you want to pass it explicitly
    def build_attention_mask(self, seq_len: int, device, dtype=torch.float32) -> torch.Tensor:
        i = torch.arange(seq_len, device=device)
        j = torch.arange(seq_len, device=device)
        mask = i[:, None] < j[None, :]
        if self.cfg.sliding_window is not None and self.cfg.sliding_window < seq_len:
            too_old = (i[:, None] - j[None, :]) > (self.cfg.sliding_window - 1)
            mask = mask | too_old
        mask = mask.to(dtype=dtype) * (-1e9)
        # shape to [1, 1, T, T] so it broadcasts over batch and heads
        return mask[None, None, :, :]


# ---------------------------
# Build helper
# ---------------------------
def build_phi3_from_spec() -> Phi3Model:
    cfg = Phi3Config()
    return Phi3Model(cfg)


if __name__ == "__main__":
    cfg = Phi3Config()
    model = Phi3Model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    print(logits.shape)  # [2, 16, 100352]

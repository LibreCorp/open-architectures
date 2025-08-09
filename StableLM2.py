# StableLM replica matching the provided spec (GQA, per-head q/k RMSNorm, RoPE dim=40, parallel residual)
# Notes:
# - Tensor/module names mirror the GGML key layout so a loader can map/transposed weights easily.
# - Quantized weights (Q4_0/Q6_K) must be dequantized to float before loading here.
# - attn_q_norm/attn_k_norm implement per-head RMSNorm with shapes [head_dim, n_heads] and [head_dim, n_kv_heads].

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StableLMConfig:
    vocab_size: int = 100352
    context_length: int = 4096
    embedding_length: int = 5120  # d_model
    feed_forward_length: int = 13824  # hidden size for MLP
    block_count: int = 40
    head_count: int = 32
    head_count_kv: int = 8
    layer_norm_epsilon: float = 1e-5
    rope_dimension_count: int = 40  # rotary dim per head
    use_parallel_residual: bool = True

    # convenience
    @property
    def head_dim(self) -> int:
        assert self.embedding_length % self.head_count == 0
        return self.embedding_length // self.head_count

    @property
    def kv_dim(self) -> int:
        return self.head_dim

    @property
    def q_proj_out(self) -> int:
        return self.embedding_length  # 32 * 160

    @property
    def kv_proj_out(self) -> int:
        return self.head_count_kv * self.kv_dim  # 8 * 160 = 1280


class RMSNorm(nn.Module):
    """Standard RMSNorm with bias option to match attn_norm/ffn_norm usage."""
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class HeadwiseRMSNorm(nn.Module):
    """RMSNorm applied per attention head.
    Parameter shape mirrors GGML tensors:
      - attn_q_norm.weight: [head_dim, n_heads]
      - attn_k_norm.weight: [head_dim, n_kv_heads]
    We store as nn.Parameter of shape (n_heads_or_kv, head_dim) for convenient broadcasting
    and expose .weight in transposed view for GGML parity via .weight_t.
    """
    def __init__(self, n_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.nh = n_heads
        self.hd = head_dim
        # (nh, hd)
        self.weight = nn.Parameter(torch.ones(n_heads, head_dim))

    @property
    def weight_t(self) -> torch.Tensor:
        """Transpose view to match GGML shape [head_dim, n_heads]."""
        return self.weight.transpose(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, nh, hd)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        y = y * self.weight.view(1, 1, self.nh, self.hd)
        return y


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 5.0e6, max_position: int = 32768):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # cache
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seqlen: int, device: torch.device, dtype: torch.dtype):
        if seqlen <= self.max_seq_len_cached:
            return
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
        self.max_seq_len_cached = seqlen

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: (B, T, hq, d), k: (B, T, hk, d)
        seqlen = q.size(1)
        device = q.device
        dtype = q.dtype
        self._build_cache(seqlen, device, dtype)
        cos = self.cos_cached[:seqlen]
        sin = self.sin_cached[:seqlen]

        def apply_rot(x):
            # x: (B, T, H, D)
            # only first rotary_dim dims are rotated; rest pass through
            x1, x2 = x[..., : self.dim], x[..., self.dim :]
            # split last dim into pairs
            x1_odd = x1[..., 1::2]
            x1_even = x1[..., ::2]
            x1_rot = torch.stack((-x1_odd, x1_even), dim=-1).reshape_as(x1)
            x1_out = x1 * cos.view(1, seqlen, 1, self.dim) + x1_rot * sin.view(1, seqlen, 1, self.dim)
            return torch.cat([x1_out, x2], dim=-1)

        return apply_rot(q), apply_rot(k)


class StableLMAttention(nn.Module):
    def __init__(self, cfg: StableLMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embedding_length
        hd = cfg.head_dim
        self.hq = cfg.head_count
        self.hk = cfg.head_count_kv

        # Projections with GGML-compatible names/shapes
        self.attn_q = nn.Linear(d, cfg.q_proj_out, bias=False)
        self.attn_k = nn.Linear(d, cfg.kv_proj_out, bias=False)
        self.attn_v = nn.Linear(d, cfg.kv_proj_out, bias=False)
        self.attn_output = nn.Linear(d, d, bias=False)

        # Per-head RMSNorm for q/k
        self.attn_q_norm = HeadwiseRMSNorm(self.hq, hd, eps=cfg.layer_norm_epsilon)
        self.attn_k_norm = HeadwiseRMSNorm(self.hk, hd, eps=cfg.layer_norm_epsilon)

        # Pre-attention norm (full-dim). GGML has both attn_norm.weight and .bias
        self.attn_norm = RMSNorm(d, eps=cfg.layer_norm_epsilon, bias=True)

        self.rope = RotaryEmbedding(cfg.rope_dimension_count, base=5.0e6, max_position=cfg.context_length)

    def _shape(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, T, D = x.size()
        head_dim = self.cfg.head_dim
        return x.view(B, T, n_heads, head_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.size()
        # pre-attn norm
        nx = self.attn_norm(x)

        # projections
        q = self._shape(self.attn_q(nx), self.hq)
        k = self._shape(self.attn_k(nx), self.hk)
        v = self._shape(self.attn_v(nx), self.hk)

        # per-head RMSNorm on q/k
        q = self.attn_q_norm(q)
        k = self.attn_k_norm(k)

        # RoPE (partial, first rope_dim dims)
        q, k = self.rope(q, k)

        # Expand k,v from hk to hq for grouped-query attention
        if self.hq % self.hk != 0:
            raise ValueError("head_count must be a multiple of head_count_kv for GQA")
        expand = self.hq // self.hk
        k = k.repeat_interleave(expand, dim=2)  # (B, T, hq, hd)
        v = v.repeat_interleave(expand, dim=2)

        # scaled dot-product attention with causal mask
        q = q.transpose(1, 2)  # (B, h, T, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.cfg.head_dim)

        # build causal mask if none provided
        if attn_mask is None:
            # (T, T) upper-triangular True for masked positions
            causal = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
            attn_scores = attn_scores.masked_fill(causal, float('-inf'))
        else:
            attn_scores = attn_scores + attn_mask  # assume mask already -inf on disallowed

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_out = torch.matmul(attn_probs, v)  # (B, h, T, d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        out = self.attn_output(attn_out)
        return out


class StableLMMLP(nn.Module):
    def __init__(self, cfg: StableLMConfig):
        super().__init__()
        d = cfg.embedding_length
        h = cfg.feed_forward_length
        # GGML naming: ffn_up, ffn_gate, ffn_down
        self.ffn_up = nn.Linear(d, h, bias=False)
        self.ffn_gate = nn.Linear(d, h, bias=False)
        self.ffn_down = nn.Linear(h, d, bias=False)
        self.ffn_norm = RMSNorm(d, eps=cfg.layer_norm_epsilon, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx = self.ffn_norm(x)
        up = self.ffn_up(nx)
        gate = self.ffn_gate(nx)
        act = F.silu(gate) * up  # SwiGLU style (silu as gate)
        return self.ffn_down(act)


class StableLMBlock(nn.Module):
    def __init__(self, cfg: StableLMConfig):
        super().__init__()
        self.attn = StableLMAttention(cfg)
        self.mlp = StableLMMLP(cfg)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if True:  # parallel residual per spec
            attn_out = self.attn(x, attn_mask)
            mlp_out = self.mlp(x)
            x = x + attn_out + mlp_out
            return x
        else:
            # (unreachable) sequential for reference
            x = x + self.attn(x, attn_mask)
            x = x + self.mlp(x)
            return x


class StableLM(nn.Module):
    def __init__(self, cfg: StableLMConfig = StableLMConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.embedding_length
        self.token_embd = nn.Embedding(cfg.vocab_size, d)
        self.blocks = nn.ModuleList([StableLMBlock(cfg) for _ in range(cfg.block_count)])
        # output_norm has bias+weight according to the tensor table
        self.output_norm = RMSNorm(d, eps=cfg.layer_norm_epsilon, bias=True)
        # lm head weight in GGML is shaped [d_model, vocab]; PyTorch Linear uses [vocab, d_model]
        self.output = nn.Linear(d, cfg.vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform for linear layers; embeddings normal
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.token_embd.weight, mean=0.0, std=0.02)


    # ----- Forward -----
    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: (B, T)
        x = self.token_embd(input_ids)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.output_norm(x)
        logits = self.output(x)
        return logits

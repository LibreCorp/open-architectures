# python >=3.9, torch >=2.1
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Config (matches your DBRX spec)
# ---------------------------
@dataclass
class DBRXConfig:
    vocab_size: int = 100_352               # inferred from token_embd/output shapes
    context_length: int = 32_768            # dbrx.context_length
    d_model: int = 6_144                    # dbrx.embedding_length
    n_layers: int = 40                      # dbrx.block_count
    n_heads: int = 48                       # dbrx.attention.head_count
    n_kv_heads: int = 8                     # dbrx.attention.head_count_kv (GQA)
    ffn_hidden: int = 10_752                # dbrx.feed_forward_length
    n_experts: int = 16                     # dbrx.expert_count
    top_k: int = 4                          # dbrx.expert_used_count
    rope_base: float = 500_000.0            # dbrx.rope.freq_base
    rms_eps: float = 1e-5                   # dbrx.attention.layer_norm_epsilon
    clamp_kqv: float = 8.0                  # dbrx.attention.clamp_kqv

    bos_token_id: int = 100_257
    eos_token_id: int = 100_257
    pad_token_id: int = 100_277


# ---------------------------
# RMSNorm (weight-only)
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
# RoPE helpers (rotate first 128 dims since head_dim = 128)
# ---------------------------
def _rope_inv_freq(dim: int, base: float, device, dtype):
    return 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))

def apply_rope(q: torch.Tensor, k: torch.Tensor, rope_dim: int, base: float):
    """
    q, k: [B, n, T, H]; rotate first rope_dim dims (<= H).
    """
    b, n, t, h = q.shape
    rope_dim = min(rope_dim, h)
    inv = _rope_inv_freq(rope_dim, base, q.device, torch.float32)
    pos = torch.arange(t, device=q.device, dtype=torch.float32)
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
# Attention (GQA, fused QKV, clamp_kqv)
# ---------------------------
class DBRXAttention(nn.Module):
    """
    Matches your per-tensor names and shapes:

      attn_qkv.weight      : GGML [6144, 8192]  -> PyTorch Linear(in=6144, out=8192) weight [8192, 6144]
         split: Q=6144, K=1024, V=1024 (48Q + 8KV heads with head_dim=128)
      attn_output.weight   : GGML [6144, 6144]  -> PyTorch [6144, 6144]

    Additional per-block norm:
      attn_norm.weight         (RMSNorm)
      attn_output_norm.weight  (RMSNorm)
    """
    def __init__(self, cfg: DBRXConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads  # 6144/48 = 128
        assert self.head_dim * self.n_heads == self.d_model

        self.attn_qkv = nn.Linear(cfg.d_model, cfg.d_model + 2 * (self.n_kv_heads * self.head_dim), bias=False)
        self.attn_output = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope_dim = self.head_dim  # 128
        self.rope_base = cfg.rope_base
        self.clamp = cfg.clamp_kqv

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

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, t, _ = x.shape

        qkv = self.attn_qkv(x)  # [B, T, 8192]
        q, k, v = torch.split(qkv, [self.d_model, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=-1)

        q = self._split_heads(q, self.n_heads)      # [B, 48, T, 128]
        k = self._split_heads(k, self.n_kv_heads)   # [B,  8, T, 128]
        v = self._split_heads(v, self.n_kv_heads)   # [B,  8, T, 128]

        # RoPE on Q,K
        q, k = apply_rope(q, k, self.rope_dim, self.rope_base)

        # Optional clamp for numerical stability (dbrx.attention.clamp_kqv)
        if self.clamp is not None and self.clamp > 0:
            q = torch.clamp(q, -self.clamp, self.clamp)
            k = torch.clamp(k, -self.clamp, self.clamp)
            v = torch.clamp(v, -self.clamp, self.clamp)

        # GQA repeat KV to 48 heads
        k = self._repeat_kv(k)                       # [B, 48, T, 128]
        v = self._repeat_kv(v)                       # [B, 48, T, 128]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, 48, T, T]
        if causal_mask is not None:
            attn_scores = attn_scores + causal_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_out = torch.matmul(attn_probs, v)       # [B, 48, T, 128]
        attn_out = self._merge_heads(attn_out)       # [B, T, 6144]
        out = self.attn_output(attn_out)             # [B, T, 6144]
        return out


# ---------------------------
# MoE FFN (16 experts, top-4, SwiGLU)
# ---------------------------
class DBRXMoE(nn.Module):
    """
    Matches your tensor inventory:

      ffn_up_exps.weight   : GGML [6144, 10752, 16] -> store as [16, 10752, 6144] in PyTorch
      ffn_gate_exps.weight : GGML [6144, 10752, 16] -> store as [16, 10752, 6144]
      ffn_down_exps.weight : GGML [10752, 6144, 16] -> store as [16, 6144, 10752]
      ffn_gate_inp.weight  : GGML [6144, 16]        -> store as [16, 6144]
    """
    def __init__(self, cfg: DBRXConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.ffn_hidden = cfg.ffn_hidden
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k

        # Expert banks as Parameters with explicit names ".weight"
        self.ffn_up_exps = nn.Parameter(torch.empty(cfg.n_experts, self.ffn_hidden, self.d_model))   # [E, H, C]
        self.ffn_gate_exps = nn.Parameter(torch.empty(cfg.n_experts, self.ffn_hidden, self.d_model)) # [E, H, C]
        self.ffn_down_exps = nn.Parameter(torch.empty(cfg.n_experts, self.d_model, self.ffn_hidden)) # [E, C, H]

        # Router: outputs logits over experts
        self.ffn_gate_inp = nn.Parameter(torch.empty(cfg.n_experts, self.d_model))                   # [E, C]

        # Per-sublayer RMSNorm weights live in the block:
        #  - ffn_norm.weight (pre-FFN)
        # No extra biases to match weight-only listing.

        # Simple init so the module runs; real weights will be loaded
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        Router computes top-k experts per token, applies expert SwiGLU,
        and combines outputs with softmax-normalized gates.
        """
        b, t, c = x.shape
        x_flat = x.reshape(b * t, c)  # [BT, C]

        # Router logits: [BT, E] ; weight stored as [E, C]
        logits = F.linear(x_flat, self.ffn_gate_inp)             # [BT, E]
        topk_val, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [BT, K]
        topk_w = F.softmax(topk_val, dim=-1, dtype=torch.float32).to(x.dtype)  # [BT, K]

        # Gather expert weights
        # up/gate: [E, H, C]; down: [E, C, H]
        W_up   = self.ffn_up_exps      # [E, H, C]
        W_gate = self.ffn_gate_exps    # [E, H, C]
        W_down = self.ffn_down_exps    # [E, C, H]

        # Compute each selected expert output efficiently
        # Select weights for the top-k experts per token
        # Shapes after gather:
        #   W_up_sel   : [BT, K, H, C]
        #   W_gate_sel : [BT, K, H, C]
        #   W_down_sel : [BT, K, C, H]
        W_up_sel   = W_up[topk_idx]           # index on first dim E
        W_gate_sel = W_gate[topk_idx]
        W_down_sel = W_down[topk_idx]

        # Compute up and gate activations: [BT, K, H]
        # x_flat: [BT, C]; W_*_sel: [BT, K, H, C]
        u = torch.einsum("bc,bkhc->bkh", x_flat, W_up_sel)
        g = torch.einsum("bc,bkhc->bkh", x_flat, W_gate_sel)
        h = F.silu(g) * u                    # SwiGLU, [BT, K, H]

        # Project down: [BT, K, C]
        y = torch.einsum("bkh,bkch->bkc", h, W_down_sel)

        # Weighted combine experts: [BT, C]
        y = torch.einsum("bkc,bk->bc", y, topk_w)

        y = y.view(b, t, c)
        return y


# ---------------------------
# Transformer Block (Attention + post-Attn Norm + MoE)
# ---------------------------
class DBRXBlock(nn.Module):
    """
    Parameter names align with your table:

      attn_norm.weight
      attn_qkv.weight
      attn_output.weight
      attn_output_norm.weight
      ffn_norm.weight
      ffn_up_exps.weight
      ffn_gate_exps.weight
      ffn_down_exps.weight
      ffn_gate_inp.weight
    """
    def __init__(self, cfg: DBRXConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = DBRXAttention(cfg)
        self.attn_output_norm = RMSNorm(cfg.d_model, cfg.rms_eps)

        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.moe = DBRXMoE(cfg)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Attention sublayer (pre-norm -> Attn -> Residual)
        h = self.attn(self.attn_norm(x), causal_mask)
        x = x + h

        # Post-attention normalization (matches presence of attn_output_norm.weight)
        x = self.attn_output_norm(x)

        # MoE FFN (pre-norm -> MoE -> Residual)
        h2 = self.moe(self.ffn_norm(x))
        x = x + h2
        return x


# ---------------------------
# Full Model
# ---------------------------
class DBRXModel(nn.Module):
    """
    Top-level parameter names and shapes (PyTorch view):
      token_embd.weight        : [vocab, 6144]      # GGML lists [6144, vocab]
      blk.{i}.attn_norm.weight : [6144]
      blk.{i}.attn_qkv.weight  : [8192, 6144]      # GGML lists [6144, 8192]
      blk.{i}.attn_output.weight: [6144, 6144]
      blk.{i}.attn_output_norm.weight: [6144]
      blk.{i}.ffn_norm.weight  : [6144]
      blk.{i}.ffn_up_exps.weight   : [16, 10752, 6144]  # GGML shows [6144, 10752, 16]
      blk.{i}.ffn_gate_exps.weight : [16, 10752, 6144]  # GGML shows [6144, 10752, 16]
      blk.{i}.ffn_down_exps.weight : [16, 6144, 10752]  # GGML shows [10752, 6144, 16]
      blk.{i}.ffn_gate_inp.weight  : [16, 6144]        # GGML shows [6144, 16]
      output_norm.weight       : [6144]
      output.weight            : [vocab, 6144]     # GGML lists [6144, vocab]
    """
    def __init__(self, cfg: DBRXConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embd = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DBRXBlock(cfg) for _ in range(cfg.n_layers)])
        self.output_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.bos_token_id = cfg.bos_token_id
        self.eos_token_id = cfg.eos_token_id
        self.pad_token_id = cfg.pad_token_id

    @staticmethod
    def _build_causal_mask(t: int, device, dtype=torch.float32):
        i = torch.arange(t, device=device)
        j = torch.arange(t, device=device)
        mask = (i[:, None] < j[None, :]).to(dtype) * (-1e9)
        return mask[None, None, :, :]  # [1,1,T,T]

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                return_logits: bool = True):
        """
        input_ids: [B, T]
        attn_mask: additive mask broadcastable to [B, 1, T, T] (optional)
        """
        b, t = input_ids.shape
        x = self.token_embd(input_ids)  # [B, T, 6144]

        causal = attn_mask if attn_mask is not None else self._build_causal_mask(t, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, causal)

        x = self.output_norm(x)
        if return_logits:
            logits = self.output(x)  # [B, T, vocab]
            return logits
        return x  # hidden states


# ---------------------------
# Build helper
# ---------------------------
def build_dbrx_from_spec() -> DBRXModel:
    cfg = DBRXConfig()
    return DBRXModel(cfg)


if __name__ == "__main__":
    cfg = DBRXConfig()
    model = DBRXModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    print(logits.shape)  # [2, 16, 100352]

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # SwiGLU = SiLU(gate) * x
    return F.silu(gate) * x


# ----------------------------
# Rotary Embeddings (RoPE)
# ----------------------------

class RotaryEmbedding(nn.Module):
    """
    RoPE with base theta=5e6 and precomputation for a max_seq_len,
    suitable for native 262,144 tokens. Uses "NTK-aware" scaling via base change.
    """
    def __init__(self, dim: int, base: float = 5e6, max_seq_len: int = 262_144, ntk_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        # Optionally adjust base to extend context (ntk_factor>=1.0)
        base = base * ntk_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        self.register_buffer("cos", torch.cos(emb), persistent=False)
        self.register_buffer("sin", torch.sin(emb), persistent=False)

    def forward(self, x: torch.Tensor, seq_start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, T, H, D] with D == self.dim
        cos = self.cos[seq_start: seq_start + x.size(1)][None, :, None, :]
        sin = self.sin[seq_start: seq_start + x.size(1)][None, :, None, :]
        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, T, H, D]
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ----------------------------
# GQA Attention with QK-Norm
# ----------------------------

class GQAAttention(nn.Module):
    def __init__(self, dim: int, n_q_heads: int, n_kv_heads: int, head_dim: int, rope: RotaryEmbedding, rms_eps: float = 1e-6, dropout: float = 0.0):
        super().__init__()
        assert (n_q_heads % n_kv_heads) == 0, "q heads must be a multiple of kv heads for GQA"
        self.dim = dim
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.h = head_dim
        self.rope = rope
        self.dropout = dropout

        # Projections sized to your dump
        self.q_proj = nn.Linear(dim, self.n_q * self.h, bias=False)   # 4096 -> 8192
        self.k_proj = nn.Linear(dim, self.n_kv * self.h, bias=False)  # 4096 -> 512
        self.v_proj = nn.Linear(dim, self.n_kv * self.h, bias=False)  # 4096 -> 512
        self.o_proj = nn.Linear(self.n_q * self.h, dim, bias=False)   # 8192 -> 4096

        # Q/K RMSNorm (QK-Norm) with dim=head_dim
        self.q_norm = RMSNorm(self.h, eps=rms_eps)
        self.k_norm = RMSNorm(self.h, eps=rms_eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, seq_start: int = 0) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_q, self.h)
        k = self.k_proj(x).view(B, T, self.n_kv, self.h)
        v = self.v_proj(x).view(B, T, self.n_kv, self.h)

        # QK-Norm per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE on first h dims (here entire head_dim)
        cos, sin = self.rope(q, seq_start=seq_start)
        q, k = apply_rotary(q, k, cos, sin)

        # Broadcast K/V across groups for GQA
        # Expand kv heads to match q heads by repeating groups
        repeat_factor = self.n_q // self.n_kv
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [B, Hq, T, D]
        k = k.transpose(1, 2)  # [B, Hq, T, D]
        v = v.transpose(1, 2)  # [B, Hq, T, D]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.h)  # [B,H,T,T]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # mask should be additive with -inf for masked

        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.training and self.dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout)

        ctx = torch.matmul(attn_probs, v)  # [B,H,T,D]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.n_q * self.h)
        out = self.o_proj(ctx)
        return out


# ----------------------------
# MoE with Top-K Router (k=8)
# Expert: SwiGLU(4096->1536->4096)
# ----------------------------

class SwiGLUExpert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        # Two "up" projections (value and gate) and one "down"
        self.wi = nn.Linear(dim, hidden, bias=False)
        self.wg = nn.Linear(dim, hidden, bias=False)
        self.wo = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.wi(x)
        gate = self.wg(x)
        return self.wo(swiglu(up, gate))


class TopKRouter(nn.Module):
    def __init__(self, dim: int, n_experts: int, k: int):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.w_gate = nn.Linear(dim, n_experts, bias=False)  # matches ffn_gate_inp.weight [4096, 128]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        logits = self.w_gate(x)  # [B, T, E]
        gates = F.softmax(logits, dim=-1)  # [B, T, E]
        topk_vals, topk_idx = torch.topk(gates, k=self.k, dim=-1)  # [B, T, k]
        return gates, topk_vals, topk_idx


class MoE(nn.Module):
    def __init__(self, dim: int, n_experts: int, k: int, expert_hidden: int):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.k = k
        self.experts = nn.ModuleList([SwiGLUExpert(dim, expert_hidden) for _ in range(n_experts)])
        self.router = TopKRouter(dim, n_experts, k)

    def forward(self, x: torch.Tensor, return_aux_loss: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Token-wise routing
        B, T, C = x.shape
        gates, topk_vals, topk_idx = self.router(x)  # [B,T,E], [B,T,k], [B,T,k]

        # Capacity-free top-k mixture: weighted sum of selected expert outputs
        out = torch.zeros_like(x)
        # Optional load balancing loss (Shazeer et al.)
        aux_loss = None
        if return_aux_loss:
            me = gates.mean(dim=(0, 1))                      # [E]
            ce = (gates > 0).float().mean(dim=(0, 1))        # crude usage proxy
            aux_loss = (self.n_experts * (me * ce).sum())

        # Compute expert outputs in a grouped way for efficiency
        # Flatten batch and time for indexing
        x_flat = x.view(B * T, C)
        topk_idx_flat = topk_idx.view(B * T, self.k)
        topk_vals_flat = topk_vals.view(B * T, self.k)

        # For each of k, gather, run expert, scatter-add
        for slot in range(self.k):
            idx_e = topk_idx_flat[:, slot]           # [BT]
            w = topk_vals_flat[:, slot].unsqueeze(-1)  # [BT,1]

            # Process tokens per expert (mask and batch)
            # Create a mapping from expert id -> token indices
            for e in range(self.n_experts):
                mask = (idx_e == e)
                if not mask.any():
                    continue
                tokens = x_flat[mask]               # [N_e, C]
                y = self.experts[e](tokens)         # [N_e, C]
                out.view(B * T, C)[mask] += w[mask] * y

        return (out, aux_loss) if return_aux_loss else (out, None)


# ----------------------------
# Transformer Block
# ----------------------------

class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_q_heads: int, n_kv_heads: int, head_dim: int, rope: RotaryEmbedding, n_experts: int, k_experts: int, expert_hidden: int, rms_eps: float = 1e-6, attn_dropout: float = 0.0, resid_dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=rms_eps)
        self.attn = GQAAttention(dim, n_q_heads, n_kv_heads, head_dim, rope, rms_eps=rms_eps, dropout=attn_dropout)

        self.ffn_norm = RMSNorm(dim, eps=rms_eps)
        self.moe = MoE(dim, n_experts=n_experts, k=k_experts, expert_hidden=expert_hidden)

        self.resid_dropout = resid_dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, seq_start: int = 0, return_aux_loss: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention
        a = self.attn_norm(x)
        a = self.attn(a, attn_mask=attn_mask, seq_start=seq_start)
        if self.training and self.resid_dropout > 0:
            a = F.dropout(a, p=self.resid_dropout)
        x = x + a

        # MoE FFN
        m = self.ffn_norm(x)
        m, aux = self.moe(m, return_aux_loss=return_aux_loss)
        if self.training and self.resid_dropout > 0:
            m = F.dropout(m, p=self.resid_dropout)
        x = x + m

        return x, aux


# ----------------------------
# Full Model
# ----------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 151_936
    dim: int = 4096
    n_layers: int = 94
    n_q_heads: int = 64
    n_kv_heads: int = 4
    head_dim: int = 128
    max_seq_len: int = 262_144
    rope_base: float = 5e6
    rope_ntk_factor: float = 1.0
    n_experts: int = 128
    k_experts: int = 8
    expert_hidden: int = 1536
    rms_eps: float = 1e-6
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    tie_output: bool = False  # set True to tie output to embedding


class Qwen3MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.rope = RotaryEmbedding(dim=cfg.head_dim, base=cfg.rope_base, max_seq_len=cfg.max_seq_len, ntk_factor=cfg.rope_ntk_factor)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=cfg.dim,
                n_q_heads=cfg.n_q_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope=self.rope,
                n_experts=cfg.n_experts,
                k_experts=cfg.k_experts,
                expert_hidden=cfg.expert_hidden,
                rms_eps=cfg.rms_eps,
                attn_dropout=cfg.attn_dropout,
                resid_dropout=cfg.resid_dropout,
            ) for _ in range(cfg.n_layers)
        ])

        self.out_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_output:
            self.lm_head.weight = self.tok_emb.weight  # tie weights

    def _causal_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Additive mask with -inf above diagonal
        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask  # [T,T]

    def forward(self, idx: torch.Tensor, seq_start: int = 0, return_aux_loss: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: [B, T] token ids
        return logits [B, T, V] and optional aux load-balancing loss
        """
        B, T = idx.shape
        x = self.tok_emb(idx)  # [B,T,C]

        attn_mask = self._causal_mask(T, x.device, x.dtype)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # broadcast to [1,1,T,T]

        aux_total = 0.0
        for blk in self.blocks:
            x, aux = blk(x, attn_mask=attn_mask, seq_start=seq_start, return_aux_loss=return_aux_loss)
            if aux is not None:
                aux_total = aux_total + aux

        x = self.out_norm(x)
        logits = self.lm_head(x)
        return logits, (aux_total if return_aux_loss else None)

    # ------------------------
    # Parameter accounting
    # ------------------------
    def count_parameters(self, include_embeddings: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if include_embeddings:
            return total
        else:
            emb = self.tok_emb.weight.numel()
            # If tied, lm_head does not add extra parameters beyond embedding
            lm = 0 if self.cfg.tie_output else self.lm_head.weight.numel()
            return total - emb - lm


# ----------------------------
# Quick sanity test and counts
# ----------------------------
if __name__ == "__main__":
    cfg = ModelConfig(
        vocab_size=151_936,
        dim=4096,
        n_layers=94,
        n_q_heads=64,
        n_kv_heads=4,
        head_dim=128,
        max_seq_len=262_144,
        rope_base=5e6,
        n_experts=128,
        k_experts=8,
        expert_hidden=1536,
        tie_output=False,
    )
    model = Qwen3MoE(cfg)

    # Parameter counts
    total_params = model.count_parameters(include_embeddings=True)
    non_emb_params = model.count_parameters(include_embeddings=False)
    print(f"Total params (with embeddings): {total_params/1e9:.3f} B")
    print(f"Non-embedding params: {non_emb_params/1e9:.3f} B")

    # Forward pass shape check
    x = torch.randint(0, cfg.vocab_size, (2, 128))  # small T for test
    logits, aux = model(x, seq_start=0, return_aux_loss=True)
    print("Logits shape:", logits.shape, "| Aux loss:", (aux.item() if aux is not None else None))

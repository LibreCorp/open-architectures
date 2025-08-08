import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config from the spec
# ----------------------------
@dataclass
class Config:
    # model sizes
    d_model: int = 2880
    vocab_size: int = 201088
    n_layers: int = 36

    # attention (GQA)
    n_heads: int = 64
    n_kv_heads: int = 8
    d_q: int = 64
    d_k: int = 64
    d_v: int = 64
    sliding_window: int = 128
    attn_dropout: float = 0.0
    rms_eps: float = 1e-5

    # context + RoPE scaling
    max_seq_len: int = 131072   # 128K
    rope_base: float = 150000.0
    rope_scaling_factor: int = 32
    rope_original_ctx: int = 4096  # linear scaling target

    # MoE
    n_experts: int = 128
    top_k: int = 4
    d_ff: int = 2880  # feed_forward_length (SwiGLU doubles internally)
    dropout: float = 0.0

# ----------------------------
# RMSNorm (instead of LayerNorm)
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms)

# ----------------------------
# SwiGLU MLP (d_model -> 2*d_ff -> d_model)
# ----------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, 2 * d_ff, bias=False)  # produces [U, G]
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        u, g = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)

# ----------------------------
# MoE: token-choice, softmax-after-topk, 128 experts, top-k=4
# ----------------------------
class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ff = SwiGLU(d_model, d_ff)
    def forward(self, x): return self.ff(x)

class TokenChoiceMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(n_experts)])
        self.gate_inp = nn.Linear(d_model, n_experts, bias=True)  # mirrors ffn_gate_inp.*

    def forward(self, x):
        # x: [B,T,D]
        B, T, D = x.shape
        logits = self.gate_inp(x)                   # [B,T,E]
        topv, topi = torch.topk(logits, self.top_k, dim=-1)  # [B,T,top_k]
        weights = F.softmax(topv, dim=-1)           # softmax AFTER top-k

        # compute only selected experts (sparse dispatch)
        y = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topi[..., k]                      # [B,T]
            # group tokens by expert id
            for e in idx.unique():
                mask = (idx == e)                   # [B,T]
                if mask.any():
                    xe = x[mask]                    # [N_e, D]
                    ye = self.experts[int(e)](xe)   # [N_e, D]
                    y[mask] += ye * weights[mask, k].unsqueeze(-1)
        return y

# ----------------------------
# RoPE with linear scaling (factor=32 from 4K -> 128K)
# ----------------------------
def rope_angles(d_head, T, base, scale, device):
    pos = torch.arange(T, device=device, dtype=torch.float32)
    # linear scaling: shrink positions by 'scale' beyond original_ctx
    pos = pos / scale
    idx = torch.arange(0, d_head, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (idx / d_head))
    return torch.einsum('t,f->tf', pos, inv_freq)  # [T, d_head/2]

def apply_rope(x, freqs):
    # x: [B, H, T, Dh]
    x1, x2 = x[..., ::2], x[..., 1::2]
    f = freqs
    sin, cos = f.sin()[None, None, :, :], f.cos()[None, None, :, :]
    xr = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return xr.flatten(-2)

# ----------------------------
# Multi-Head Attention (GQA + sink + alternation)
#   Q: 64 heads × 64
#   K,V: 8  heads × 64  (shared across 64 Q heads by repeating)
# ----------------------------
class RoPEGQAAttn(nn.Module):
    def __init__(self, cfg: Config, use_sliding_window: bool):
        super().__init__()
        self.cfg = cfg
        self.use_sw = use_sliding_window

        q_dim = cfg.n_heads * cfg.d_q                # 4096
        k_dim = cfg.n_kv_heads * cfg.d_k             # 512
        v_dim = cfg.n_kv_heads * cfg.d_v             # 512
        self.qkv = nn.Linear(cfg.d_model, q_dim + k_dim + v_dim, bias=True)

        self.proj_out = nn.Linear(cfg.n_heads * cfg.d_v, cfg.d_model, bias=True)  # 4096 -> 2880

        # RoPE cache (with scaling)
        scale = cfg.rope_scaling_factor
        self.register_buffer(
            "rope_freq",
            rope_angles(cfg.d_q, cfg.max_seq_len, cfg.rope_base, scale, device="cpu"),
            persistent=False
        )

        # per-head learned attention sink (64 heads)
        self.attn_sink = nn.Parameter(torch.zeros(cfg.n_heads))

        self.attn_drop = nn.Dropout(cfg.attn_dropout)

    def _causal(self, T, device):
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        return (j > i)

    def _sliding_mask(self, T, device, window):
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        return (j > i) | ((i - j) > window)

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B,T,5120]
        q, k, v = torch.split(qkv, [self.cfg.n_heads*self.cfg.d_q,
                                    self.cfg.n_kv_heads*self.cfg.d_k,
                                    self.cfg.n_kv_heads*self.cfg.d_v], dim=-1)
        # reshape
        q = q.view(B, T, self.cfg.n_heads, self.cfg.d_q).transpose(1, 2)  # [B,64,T,64]
        k = k.view(B, T, self.cfg.n_kv_heads, self.cfg.d_k).transpose(1, 2)  # [B,8,T,64]
        v = v.view(B, T, self.cfg.n_kv_heads, self.cfg.d_v).transpose(1, 2)  # [B,8,T,64]

        # RoPE on Q,K (use q dim for angles)
        freqs = self.rope_freq[:T].to(q.device)  # [T, 32]
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # GQA: repeat K,V to match 64 Q heads (64/8 = 8)
        repeat = self.cfg.n_heads // self.cfg.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)  # [B,64,T,64]
        v = v.repeat_interleave(repeat, dim=1)  # [B,64,T,64]

        # scaled dot-product
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.cfg.d_q)  # [B,64,T,T]

        # causal + optional sliding window
        mask = self._sliding_mask(T, x.device, self.cfg.sliding_window) if self.use_sw else self._causal(T, x.device)
        logits = logits.masked_fill(mask, float("-inf"))

        if attn_mask is not None:
            logits = logits + attn_mask  # additive mask

        # softmax with learned sink in denominator
        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)      # [B,64,T,1]
        sink = self.attn_sink.view(1, self.cfg.n_heads, 1, 1)          # [1,64,1,1]
        denom_log = torch.logaddexp(logsumexp, sink)                   # [B,64,T,1]
        attn = torch.exp(logits - denom_log)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)                      # [B,64,T,64]
        y = y.transpose(1, 2).contiguous().view(B, T, self.cfg.n_heads * self.cfg.d_v)  # [B,T,4096]
        return self.proj_out(y)                        # [B,T,2880]

# ----------------------------
# Transformer Block (RMSNorm + Attn + MoE)
# ----------------------------
class Block(nn.Module):
    def __init__(self, cfg: Config, use_sliding_window: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_norm  = RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        self.attn = RoPEGQAAttn(cfg, use_sliding_window)
        self.moe  = TokenChoiceMoE(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.moe(self.ffn_norm(x))
        return x

# ----------------------------
# Full Model
# ----------------------------
class TokenChoiceMoETransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        blocks = []
        for i in range(cfg.n_layers):
            # alternate: even=full, odd=sliding-window(128)
            blocks.append(Block(cfg, use_sliding_window=(i % 2 == 1)))
        self.blocks = nn.ModuleList(blocks)

        self.output_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, attn_mask=None):
        x = self.drop(self.tok_emb(idx))  # [B,T,2880]
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.output_norm(x)
        return self.lm_head(x)

# -------------- quick smoke test --------------
if __name__ == "__main__":
    cfg = Config()
    m = TokenChoiceMoETransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 256))
    y = m(x)
    print(y.shape)  # [2, 256, 201088]

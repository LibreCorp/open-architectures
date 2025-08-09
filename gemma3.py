from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- Utils ---------------------------- #

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, name: Optional[str] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self._name = name
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)

# Per-head RMSNorm on the last dim (=head_dim)
class HeadRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, T, head_dim)
        return _rms_norm(x, self.weight, self.eps)

# RoPE with base=10_000 or custom; here base=10_000 to match LLaMA-style unless overridden
class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        # q,k: (B, n, T, D). positions: (T,)
        # Build sinusoid
        freqs = torch.einsum('t,d->td', positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, D)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        def apply_rotary(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
            return x * cos + x_rot * sin
        return apply_rotary(q), apply_rotary(k)

# Sliding-window causal mask (banded attention)
class SlidingWindowMask:
    def __init__(self, window: int):
        self.window = window
    def build(self, qlen: int, klen: int, device: torch.device) -> torch.Tensor:
        # mask True for allowed, False for masked (we'll convert to -inf later)
        i = torch.arange(qlen, device=device)[:, None]
        j = torch.arange(klen, device=device)[None, :]
        # causal and within window
        allowed = (j <= i) & (j >= i - self.window + 1)
        return allowed

# ---------------------------- Attention ---------------------------- #

class Gemma3Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, window: int, attn_ln_eps: float = 1e-6, block_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.window = window
        self.heads_per_kv = n_heads // n_kv_heads

        # Projections match your shapes
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Per-head RMSNorm for q and k (size 128)
        self.q_head_norm = HeadRMSNorm(head_dim, eps=attn_ln_eps)
        self.k_head_norm = HeadRMSNorm(head_dim, eps=attn_ln_eps)

        self.rope = RoPE(head_dim)
        self.mask_builder = SlidingWindowMask(window)

        # register parameter names to match table (for state_dict remapping)
        self.q_proj.weight._name = f"blk.{block_idx}.attn_q.weight"
        self.k_proj.weight._name = f"blk.{block_idx}.attn_k.weight"
        self.v_proj.weight._name = f"blk.{block_idx}.attn_v.weight"
        self.o_proj.weight._name = f"blk.{block_idx}.attn_output.weight"
        self.q_head_norm.weight._name = f"blk.{block_idx}.attn_q_norm.weight"
        self.k_head_norm.weight._name = f"blk.{block_idx}.attn_k_norm.weight"

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_kv, T, D) -> (B, n_heads, T, D)
        if self.n_kv == self.n_heads:
            return x
        b, nk, t, d = x.shape
        x = x[:, :, :, :].unsqueeze(2)  # (B, nk, 1, T, D)
        x = x.expand(b, nk, self.heads_per_kv, t, d)
        return x.reshape(b, nk * self.heads_per_kv, t, d)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)    # (B, K, T, D)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        # Per-head RMSNorm on q,k
        q = self.q_head_norm(q)
        k = self.k_head_norm(k)

        # Rotary embeddings
        if pos is None:
            pos = torch.arange(T, device=x.device)
        q, k = self.rope(q, k, pos)

        # GQA expand kv to H
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Attention with sliding window
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self.mask_builder.build(T, T, x.device)
        attn_scores = attn_scores.masked_fill(~allowed[None, None, :, :], float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn, v)  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        y = self.o_proj(y)
        return y

# ---------------------------- MLP ---------------------------- #

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, block_idx: int = 0):
        super().__init__()
        # up_proj / gate_proj shapes match table
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        # name hints
        self.up_proj.weight._name = f"blk.{block_idx}.ffn_up.weight"
        self.gate_proj.weight._name = f"blk.{block_idx}.ffn_gate.weight"
        self.down_proj.weight._name = f"blk.{block_idx}.ffn_down.weight"
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ---------------------------- Text Block ---------------------------- #

class Gemma3Block(nn.Module):
    def __init__(self, cfg: 'Gemma3Config', idx: int):
        super().__init__()
        self.idx = idx
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.attn = Gemma3Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.sliding_window, cfg.eps, block_idx=idx)
        self.post_attention_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden, block_idx=idx)
        # expose names to match the table
        self.attn_norm.weight._name = f"blk.{idx}.attn_norm.weight"
        self.post_attention_norm.weight._name = f"blk.{idx}.post_attention_norm.weight"
        self.mlp_norm.weight._name = f"blk.{idx}.ffn_norm.weight"

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), pos)
        h = h + self.mlp(self.post_attention_norm(h))
        return h

# ---------------------------- Vision Encoder ---------------------------- #

class ViTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_hidden: int, eps: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.ln2 = nn.LayerNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, mlp_hidden, bias=True)
        self.fc2 = nn.Linear(mlp_hidden, dim, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # simple scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        attn = attn.softmax(dim=-1)
        y = torch.matmul(attn, v)
        x = x + self.proj(y)
        y = self.fc2(F.gelu(self.fc1(self.ln2(x))))
        x = x + y
        return x

class VisionEncoder(nn.Module):
    def __init__(self, cfg: 'Gemma3Config'):
        super().__init__()
        d = cfg.vision_dim
        self.patch = nn.Conv2d(cfg.vision_channels, d, kernel_size=cfg.patch, stride=cfg.patch, bias=True)
        self.pos = nn.Parameter(torch.empty(d, (cfg.image_size // cfg.patch) ** 2))  # (C, 4096)
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(d, cfg.vision_heads, cfg.vision_mlp_hidden, cfg.vision_eps) for _ in range(cfg.vision_blocks)])
        self.post_ln = nn.LayerNorm(d, eps=cfg.vision_eps)
        # register names to help mapping
        self.patch.weight._name = "v.patch_embedding.weight"
        self.patch.bias._name = "v.patch_embedding.bias"
        self.pos._name = "v.position_embedding.weight"
        self.post_ln.weight._name = "v.post_layernorm.weight"
        self.post_ln.bias._name = "v.post_layernorm.bias"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 3, 896, 896)
        x = self.patch(img)  # (B, 1152, 64, 64)
        x = x.flatten(2).transpose(1, 2)  # (B, 4096, 1152)
        x = x + self.pos.transpose(0, 1)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.post_ln(x)
        return x  # (B, 4096, 1152)

class VisionToMM(nn.Module):
    """Produce 256 multimodal tokens from 64x64 patches via 4x4 avg pooling -> 16x16 tokens.
    Each 1152-d token is layernormed then linearly projected to 5376-d text space.
    Weight names: mm.mm_soft_emb_norm.weight (RMSNorm) and mm.mm_input_projection.weight.
    """
    def __init__(self, cfg: 'Gemma3Config'):
        super().__init__()
        self.pool_kernel = 4  # 64->16
        self.norm = RMSNorm(cfg.vision_dim, eps=cfg.eps)
        self.proj = nn.Linear(cfg.vision_dim, cfg.d_model, bias=False)
        # naming
        self.norm.weight._name = "mm.mm_soft_emb_norm.weight"
        self.proj.weight._name = "mm.mm_input_projection.weight"
        self.tokens_per_image = cfg.mm_tokens

    def forward(self, vis_seq: torch.Tensor) -> torch.Tensor:
        # vis_seq: (B, 4096, 1152) from 64x64 grid
        B, N, C = vis_seq.shape
        S = int(math.sqrt(N))  # 64
        x = vis_seq.view(B, S, S, C).permute(0, 3, 1, 2)  # (B, C, 64, 64)
        k = self.pool_kernel
        x = F.avg_pool2d(x, kernel_size=k, stride=k)  # (B, C, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (B, 256, C)
        x = self.norm(x)
        x = self.proj(x)  # (B, 256, d_model)
        return x

# ---------------------------- Model ---------------------------- #

@dataclass
class Gemma3Config:
    d_model: int = 5376
    n_heads: int = 32
    n_kv_heads: int = 16
    head_dim: int = 128
    mlp_hidden: int = 21504
    blocks: int = 62
    context_len: int = 131072
    sliding_window: int = 1024
    eps: float = 1e-6
    vocab_size: int = 262144
    # vision
    image_size: int = 896
    patch: int = 14
    vision_dim: int = 1152
    vision_heads: int = 16
    vision_mlp_hidden: int = 4304
    vision_blocks: int = 27
    vision_eps: float = 1e-6
    vision_channels: int = 3
    # mm
    mm_tokens: int = 256

class Gemma3Model(nn.Module):
    def __init__(self, cfg: Gemma3Config = Gemma3Config()):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Gemma3Block(cfg, i) for i in range(cfg.blocks)])
        self.out_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # names for mapping
        self.tok_embeddings.weight._name = "token_embd.weight"
        self.out_norm.weight._name = "output_norm.weight"

        # vision pathway
        self.vision = VisionEncoder(cfg)
        self.mm = VisionToMM(cfg)

    def forward_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_embeddings(input_ids)
        for blk in self.blocks:
            x = blk(x, pos)
        x = self.out_norm(x)
        logits = self.lm_head(x)
        return logits

    def forward_mm(self, input_ids: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, 896, 896)
        B, T = input_ids.shape
        txt = self.tok_embeddings(input_ids)
        vis_seq = self.vision(images)              # (B, 4096, 1152)
        mm_tokens = self.mm(vis_seq)               # (B, 256, 5376)
        x = torch.cat([mm_tokens, txt], dim=1)     # prepend image tokens
        pos = torch.arange(x.size(1), device=x.device)
        for blk in self.blocks:
            x = blk(x, pos)
        x = self.out_norm(x)
        logits = self.lm_head(x)
        # return logits for all tokens (image+text); callers can slice
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return x_norm * self.scale
        
# Feed-forward with SwiGLU (no internal norm)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj_in  = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim,      bias=False)

    def forward(self, x_norm):
        # x_norm comes already normalized by the outer RMSNorm
        u, v = self.proj_in(x_norm).chunk(2, dim=-1)
        return self.proj_out(F.silu(u) * v)

# Rotary positional embeddings
def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=500000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin()[None, :, None, :], emb.cos()[None, :, None, :]

def apply_rotary(q, k, sin, cos):
    q2 = (q * cos) + (rotate_half(q) * sin)
    k2 = (k * cos) + (rotate_half(k) * sin)
    return q2, k2

 class MoEFeedForward(nn.Module):
    def __init__(self, dim, expert_dim, num_experts, topk=1):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        # two-stage gating
        self.ffn_gate_inp  = nn.Linear(dim, num_experts, bias=False)          # [5120→128]
        self.ffn_gate_exps = nn.Parameter(torch.empty(num_experts, expert_dim), requires_grad=False)
        # per-expert down/up projections (quantized will be loaded here)
        self.ffn_down_shexp = nn.Parameter(torch.empty(expert_dim, dim),       requires_grad=False)
        self.ffn_down_exps  = nn.Parameter(torch.empty(num_experts, expert_dim, dim), requires_grad=False)
        self.ffn_up_shexp   = nn.Parameter(torch.empty(dim, expert_dim),       requires_grad=False)
        self.ffn_up_exps    = nn.Parameter(torch.empty(num_experts, dim, expert_dim), requires_grad=False)

    def forward(self, x):
        B,T,D = x.shape
        # 1) per-token gate hidden
        gate_h = F.silu(self.ffn_gate_inp(x))            # [B,T,num_experts]
        # 2) score each expert: (B,T,num_experts)  — no softmax yet
        #    here we broadcast gate_h * gate_exps weights if needed
        #    for top-k we can just pick indices
        topk_vals, topk_idx = gate_h.topk(self.topk, dim=-1)

        # 3) dispatch to each expert
        out = torch.zeros_like(x)
        x_flat = x.view(-1, D)
        idx_flat = topk_idx.view(-1)
        for e in range(self.num_experts):
            mask = (idx_flat == e)
            if not mask.any(): continue
            inp = x_flat[mask]  # [N_e, D]
            # down_shexp + down_exps[e]
            z = inp @ self.ffn_down_shexp.t() + inp @ self.ffn_down_exps[e].t()
            z = F.silu(z)
            # up_shexp + up_exps[e]
            y = z @ self.ffn_up_shexp.t() + z @ self.ffn_up_exps[e].t()
            out.view(-1, D)[mask] = y

        return out.view(B, T, D)

# Efficient MoE FeedForward
class MoEFeedForward(nn.Module):
    def __init__(self, dim, expert_dim, num_experts, topk=1):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.experts = nn.ModuleList([FeedForward(dim, expert_dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        scores = self.router(x)  # [B,T,E]
        probs  = F.softmax(scores, dim=-1) # [B,T,E]
        # 3) get top-k expert indices
        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
        _, idx = scores.topk(self.topk, dim=-1)  # [B,T,topk]
        idx = idx.view(-1)
        x_flat = x.view(-1, D)
        out_flat = torch.zeros_like(x_flat)
        # dispatch per expert only on selected tokens
        for e, expert in enumerate(self.experts):
            mask = (idx == e)
            if mask.any():
                inp = x_flat[mask]
                out_flat[mask] = expert(inp.unsqueeze(1)).squeeze(1)
        return out_flat.view(B, T, D)

# Transformer block with MoE and chunked attention
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, heads_kv, head_dim, ffn_dim,
                 use_moe=False, moe_dim=None, moe_experts=None, chunk_size=8192):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, heads=40, heads_kv=8, head_dim=128, chunk_size=8192, rope_base=500_000)
        self.ffn_norm = RMSNorm(dim)
        if use_moe:
            self.ffn = MoEFeedForward(dim, moe_dim, moe_experts)
        else:
            self.ffn = FeedForward(dim, ffn_dim)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

# Vision Transformer block remains unchanged
class VisionBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, ffn_dim, rope_base=10000):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        # use correct RoPE base for vision
        self.attn = MultiHeadAttention(dim, heads=16, heads_kv=16, head_dim=88, chunk_size=None, rope_base=10_000)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

# Full Llama4 model with spec fixes
class Llama4(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_length = 1_048_576

        # Token embedding and language blocks
        self.token_emb = nn.Embedding(202048, 5120)
        self.blocks = nn.ModuleList()
        for i in range(48):
            use_moe = (i % 2 == 1)
            self.blocks.append(
                TransformerBlock(
                    dim=5120,
                    heads=40,
                    heads_kv=8,
                    head_dim=128,
                    ffn_dim=16384,
                    use_moe=use_moe,
                    moe_dim=8192,
                    moe_experts=128,
                    chunk_size=8192
                )
            )
        self.norm_out = RMSNorm(5120)
        # Tie output head to token embeddings for exact parity
        self.head = nn.Linear(5120, 202048, bias=False)
        self.head.weight = self.token_emb.weight

        # Vision encoder components
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, 1408, kernel_size=14, stride=14)
        # Class token and positional embeddings
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, 1408))
        num_patches = (336 // 14) ** 2
        self.positional_embedding_vlm = nn.Parameter(torch.zeros(1, num_patches + 1, 1408))

        # Pre & post RMS norms
        self.layernorm_pre = RMSNorm(1408)
        self.layernorm_post = RMSNorm(1408)

        # Vision transformer blocks
        self.v_blocks = nn.ModuleList([
            VisionBlock(
                dim=1408,
                heads=16,
                head_dim=88,
                ffn_dim=5632,
                rope_base=10000
            )
            for _ in range(34)
        ])

        # Pixel-unshuffle for adapter input
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        # Adapter MLP: 4096->5632->4096
        self.vision_preproj   = nn.Linear(5632, 4096, bias=True)
        self.vision_adapter = nn.Sequential(
            nn.SiLU(),
            nn.Linear(4096, 5632, bias=True),
            nn.SiLU(),
            nn.Linear(5632, 4096, bias=True)
        )
        
        # Project adapter output into language dimension
        self.lv_proj = nn.Linear(4096, 5120, bias=False)

    def forward(self, tokens, images=None):
        # Language trunk
        x = self.token_emb(tokens)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)

        if images is not None:
            B = images.shape[0]
            # Patch embeddings
            v = self.patch_embed(images)                   # [B,1408,24,24]
            v = v.flatten(2).transpose(1, 2)                # [B,576,1408]

            # Prepend class token and add positional embeddings
            cls = self.class_embedding.expand(B, -1, -1)   # [B,1,1408]
            v = torch.cat([cls, v], dim=1)                 # [B,577,1408]
            v = v + self.positional_embedding_vlm         # broadcast add

            # Pre-norm, transformer blocks, post-norm
            v = self.layernorm_pre(v)
            for vb in self.v_blocks:
                v = vb(v)
            v = self.layernorm_post(v)

            # Adapter pathway
            p_feat = self.patch_embed(images)         # [B,1408,24,24]
            p = self.pixel_unshuffle(p_feat)          # [B,5632,12,12]
            p = p.flatten(2).transpose(1, 2)          # [B,144,5632]
            p = self.vision_preproj(p)                # [B,144,4096]
            p = self.vision_adapter(p)                # [B,144,4096]
            p = self.lv_proj(p)

            # Concatenate modalities
            x = torch.cat([x, p], dim=1)

        return self.head(x)


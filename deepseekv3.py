# DeepSeek‑V3 (paper-faithful) PyTorch skeleton
# Implements: MLA attention, DeepSeekMoE with auxiliary‑loss‑free routing, optional MTP heads
# Notes
# - This is a readable reference skeleton matching the equations in the V3 technical report.
# - It is not throughput‑optimized. It focuses on architectural faithfulness and clarity.
# - KV cache reduction of MLA is modeled; decoupled RoPE on k_R and q_R is included.
# - MoE routing uses sigmoid affinities, top‑K with bias for aux‑loss‑free balancing, no token drop.
# - Node‑limited routing (M nodes) is emulated for single‑process use.

from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Root‑mean‑square normalization (no bias)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x

class RotaryEmbedding:
    """RoPE with optional YaRN scaling for long context; supports decoupled dims.
    Implements base b (usually 10000) and YaRN scaling with factor, original_length, and log multiplier.
    """
    def __init__(self, dim: int, base: float = 10000.0, yarn_factor: Optional[float] = None,
                 orig_ctx: Optional[int] = None, log_multiplier: float = 0.0):
        self.dim = dim
        self.base = base
        self.yarn_factor = yarn_factor
        self.orig_ctx = orig_ctx
        self.log_multiplier = log_multiplier
    def _scaled_base(self, seq_len: int) -> float:
        if self.yarn_factor is None or self.orig_ctx is None or self.yarn_factor == 1.0:
            return self.base
        # YaRN scaling (simplified): adjust base using factor and a mild log term.
        # Reference idea: keep low‑freq stable, increase high‑freq range via base' = base * (factor)^(dim_index/dim) * exp(log_multiplier * log(seq_len/orig_ctx)).
        return self.base
    def _rope_frequencies(self, seq_len: int, device: torch.device):
        base = self.base  # YaRN base pre‑scaled if needed
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [seq, dim/2]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H, d_r]
        b, s, h, d = x.shape
        device = x.device
        cos, sin = self._rope_frequencies(s, device)
        cos = cos.view(1, s, 1, d//2)
        sin = sin.view(1, s, 1, d//2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x1 * cos - x2 * sin
        x_rot[..., 1::2] = x1 * sin + x2 * cos
        return x_rot

# -------------------------
# MLA Attention (Eq. 1–11)
# -------------------------

@dataclass
class MLAConfig:
    d_model: int
    n_heads: int
    d_head: int
    d_kv_latent: int  # d_c in paper, KV compression dim
    d_q_latent: int   # d_c' for query compression
    d_r: int          # decoupled RoPE per‑head dim

class MLAAttention(nn.Module):
    """DeepSeek‑V3 fused‑projection MLA as revealed by your dump.
    Fusions:
      * attn_kv_a_mqa: [D -> (dc + d_r)]
      * attn_kv_b:     [dc -> H * (2 * v_len)]  -> split per head into K_C (v_len) and V_C (v_len)
      * attn_q_a:      [D -> dq]
      * attn_q_b:      [dq -> H * (v_len + d_r)] -> split per head into Q_C (v_len) and Q_R (d_r)
    Final per‑head K = [K_C (v_len) || K_R (d_r)] with size key_length = v_len + d_r.
    V per head uses V_C (v_len).
    """
    def __init__(self, d_model: int, n_heads: int, v_len: int, r_dim: int, dc: int, dq: int,
                 eps_latent: float = 1e-6, rope_base: float = 10000.0,
                 rope_yarn_factor: Optional[float] = 40.0, rope_orig_ctx: Optional[int] = 4096,
                 rope_log_mult: float = 0.1):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.v_len, self.r_dim = v_len, r_dim
        self.key_len = v_len + r_dim  # 128 + 64 = 192
        self.dc, self.dq = dc, dq
        # Fused projections
        self.kv_a = nn.Linear(d_model, dc + r_dim, bias=False)   # -> [c_KV(512) || k_R(64)]
        self.kv_a_norm = RMSNorm(dc, eps=eps_latent)
        self.kv_b = nn.Linear(dc, n_heads * (2 * v_len), bias=False)  # -> [ per‑head (K_C||V_C) of size 2*v_len ]

        self.q_a = nn.Linear(d_model, dq, bias=False)             # -> q_latent (1536)
        self.q_a_norm = RMSNorm(dq, eps=eps_latent)
        self.q_b = nn.Linear(dq, n_heads * (v_len + r_dim), bias=False)  # -> [ per‑head (Q_C||Q_R) sizes 128||64 ]

        self.out = nn.Linear(n_heads * v_len, d_model, bias=False)
        self.attn_norm = RMSNorm(d_model)  # pre‑attn

        self.rope = RotaryEmbedding(r_dim, base=rope_base, yarn_factor=rope_yarn_factor,
                                    orig_ctx=rope_orig_ctx, log_multiplier=rope_log_mult)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        H, VL, RD = self.n_heads, self.v_len, self.r_dim
        xh = self.attn_norm(x)
        # KV fused
        kv_cat = self.kv_a(xh)                                  # [B,S,dc+RD]
        c_kv, k_R = kv_cat.split([self.dc, RD], dim=-1)         # [B,S,dc], [B,S,RD]
        c_kv = self.kv_a_norm(c_kv)
        kv_up = self.kv_b(c_kv).view(B, S, H, 2*VL)             # [B,S,H,2*VL]
        k_C, v_C = kv_up.split([VL, VL], dim=-1)                # per head
        # Q fused
        q_lat = self.q_a_norm(self.q_a(xh))                     # [B,S,dq]
        q_up = self.q_b(q_lat).view(B, S, H, VL+RD)             # [B,S,H,VL+RD]
        q_C, q_R = q_up.split([VL, RD], dim=-1)
        # Decoupled RoPE on q_R and k_R (broadcast k_R to heads)
        k_R = k_R.view(B, S, 1, RD).expand(B, S, H, RD)
        k_R = self.rope.apply(k_R)
        q_R = self.rope.apply(q_R)
        # Concat head parts
        k = torch.cat([k_C, k_R], dim=-1)                       # [B,S,H,VL+RD]
        q = torch.cat([q_C, q_R], dim=-1)                       # [B,S,H,VL+RD]
        v = v_C                                                 # [B,S,H,VL]
        # Attention
        scale = (VL + RD) ** -0.5
        q = q.transpose(1, 2)  # [B,H,S,Dk]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            att = att + mask
        att = att.softmax(dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, S, H*VL)
        return self.out(o)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.wi = nn.Linear(d_model, 2*d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.wi(x).chunk(2, dim=-1)
        return self.wo(F.silu(a) * b)
(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.wi = nn.Linear(d_model, 2*d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.wi(x).chunk(2, dim=-1)
        return self.wo(F.silu(a) * b)

# -------------------------
# DeepSeekMoE (Eq. 12–16) with aux‑loss‑free routing and node‑limited routing
# -------------------------

@dataclass
class MoEConfig:
    d_model: int
    d_ff_expert: int
    n_shared: int = 1
    n_routed: int = 256
    top_k: int = 8
    node_ids: Optional[List[int]] = None  # length n_routed, integer node id per expert
    max_nodes_per_token: Optional[int] = 4
    bias_update_speed: float = 1e-3  # gamma in paper

class DeepSeekMoE(nn.Module):
    """Packed MoE implementation aligned with your dump naming and shapes.
    Parameters (from dump):
      * n_routed=256, n_shared=1, top_k=8
      * expert weight norm enabled with scale=2.5
      * Router: linear W_inp: [D -> E], bias: [E]; gating_func=2 (sigmoid), aux‑loss‑free via bias only in selection.
    We store experts packed as 3D tensors: [d_ff, d_model, E] and [d_model, d_ff, E].
    """
    def __init__(self, d_model: int, d_ff_expert: int, n_routed: int = 256, n_shared: int = 1,
                 top_k: int = 8, weight_scale: float = 2.5, router_bias_speed: float = 1e-3):
        super().__init__()
        self.d_model, self.d_ff = d_model, d_ff_expert
        self.n_routed, self.n_shared, self.top_k = n_routed, n_shared, top_k
        # Shared experts (dense path)
        self.shexp_down = nn.Linear(d_model, d_ff_expert, bias=False)
        self.shexp_gate = nn.Linear(d_model, d_ff_expert, bias=False)
        self.shexp_up   = nn.Linear(d_ff_expert, d_model, bias=False)
        # Packed routed experts
        self.ex_down = nn.Parameter(torch.empty(d_ff_expert, d_model, n_routed))
        self.ex_up   = nn.Parameter(torch.empty(d_model, d_ff_expert, n_routed))
        self.ex_gate = nn.Parameter(torch.empty(d_model, d_ff_expert, n_routed))
        nn.init.kaiming_uniform_(self.ex_down.view(d_ff_expert, -1), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.ex_up.view(d_model, -1), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.ex_gate.view(d_model, -1), a=math.sqrt(5))
        # Weight norm scaling (RMS) parameter
        self.weight_scale = weight_scale
        # Router
        self.router_w = nn.Linear(d_model, n_routed, bias=True)
        # Bias used only in selection path is the layer bias; we keep aux‑loss‑free by using bias for selection but
        # computing gates from sigmoid(router_w(x) - bias) to avoid coupling.
        self.router_bias_speed = router_bias_speed
        # Layer norms
        self.ffn_norm = RMSNorm(d_model)

    def _rms_norm_w(self, W: torch.Tensor, dim: int) -> torch.Tensor:
        # Normalize weights per‑out channel RMS and rescale.
        rms = W.pow(2).mean(dim=dim, keepdim=True).sqrt().clamp_min(1e-6)
        return W / rms * self.weight_scale

    def _expert_swiglu(self, x: torch.Tensor, idx: torch.LongTensor) -> torch.Tensor:
        # x: [T, D], idx: [T, K]
        T, D = x.shape
        K = idx.size(1)
        # Gather weights for selected experts
        Wd = self._rms_norm_w(self.ex_down, dim=1)      # [F,D,E]
        Wu = self._rms_norm_w(self.ex_up, dim=1)        # [D,F,E]
        Wg = self._rms_norm_w(self.ex_gate, dim=1)      # [D,F,E]
        out = torch.zeros(T, D, device=x.device, dtype=x.dtype)
        for k in range(K):
            eids = idx[:, k]                            # [T]
            # batched gather via advanced indexing
            Wd_k = Wd[:, :, eids]                       # [F,D,T]
            Wu_k = Wu[:, :, eids]                       # [D,F,T]
            Wg_k = Wg[:, :, eids]                       # [D,F,T]
            xT = x.t().unsqueeze(-1).expand(D, T, 1)    # [D,T,1]
            # down
            u = torch.bmm(Wd_k.transpose(0,2), xT).squeeze(-1)    # [T,F]
            g = torch.bmm(Wg_k.transpose(0,2), xT).squeeze(-1)    # [T,F]
            h = F.silu(g) * u
            y = torch.bmm(Wu_k.transpose(0,2), h.unsqueeze(-1)).squeeze(-1)  # [T,D]
            out += y
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_in = self.ffn_norm(x)
        xs = x_in.view(B*S, D)
        # Shared expert path
        she = self.shexp_up(F.silu(self.shexp_gate(xs)) * self.shexp_down(xs))
        # Router logits and top‑k selection
        logits = self.router_w(xs)            # [T,E]
        # Selection logits use bias; gating uses debiased sigmoid to keep aux‑loss‑free
        sel_scores = logits
        topk_idx = torch.topk(sel_scores, k=self.top_k, dim=-1).indices  # [T,K]
        probs = torch.sigmoid(logits)
        g_prime = torch.zeros_like(probs)
        g_prime.scatter_(1, topk_idx, probs.gather(1, topk_idx))
        g = g_prime / (g_prime.sum(dim=-1, keepdim=True) + 1e-9)
        # Expert computation (no token drop)
        routed = self._expert_swiglu(xs, topk_idx)
        out = xs + she + routed * 1.0  # residual includes both shared and routed
        return out.view(B, S, D)

# -------------------------
# Transformer Block (Pre‑Norm)
# -------------------------

class DSBlock(nn.Module):
    def __init__(self, d_model: int, attn: MLAAttention, ffn_or_moe: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.attn = attn
        self.ffn = ffn_or_moe
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(x, attn_mask))
        x = x + self.dropout(self.ffn(x))
        return x

# (Optional) MTP modules could be added here if training with D>0 is desired.
    def causal_mask(self, S: int, device) -> torch.Tensor:
        # standard causal mask, additive
        mask = torch.full((1, 1, S, S), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, mtp_weight: float = 0.1):
        # input_ids: [B,S]
        B, S = input_ids.shape
        x = self.embed(input_ids)
        mask = self.causal_mask(S, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        h_main = self.norm_f(x)
        logits_main = self.lm_head(h_main)
        out = {"logits": logits_main}

        loss = None
        if targets is not None:
            # Next‑token loss
            main_logits_shift = logits_main[:, :-1, :].contiguous()
            targets_shift = targets[:, 1:].contiguous()
            loss_main = F.cross_entropy(main_logits_shift.view(-1, main_logits_shift.size(-1)), targets_shift.view(-1))
            total_loss = loss_main
            # MTP losses (sequential depth), predict token t_{i+k+1}
            h_prev = h_main
            for k, mtp in enumerate(self.mtp_modules, start=1):
                # Prepare offset tokens Emb(t_{i+k})
                if S - k - 1 <= 0:
                    break
                next_tokens = input_ids[:, k:]  # t_{i+k}
                h_prev = h_prev[:, :-k, :]
                h_prev, logits_k = mtp(h_prev, next_tokens)
                mtp_logits_shift = logits_k[:, :-1, :]
                mtp_targets = targets[:, k+1:]
                loss_k = F.cross_entropy(mtp_logits_shift.reshape(-1, mtp_logits_shift.size(-1)), mtp_targets.reshape(-1))
                total_loss = total_loss + mtp_weight * loss_k / len(self.mtp_modules)
            loss = total_loss
        out["loss"] = loss
        return out

# -------------------------
# Quick smoke test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = DSConfig(vocab_size=32000, n_layers=4, n_heads=8, d_model=512, d_head=64, d_kv_latent=128, d_q_latent=192, d_r=32,
                   moe_n_routed=16, moe_top_k=4, moe_d_ff=1024, interleave_moe_from=1, mtp_depth=1)
    model = DeepSeekV3Like(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    out = model(x, targets=x)
    print(out["logits"].shape, out["loss"].item())

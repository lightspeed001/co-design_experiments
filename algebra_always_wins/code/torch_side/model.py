"""
model.py — the model as the reader actually writes it, plus the arithmetic the post quotes.

Nothing here is exotic: this is a vanilla pre-norm decoder block. The point of the
file is that the *definition* is written in a vocabulary (modules, heads, residuals)
that appears nowhere in the profile, and that a small amount of arithmetic on the
shapes already tells you which side of the roofline each part runs on.

Run:
    python model.py            # prints param counts, per-token bytes, and the AI table
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# The definition layer: modules, heads, residuals. This is effective theory level 0.
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    d_model: int = 4096
    n_heads: int = 32
    n_layers: int = 32
    d_ff: int = 14336          # SwiGLU inner width, ~3.5 x d_model
    vocab: int = 128_000

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """Reads as a normalization. Compiles to: reduce, rsqrt, broadcast multiply.

    Note for the post: this is *not* a matmul, and it is roughly 3 full passes over
    the activations. It is memory traffic with a little arithmetic attached.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))


class Attention(nn.Module):
    """Multi-head self-attention, written two ways.

    `explicit=True`  — the pedagogical form: softmax(QK^T / sqrt(d))V, with the N^2
                       score matrix materialized.
    `explicit=False` — the form that ships: a fused kernel whose whole trick is that
                       softmax's max and log-sum-exp are *associative*, so the scores
                       never have to exist in memory.

    Both are the same algebra. Only one of them respects the memory hierarchy.
    """

    def __init__(self, cfg: Config, explicit: bool = False):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.explicit = explicit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        # (B, T, C) -> (B, H, T, dh): a reshape *and* a transpose. In the compiler this
        # becomes a layout assertion, and layout is where "algebra" meets "bytes".
        q = q.view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)

        if self.explicit:
            # Two GEMMs with an N^2 tensor between them. At T=4096 this intermediate is
            # 32 heads x 4096 x 4096 x 2 B = 1 GiB per layer, per batch element.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.cfg.d_head)
            probs = torch.softmax(scores, dim=-1)
            y = probs @ v
        else:
            # Fused. Same algebra, reassociated; the score matrix never lands in HBM.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    """SwiGLU. Three matmuls and one multiply: the 3rd matmul is the 'cheap' one only
    if you are looking at FLOPs. At decode shapes it is the expensive one."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # up
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)   # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """One decoder block. Two residuals: two elementwise adds. Free in FLOPs,
    ~2 x d_model x 2 B of traffic per token, every layer, forever."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------------------
# The arithmetic. This is the part of the post you can check without a GPU.
# --------------------------------------------------------------------------------------
def per_token_bytes(cfg: Config, dtype_bytes: int = 2) -> dict[str, float]:
    """Weight bytes touched to produce ONE token, grouped by where they live.

    At batch size 1 every weight is read exactly once — so this is also the HBM traffic
    per token, and it sets a hard ceiling on decode speed that no kernel can beat.
    """
    qkv = 3 * cfg.d_model * cfg.d_model
    proj = cfg.d_model * cfg.d_model
    mlp = 3 * cfg.d_model * cfg.d_ff          # w1 + w3 + w2
    attn = cfg.n_layers * (qkv + proj)
    ffn = cfg.n_layers * mlp
    norm = cfg.n_layers * 2 * cfg.d_model     # RMSNorm weights, negligible but real
    return {
        "attention projections": attn * dtype_bytes,
        "mlp": ffn * dtype_bytes,
        "norms": norm * dtype_bytes,
        "total": (attn + ffn + norm) * dtype_bytes,
    }


def kv_cache_bytes(cfg: Config, ctx: int, batch: int = 1, dtype_bytes: int = 2) -> float:
    """K and V for every layer: 2 x layers x ctx x d_model x bytes (per batch item)."""
    return 2 * cfg.n_layers * ctx * cfg.d_model * dtype_bytes * batch


def arithmetic_intensity(m: int, n: int, k: int, bytes_per_elem: int = 2) -> dict:
    """FLOPs, bytes and AI for a dense (m x k) @ (k x n) with a distinct C.

    The 2 in `2*m*n*k` is one multiply + one add. If you count only multiplies you will
    be off by 2x for the rest of your career.
    """
    flops = 2 * m * n * k
    # Lower bound on traffic: each operand must be read at least once, C written once.
    # (With tiling, A and B get re-read; this is the *irreducible* part.)
    nbytes = (m * k + k * n + m * n) * bytes_per_elem
    return {"flops": flops, "bytes": nbytes, "ai": flops / nbytes}


def ridge_point(tflops: float = 989.4, tbytes_s: float = 3.35) -> float:
    """H100 SXM5: ~989 TFLOP/s dense BF16 (1979 is the 2:4 *sparse* marketing number),
    ~3.35 TB/s HBM3. The ratio is the machine's opinion about what shape your work
    should be in."""
    return (tflops * 1e12) / (tbytes_s * 1e12)


if __name__ == "__main__":
    cfg = Config()
    b = per_token_bytes(cfg)

    print("=" * 78)
    print(f"7B-shaped config: {cfg.n_layers}L x {cfg.d_model}d, {cfg.n_heads} heads, "
          f"d_ff={cfg.d_ff}")
    print("=" * 78)
    for k_, v_ in b.items():
        print(f"  {k_:24s} {v_ / 1e9:8.2f} GB  (bf16)")

    # The single most useful number in inference: every parameter is read once per
    # token and costs 2 FLOPs, so bf16 decode has an arithmetic intensity of
    #   2 FLOP / 2 B = 1 FLOP/byte
    # *by construction*, independent of model size. The ridge is at 295. Decode is
    # therefore ~295x on the wrong side of the machine, always, forever, until you
    # batch it, quantize it, or both.
    n_params = cfg.n_layers * (4 * cfg.d_model**2 + 3 * cfg.d_model * cfg.d_ff)
    print(f"\n  core params:                {n_params / 1e9:8.2f} B")
    print(f"  whole-model decode AI:      {2 * n_params / b['total']:8.2f} FLOP/byte "
          f"(ridge = {ridge_point():.0f})")

    # A 3.35 TB/s HBM stack reading `total` bytes per token sets the decode ceiling.
    ceiling = 3.35e12 / b["total"]
    print(f"\n  decode ceiling @ 3.35 TB/s: {ceiling:,.1f} tok/s  "
          f"(= {1e3 / ceiling:.2f} ms/token, before KV reads)")
    print(f"  KV cache @ 8k ctx:          {kv_cache_bytes(cfg, 8192) / 1e9:.2f} GB")
    print(f"  KV read per token @ 8k:     {kv_cache_bytes(cfg, 8192) / 1e6:.0f} MB "
          f"(+{kv_cache_bytes(cfg, 8192) / b['total'] * 100:.0f}% traffic)")

    print("\n" + "=" * 78)
    print(f"Roofline: ridge point = {ridge_point():.0f} FLOP/byte")
    print("=" * 78)
    print(f"{'case':38s} {'GFLOP':>10s} {'MB':>9s} {'AI':>9s} {'bound':>10s}")
    cases = [
        ("prefill GEMM  4096x4096x4096", 4096, 4096, 4096),
        ("prefill GEMM  2048x4096x4096", 2048, 4096, 4096),
        ("decode  GEMM  batch 32, 4096x4096x4096", 32, 4096, 4096),
        ("decode  GEMV  batch 1,  1x4096x4096", 1, 4096, 4096),
    ]
    for name, m, n, k in cases:
        r = arithmetic_intensity(m, n, k)
        bound = "compute" if r["ai"] > ridge_point() else "MEMORY"
        print(f"{name:38s} {r['flops'] / 1e9:10.1f} {r['bytes'] / 1e6:9.2f} "
              f"{r['ai']:9.1f} {bound:>10s}")

    print("\nNote: the GEMV row is a *vector*, not a matrix. Its FLOPs are irrelevant;")
    print("its bytes are the entire cost. Same operator, 295x further from the ridge.")

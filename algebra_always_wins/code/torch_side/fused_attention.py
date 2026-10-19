"""
fused_attention.py — the FlashAttention forward pass, in ~60 lines of Triton.

Why this file exists in a post about lowering: it is the clearest example of the
thesis in human-authored code. FlashAttention is usually described as an "IO-aware
exact attention algorithm" (true). It is also, in compiler terms, a *reassociation
proof*: max and log-sum-exp form a monoid, so the softmax over the key dimension can
be computed in blocks, accumulated, and corrected on the fly. Once you see that, the
kernel is just two dots with a monoid sandwich — which is exactly what you will see
again in section 5's `vector.contract` -> `nvgpu.mma.sync` lowering.

Run (needs a CUDA GPU):
    python fused_attention.py
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    # ---------------------------------------------------------------------------------
    # 1. Program mapping. Q is tiled along the sequence axis; each program owns one
    #    BLOCK_M x HEAD_DIM slab of queries and walks the *whole* K/V stream.
    #    In MLIR this is a `scf.forall` over (M, batch*heads) plus a reduction loop:
    #    exactly the tiling we will watch appear in the lowering, except that here a
    #    human wrote it.
    # ---------------------------------------------------------------------------------
    start_m = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_z, off_h = off_zh // H, off_zh % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn

    # ---------------------------------------------------------------------------------
    # 2. The monoid's identity elements. (m, l, acc) is the running summary of
    #    everything seen so far. This triple *is* the online-softmax state; there is no
    #    other place the N^2 score matrix could hide.
    # ---------------------------------------------------------------------------------
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], tl.float32)                 # running sum of exp
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)       # running output

    # exp2 is cheaper than exp; fold the log2(e) into the scale once, up front.
    qk_scale = (1.0 / math.sqrt(HEAD_DIM)) * 1.44269504
    q = tl.load(q_ptrs)                                   # (BLOCK_M, HEAD_DIM)
    q = (q * qk_scale).to(tl.float16)

    # ---------------------------------------------------------------------------------
    # 3. The reduction loop. Each step is: matmul, online-softmax update, matmul.
    #    Every step is *the same algebra* regardless of BLOCK_N — that is what makes
    #    the tiling legal. Change BLOCK_N from 64 to 128 and you change the schedule,
    #    not the result (to within fp rounding — see section 7 of the post).
    # ---------------------------------------------------------------------------------
    for start_n in range(0, (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn)         # (BLOCK_N, HEAD_DIM)

        qk = tl.dot(q, tl.trans(k))                       # (BLOCK_M, BLOCK_N) — GEMM #1

        if IS_CAUSAL:
            # Mask future positions with -inf so exp gives exactly 0.
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, float("-inf"))

        # --- online softmax: three lines that are a monoid combine ---------------
        m_new = tl.maximum(m_i, tl.max(qk, 1))            # associative max
        alpha = tl.exp2(m_i - m_new)                      # rescale for the new max
        l_i = l_i * alpha + tl.sum(tl.exp2(qk - m_new[:, None]), 1)
        acc = acc * alpha[:, None]                        # correct the past
        m_i = m_new
        # -------------------------------------------------------------------------

        p = tl.exp2(qk - m_i[:, None]).to(tl.float16)     # probabilities, unnormalized
        v = tl.load(v_ptrs + start_n * stride_vk)         # (BLOCK_N, HEAD_DIM)
        acc = tl.dot(p, v, acc)                           # GEMM #2, accumulating

    # ---------------------------------------------------------------------------------
    # 4. Final normalization: divide once, at the end, instead of every block.
    #    Note we never materialized (BLOCK_M x N_CTX). That is the entire trick.
    # ---------------------------------------------------------------------------------
    acc = acc / l_i[:, None]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(o_ptrs, acc.to(tl.float16))


def attention(q, k, v, causal=True, block_m=128, block_n=64):
    """q,k,v: (Z, H, N_CTX, HEAD_DIM), contiguous. Returns (Z, H, N_CTX, HEAD_DIM)."""
    Z, H, N_CTX, HEAD_DIM = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, block_m), Z * H)
    _attn_fwd[grid](
        q, k, v, out,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        Z, H, N_CTX,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=HEAD_DIM, IS_CAUSAL=causal,
        num_warps=8, num_stages=3,
    )
    return out


def reference(q, k, v, causal=True):
    """The explicit form. Same math, N^2 bytes. This is the thing FlashAttention beats."""
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        n = s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(n, n, device=s.device, dtype=torch.bool), 1), float("-inf"))
    return torch.softmax(s, dim=-1) @ v


if __name__ == "__main__":
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA; the arithmetic you can check without a GPU is in model.py")

    # Decode-shaped (one query token) and prefill-shaped (whole prompt).
    for name, n_ctx in [("decode  T=1", 1), ("prefill T=4096", 4096)]:
        q, k, v = (torch.randn(1, 32, n_ctx, 128, device="cuda", dtype=torch.float16)
                   for _ in range(3))
        out = attention(q, k, v)
        ref = reference(q, k, v)
        err = (out.float() - ref.float()).abs().max().item()
        flops = 4 * 32 * n_ctx * n_ctx * 128 if n_ctx > 1 else 4 * 32 * 1 * 128  # ~2 GEMMs
        print(f"{name:16s} max|err| = {err:.4e}   ~{flops / 1e9:8.2f} GFLOP")

    # Same kernel, two BLOCK_N: the schedule moves, the answer does not (much).
    q, k, v = (torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16) for _ in range(3))
    a = attention(q, k, v, block_n=64)
    b = attention(q, k, v, block_n=128)
    print(f"BLOCK_N 64 vs 128: max|diff| = {(a - b).abs().max().item():.4e}  "
          f"<- same algebra, different summation tree")

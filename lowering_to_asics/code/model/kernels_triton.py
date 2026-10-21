"""Triton kernels for TinyMoE — the *tile-level* language of the series.

These kernels are the portable middle of the story: Triton programs are
tile-level (blocks of data in, blocks of data out), which is precisely the
abstraction level that maps well onto ASIC compute fabrics. In later posts we
lower equivalents of these four kernels to each target's IR.

NOTE ON EXECUTION (why CUDA shows up): Triton's mature backends emit GPU
machine code (NVPTX → CUDA, AMDGCN, ...). Running these kernels needs a GPU.
On CPU-only machines, use the PyTorch paths in tiny_moe.py — they are the
reference numerics we check every lowering against. (Triton-CPU exists but is
experimental; see post 0.1, "Why C++/CUDA (still) enters here".)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: tiled GEMM. The kernel code/lowerer/src/main.rs lowers *this* shape
# (C += A @ B) from tile IR to loops for us.
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes are compile-time constants in Triton: the *hardware tile* is
    # baked into the schedule here, exactly like a scratchpad tile on an ASIC.
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32 regardless of element type (standard practice).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-dimension loop = the reduction the ASIC will either unroll (Groq),
    # stream through a systolic array (TPU), or keep as a loop (Tensix).
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)          # <-- tl.dot is the tensor-engine / MXU op
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Host-side launcher. `grid` = how the machine's parallelism is exposed."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# ---------------------------------------------------------------------------
# Kernel 3: RMSNorm. Reads the row twice, writes once — pure bandwidth.
# ---------------------------------------------------------------------------
@triton.jit
def rms_norm_kernel(
    x_ptr, y_ptr, w_ptr,
    stride_x, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # Reduction over the row: on wide vector engines this is one pass; on
    # narrow ones it becomes tree-shaped work in the schedule.
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    y = (x * rstd * w).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_x + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 4: fused SwiGLU elementwise (silu(a) * b). Memory-bound, and the
# classic candidate for epilogue fusion into the preceding GEMM.
# ---------------------------------------------------------------------------
@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, out_ptr, n_elements,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0)
    silu = g * tl.sigmoid(g)        # swish/silu
    tl.store(out_ptr + offs, silu * u, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: attention forward, flash-style (online softmax). Written as one
# kernel on purpose: fusion of QK^T / softmax / PV is exactly what ASIC stacks
# must reproduce (TPU: MXU + vector unit in Mosaic; Trainium: tensor+vector
# engines; Groq: fully unrolled into the static schedule).
# ---------------------------------------------------------------------------
@triton.jit
def attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    S, sm_scale,
    stride_qb, stride_qh,
    stride_kb, stride_kh,
    stride_vb, stride_vh,
    stride_ob, stride_oh,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program = one (batch, head, query-tile) slice.
    start_m = tl.program_id(0)
    head = tl.program_id(1)
    bh = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + bh * stride_qb + head * stride_qh
    k_base = k_ptr + bh * stride_kb + head * stride_kh
    v_base = v_ptr + bh * stride_vb + head * stride_vh

    q = tl.load(q_base + offs_m[:, None] * stride_qh + offs_d[None, :],
                mask=offs_m[:, None] < S, other=0.0)

    # Online softmax state: running max and running denominator.
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # Causal: a query tile at start_m only ever sees key tiles 0..start_m.
    # Turning a data-independent mask into a *loop bound* is the oldest
    # lowering trick in the book — we will do it in Rust for the TPU post too.
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[:, None] * stride_kh + offs_d[None, :],
                    mask=offs_n[:, None] < S, other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vh + offs_d[None, :],
                    mask=offs_n[:, None] < S, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale          # phase 1: QK^T
        qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, float("-inf"))

        # Online softmax update (fuse-safe: no full S×S materialization).
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])                  # phase 2: exp
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)   # phase 3: PV
        m_i = m_ij

    o = acc / l_i[:, None]
    tl.store(o_ptr + bh * stride_ob + head * stride_oh
             + offs_m[:, None] * stride_oh + offs_d[None, :],
             o.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < S)

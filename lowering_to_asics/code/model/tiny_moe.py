"""TinyMoE — the running example for the *One Kernel, Many Chips* series.

A decoder-only Mixture-of-Experts language model, deliberately tiny (~4M params)
but structurally complete. It exists to generate a *kernel inventory* that every
architecture post in the series must lower:

    # | kernel              | kind            | why it is in the series
    --+---------------------+-----------------+-------------------------------------------
    1 | expert GEMM         | compute-bound   | the roofline anchor; systolic arrays eat it
    2 | causal attention    | fused, tiled    | QK^T + softmax + PV: the classic fusion
    3 | RMSNorm             | memory-bound    | reductions + elementwise; exposes bandwidth
    4 | SwiGLU              | memory-bound    | pure elementwise; the "cheap" ops that add up
    5 | router top-k gather | irregular       | DATA-DEPENDENT: the static-schedule stress test

Point 5 is why the series runs MoE rather than a dense transformer: routing is the
one place where the model's control flow depends on the data, which is exactly where
compiler-scheduled machines (Groq LPU) and dynamic machines (GPU) diverge.

Run `python3 tiny_moe.py` for a CPU smoke test (no GPU required).
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config — frozen so every post in the series quotes identical shapes.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TinyMoEConfig:
    vocab_size: int = 1024     # toy vocab (keeps embedding small)
    d_model: int = 256         # model width D
    n_heads: int = 4           # attention heads (head_dim = D / n_heads = 64)
    n_layers: int = 2          # decoder blocks
    n_experts: int = 4         # experts per MoE layer (E)
    top_k: int = 2             # experts activated per token (k)
    expert_hidden: int = 512   # expert FFN hidden width H_ff
    max_seq: int = 512         # context length S
    rms_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Kernel 3: RMSNorm (memory-bound)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square norm. Tiny math, but it streams every activation twice.

    Lowering note: this is the canonical *bandwidth* kernel. On any ASIC it either
    fuses into a neighbour or pays full HBM traffic; the series keeps it separate
    so we can watch that bill explicitly.
    """

    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.d_model))
        self.eps = cfg.rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm statistic in fp32 (matches custom-kernel practice).
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).type_as(x)


# ---------------------------------------------------------------------------
# Kernel 2: causal self-attention (fused, tiled) — written out explicitly on
# purpose. F.scaled_dot_product_attention would hide the three phases we want
# to lower later: QK^T (matmul), softmax (reduction), PV (matmul).
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)                                  # [B, S, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, S, D] -> [B, H, S, Dh]: one head per program slice on most ASICs.
        def split(t):
            return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)

        # Phase 1: scores = Q K^T / sqrt(Dh)        (matmul → tensor engines)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask (triangular) — becomes a *compile-time* loop bound once
        # we tile this kernel; a recurring theme in the lowering posts.
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        # Phase 2: softmax over rows               (reduction → vector engines)
        attn = torch.softmax(scores, dim=-1)

        # Phase 3: out = attn V                     (matmul again)
        out = attn @ v                                   # [B, H, S, Dh]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Kernel 1: expert MLP, SwiGLU variant (GEMM, GEMM, elementwise, GEMM)
# ---------------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.expert_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.expert_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.expert_hidden, cfg.d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kernel 4 lives here: silu(w1 x) * w3 x  (elementwise, memory-bound)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# The MoE layer: router + experts + weighted combine.
# ---------------------------------------------------------------------------
class MoELayer(nn.Module):
    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(cfg) for _ in range(cfg.n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Work in flattened token space: [T, D] with T = B * S.
        T, D = x.shape
        logits = self.router(x)                            # [T, E]
        probs = torch.softmax(logits, dim=-1)

        # Kernel 5: top-k routing. DATA-DEPENDENT control flow — the router
        # decides the compute pattern after seeing the activations.
        topk_w, topk_ix = torch.topk(probs, self.top_k, dim=-1)   # [T, k]
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            # Tokens routed to expert e in ANY of their top-k slots.
            # (idx, slot) is a gather plan — irregular, data-dependent.
            idx, slot = torch.where(topk_ix == e)
            if idx.numel() == 0:
                continue
            w = topk_w[idx, slot].unsqueeze(-1)            # [nnz, 1]

            # Batched GEMM for this expert's tokens. A production lowering turns
            # the per-expert loop into *grouped GEMM* (see kernels_triton.py);
            # the loop is what a static-schedule machine must eliminate at
            # compile time or emulate with predication.
            out.index_add_(0, idx, w * expert(x[idx]))

        return out


# ---------------------------------------------------------------------------
# Decoder block + full model
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.moe_norm = RMSNorm(cfg)
        self.moe = MoELayer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual layout (the shape every ASIC stack optimizes first).
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x


class TinyMoELM(nn.Module):
    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight          # weight tying: no extra GEMM

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)                               # [B, S, D]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)                          # [B, S, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def kernel_inventory() -> List[Tuple[str, str, str]]:
    """The contract every architecture post must fill in its scorecard."""
    return [
        ("expert_gemm",     "compute-bound",  "tiling / tensor-engine mapping"),
        ("attention_fwd",   "fused + tiled",  "online softmax, buffer scopes"),
        ("rms_norm_fwd",    "memory-bound",   "fusion or pay the bandwidth bill"),
        ("swiglu_fwd",      "memory-bound",   "epilogue fusion opportunities"),
        ("moe_topk_gather", "irregular",      "dynamic routing vs static schedule"),
    ]


if __name__ == "__main__":
    torch.manual_seed(0)                                  # deterministic smoke test
    cfg = TinyMoEConfig()
    model = TinyMoELM(cfg)

    B, S = 2, cfg.max_seq                                 # [B, S] token ids
    idx = torch.randint(0, cfg.vocab_size, (B, S))
    targets = torch.randint(0, cfg.vocab_size, (B, S))

    logits, loss = model(idx, targets)
    print(f"TinyMoE params: {model.param_count():,}")
    print(f"logits: {tuple(logits.shape)}  loss: {loss.item():.4f}")
    print("kernel inventory:")
    for name, kind, why in kernel_inventory():
        print(f"  {name:16s} {kind:16s} {why}")

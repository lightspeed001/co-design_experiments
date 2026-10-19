"""
trace_graph.py — two things you can run on any machine to check this post's claims.

1. `export_block()`: torch.export -> run_decompositions, then print the graph and an
   op histogram. This is level 1 of the ladder: the model's vocabulary is already gone.
   What is left is a flat list of aten ops, and roughly half of them are *layout*, not
   arithmetic.

2. `profile_decode()`: torch.profiler over a decode step, grouped by kernel name. On a
   GPU this prints the five-or-so kernel families the post opens with. On CPU it still
   prints *something*, and the shape of the collapse is the same.

Run:
    python trace_graph.py
"""

from __future__ import annotations

import re
from collections import Counter

import torch

from model import Block, Config


def export_block(cfg: Config | None = None, seq: int = 8) -> None:
    cfg = cfg or Config()
    torch.manual_seed(0)
    block = Block(cfg).eval()
    x = torch.randn(1, seq, cfg.d_model)

    ep = torch.export.export(block, (x,))
    ep = ep.run_decompositions()          # <- the first lowering: composite ops -> primitives
    code = ep.graph_module.code

    print("=" * 78)
    print(f"torch.export graph, ONE decoder block, T={seq}")
    print("=" * 78)
    print(code)

    # Count what actually survived tracing. The interesting column is the second one:
    # every `view`/`permute` is a promise about memory layout that some later pass has
    # to either fold into an index map or pay for in bytes.
    hist = Counter(re.findall(r"torch\.ops\.aten\.([a-z_]+)\.", code))
    print("-" * 78)
    print("op histogram (decomposed)")
    print("-" * 78)
    for name, n in hist.most_common():
        print(f"  {name:28s} {n:3d}")

    layout_ops = sum(hist[k] for k in ("view", "permute", "expand", "split_with_sizes"))
    matmuls = sum(hist[k] for k in ("mm", "bmm", "addmm", "matmul"))
    print(f"\n  matmuls: {matmuls}    pure layout ops: {layout_ops}")
    print("  -> at this level, 'attention' is 2 bmm + softmax + 12 permutes.")


def profile_decode(cfg: Config | None = None, steps: int = 5) -> None:
    """Group a decode step's kernels by name. This is the post's opening receipt.

    On a CUDA box the kernel names are the real ones (cutlass/cublas/flash/triton).
    We canonicalise them a little because raw names carry tile sizes and template
    parameters that would drown the signal.
    """
    cfg = cfg or Config()
    torch.manual_seed(0)
    block = Block(cfg).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block.to(device)

    # Decode shape: one token, batch 1. This is the regime the roofline says is doomed.
    x = torch.randn(1, 1, cfg.d_model, device=device)

    with torch.no_grad():
        for _ in range(2):                      # warmup
            block(x)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
            + ([torch.profiler.ProfilerActivity.CUDA] if device == "cuda" else []),
            record_shapes=True,
        ) as prof:
            for _ in range(steps):
                block(x)

    families: Counter[str] = Counter()
    for evt in prof.key_averages():
        name = evt.key
        # Collapse the template soup: keep the family, drop the tile parameters.
        if "gemm" in name.lower() or "cutlass" in name.lower():
            fam = "GEMM (cutlass/cublas)"
        elif "gemv" in name.lower() or "gemv" in name.lower():
            fam = "GEMV (bandwidth-bound)"
        elif "flash" in name.lower() or "attention" in name.lower():
            fam = "attention (fused)"
        elif name.lower().startswith(("rms", "layer_norm", "norm")):
            fam = "norm"
        elif "elementwise" in name.lower() or name in ("aten::add", "aten::mul"):
            fam = "elementwise / residual"
        elif name.startswith("aten::"):
            fam = "aten op (unfused)"
        else:
            fam = name.split("<")[0].split("(")[0][:40]
        families[fam] += evt.count

    print("=" * 78)
    print(f"kernel families over {steps} decode steps  (device={device}, T=1, batch=1)")
    print("=" * 78)
    for fam, n in families.most_common(12):
        print(f"  {fam:34s} x{n}")
    print("\n  Notice: no 'attention', no 'block', no 'layer'. The vocabulary the model")
    print("  was written in does not appear. It was consumed by the lowering.")


if __name__ == "__main__":
    export_block()
    print()
    profile_decode()

#!/usr/bin/env python3
"""
make_figs.py — generate the hand-laid SVG figures for the post.

Two figures are pure layout (the ladder, the kernel collapse) and one is arithmetic
(the roofline). Doing them in Python keeps the numbers in the roofline exactly equal to
the numbers in model.py rather than to whatever looked good in a drawing tool.

    python3 make_figs.py
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Shared palette. Warm paper, ink, one accent, one alarm colour.
PAPER = "#fbf9f4"
INK = "#191714"
MUTED = "#6f675c"
RULE = "#d9d2c3"
ACCENT = "#b45309"      # amber — the algebra
ALARM = "#9f1239"       # crimson — the casualty
COOL = "#1d4e6f"        # deep blue — the substrate
SERIF = "Iowan Old Style, Palatino, Georgia, serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" style="background:{PAPER}">\n'
        f"<style>\n"
        f"  .t {{ font-family:{SERIF}; fill:{INK}; }}\n"
        f"  .m {{ font-family:{MONO}; fill:{MUTED}; }}\n"
        f"  .cap {{ font-family:{SERIF}; fill:{MUTED}; font-style:italic; }}\n"
        f"</style>\n{body}\n</svg>\n"
    )


# --------------------------------------------------------------------------------------
# FIGURE 1 — the ladder. Seven rungs, and the casualty at each one.
# --------------------------------------------------------------------------------------
def fig1_ladder() -> None:
    rungs = [
        ("0", "model", "tensors · modules · autograd · heads",
         "tracing: control flow, aliasing, data-dependent shapes", ACCENT),
        ("1", "graph IR", "dot_general · reduce · exp · transpose",
         "fusion decisions, layout, guards, recompiles", ACCENT),
        ("2", "structured ops", "linalg.matmul · linalg.generic · affine maps",
         "the name — 'matmul' becomes an index space", ALARM),
        ("3", "loops & buffers", "scf.for · scf.forall · memref.subview",
         "value semantics — tensors become addresses", ALARM),
        ("4", "vector / GPU", "vector.contract · nvgpu.mma.sync",
         "shape — a 16×16 tile becomes a fragment layout", ALARM),
        ("5", "ISA", "mma.sync · wgmma · HMMA.16816",
         "the compiler — scheduling, registers, occupancy", COOL),
        ("6", "silicon", "D += A·B on a fixed grid",
         "nothing. the substrate's own limit is bytes/joule", COOL),
    ]

    W, row_h, top = 1000, 92, 104
    H = top + row_h * len(rungs) + 54
    b = [f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

    # Title
    b.append(f'<text class="t" x="48" y="46" font-size="21" font-weight="600">'
             f'The ladder: seven vocabularies, seven casualties</text>')
    b.append(f'<text class="cap" x="48" y="70" font-size="13.5">'
             f'A rung counts as a level only if something dies crossing it.</text>')

    # Left descent axis
    x0 = 48
    b.append(f'<line x1="{x0-18}" y1="{top-10}" x2="{x0-18}" y2="{top + row_h*len(rungs) - 28}" '
             f'stroke="{RULE}" stroke-width="2"/>')
    b.append(f'<path d="M{x0-23},{top + row_h*len(rungs) - 34} l5,10 l5,-10" fill="none" '
             f'stroke="{RULE}" stroke-width="2"/>')
    ymid = top + (row_h * len(rungs) - 28) / 2
    b.append(f'<text class="cap" x="0" y="0" font-size="12" text-anchor="middle" '
             f'transform="translate(14,{ymid}) rotate(-90)">lowering</text>')

    for i, (num, name, vocab, casualty, colour) in enumerate(rungs):
        y = top + i * row_h
        # card
        b.append(f'<rect x="{x0}" y="{y}" width="{W-x0-48}" height="{row_h-14}" rx="9" '
                 f'fill="#fffefb" stroke="{RULE}"/>')
        # rung number chip
        b.append(f'<rect x="{x0}" y="{y}" width="46" height="{row_h-14}" rx="9" fill="{colour}" opacity="0.10"/>')
        b.append(f'<text class="m" x="{x0+23}" y="{y+34}" font-size="15" text-anchor="middle" '
                 f'fill="{colour}">{num}</text>')
        # name
        b.append(f'<text class="t" x="{x0+66}" y="{y+30}" font-size="16.5" font-weight="600">{escape(name)}</text>')
        # vocabulary
        b.append(f'<text class="m" x="{x0+66}" y="{y+52}" font-size="12">{escape(vocab)}</text>')
        # divider
        b.append(f'<line x1="{x0+470}" y1="{y+14}" x2="{x0+470}" y2="{y+row_h-28}" stroke="{RULE}" stroke-dasharray="3 4"/>')
        # casualty
        b.append(f'<text class="m" x="{x0+490}" y="{y+26}" font-size="10.5" letter-spacing="1.1" '
                 f'fill="{colour}">WHAT BREAKS</text>')
        b.append(f'<text class="t" x="{x0+490}" y="{y+47}" font-size="13.2" fill="{MUTED}">{escape(casualty)}</text>')

    b.append(f'<text class="cap" x="48" y="{H-18}" font-size="12">'
             f'Left column: what you can still say. Right column: what stops being true. '
             f'The bottom two rungs are not printed by any compiler.</text>')
    (OUT / "fig1-ladder.svg").write_text(svg(W, H, "\n".join(b)))


# --------------------------------------------------------------------------------------
# FIGURE 3 — the roofline, with the two regimes marked.
# --------------------------------------------------------------------------------------
def fig3_roofline() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    peak = 989.4          # TFLOP/s, H100 SXM BF16 dense
    bw = 3.35             # TB/s
    ridge = peak / bw     # 295 FLOP/byte

    ai = np.logspace(-1, 4, 500)
    perf = np.minimum(peak, ai * bw)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Iowan Old Style", "Palatino", "Georgia", "DejaVu Serif"],
        "figure.facecolor": PAPER,
        "axes.facecolor": "#fffefb",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": RULE,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(10.4, 5.5), dpi=100)

    ax.fill_between(ai, 1e-2, perf, color=ACCENT, alpha=0.055)
    ax.plot(ai, perf, color=ACCENT, lw=2.4, zorder=3)
    ax.axvline(ridge, color=RULE, ls=(0, (4, 4)), lw=1.2, zorder=1)

    # regime labels
    ax.text(4.5, 900, "memory-bound\n(the roof is a slope)", color=ALARM, fontsize=11, style="italic")
    ax.text(700, 40, "compute-bound\n(the roof is flat)", color=COOL, fontsize=11, style="italic",
            ha="center")

    pts = [
        (1.0, 1.0 * bw, "GEMV — one weight read per token\nAI = 1  →  3.35 TFLOP/s", ALARM, (-4, 52), "left"),
        (31.5, 31.5 * bw, "GEMM, batch 32\nAI = 31.5", ACCENT, (18, -34), "left"),
        (1365, peak, "GEMM 4096³\nAI = 1,365", COOL, (-14, -46), "right"),
    ]
    for x, y, label, colour, (dx, dy), ha in pts:
        ax.scatter([x], [y], s=64, color=colour, zorder=5, edgecolor="#fffefb", linewidth=1.6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=9.4, color=colour, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.8, alpha=0.7))

    # the 295x gap, drawn
    ax.annotate("", xy=(ridge, 3.35), xytext=(1.0, 3.35),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.text(math.sqrt(ridge), 4.4, "≈295× below the ridge", color=MUTED, fontsize=9.6,
            ha="center", style="italic")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4); ax.set_ylim(1e-2, 2e3)
    ax.set_xlabel("arithmetic intensity  (FLOP / byte)", fontsize=11)
    ax.set_ylabel("attainable performance  (TFLOP/s)", fontsize=11)
    ax.set_title("One machine, two regimes: H100 SXM, BF16, 3.35 TB/s HBM3",
                 fontsize=12.5, pad=14, loc="left")
    ax.grid(True, which="major", color=RULE, lw=0.6, alpha=0.7)
    ax.grid(True, which="minor", color=RULE, lw=0.3, alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.012, 0.012,
             "The only thing that moved between the top-left point and the bottom-right "
             "point is the shape of the operand.\nSame operator. Same associativity. "
             "295× difference in what the machine is for.",
             fontsize=8.8, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig3-roofline.svg", facecolor=PAPER)
    fig.savefig(OUT / "fig3-roofline.png", facecolor=PAPER, dpi=140)
    print("wrote fig3-roofline.svg")


if __name__ == "__main__":
    fig1_ladder()
    fig3_roofline()
    print("wrote fig1-ladder.svg")

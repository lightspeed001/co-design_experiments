#!/usr/bin/env python3
"""
crop_figs.py — trim the dead canvas around mermaid's output.

`mmdc -w 1000` pads a graph to a 1000px-wide canvas regardless of how much it actually
draws, so a wide-and-short flowchart arrives with a screenful of empty paper above and
below it. This measures each SVG's real drawing bbox (via headless Chrome, see
crop_svg.mjs) and rewrites the viewBox to match.

    node ../tools/crop_svg.mjs ../figures | python3 crop_figs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FIGS = Path(__file__).resolve().parent.parent / "figures"
PAD = 14


def crop(path: Path, box: dict) -> None:
    s = path.read_text()
    x, y = box["x"] - PAD, box["y"] - PAD
    w, h = box["w"] + 2 * PAD, box["h"] + 2 * PAD
    # <svg ... viewBox="0 0 W H" ...>  ->  cropped viewBox and intrinsic size
    s = re.sub(r'viewBox="[^"]*"', f'viewBox="{x:.1f} {y:.1f} {w:.1f} {h:.1f}"', s, count=1)
    s = re.sub(r'\swidth="[^"]*"', f' width="{w:.0f}"', s, count=1)
    s = re.sub(r'\sheight="[^"]*"', f' height="{h:.0f}"', s, count=1)
    path.write_text(s)
    print(f"  {path.name:18s} {w:6.0f} x {h:5.0f}   (aspect {w/h:.2f})")


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            continue
        box = json.loads(line)
        crop(FIGS / box["file"], box)

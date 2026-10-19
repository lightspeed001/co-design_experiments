#!/usr/bin/env python3
"""
build_html.py — render post.md as a single self-contained HTML file.

Why this exists: the in-app preview sandbox has no network, so nothing external
(fonts, CDN mermaid, stylesheets) can load. So we inline everything: figures become
base64 data-URIs, Pygments writes the syntax colours at build time, and the CSS is a
<style> block. One file, no dependencies, drop it anywhere.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "post.md"
OUT = ROOT / "algebra-always-wins.html"

# MLIR has no Pygments lexer; LLVM is the closest relative (and honestly not that close).
LEXER_ALIASES = {"mlir": "text", "sass": "asm"}   # MLIR: plain — the LLVM lexer mangles it


class _InlineFiguresProcessor:
    """Replace ![alt](figures/x.svg) with a data-URI <img>, so the file is portable."""

    def __init__(self, md):
        self.md = md

    @staticmethod
    def _data_uri(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    @staticmethod
    def _display_width(path: Path) -> str:
        """Very tall diagrams get a narrower column so they stay on one screen."""
        try:
            head = path.read_text()[:2000]
            m = re.search(r'viewBox="[\d.eE+-]+ [\d.eE+-]+ ([\d.]+) ([\d.]+)"', head)
            if not m:
                return "100%"
            aspect = float(m.group(1)) / float(m.group(2))
        except Exception:
            return "100%"
        if aspect < 0.45:
            return "46%"
        if aspect < 0.9:
            return "72%"
        return "100%"

    def run(self, root):
        def walk(el):
            if el.tag == "img":
                src = el.get("src", "")
                if src.startswith("figures/"):
                    path = ROOT / src
                    if path.exists():
                        el.set("src", self._data_uri(path))
                        el.set("class", "fig")
                        if path.suffix == ".svg":
                            el.set("style", f"width:{self._display_width(path)};margin-inline:auto")
            for child in list(el):
                walk(child)

        walk(root)
        return root


class InlineFigures(markdown.extensions.Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(_InlineFiguresProcessor(md), "inline_figures", 5)


CSS = """
:root {
  --paper: #fbf9f4; --ink: #191714; --muted: #6f675c; --rule: #d9d2c3;
  --accent: #b45309; --alarm: #9f1239; --cool: #1d4e6f;
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
* { box-sizing: border-box; }
html { background: var(--paper); }
body {
  margin: 0 auto; padding: 4rem 1.6rem 8rem; max-width: 46rem;
  background: var(--paper); color: var(--ink);
  font-family: var(--serif); font-size: 1.185rem; line-height: 1.66;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
}
h1, h2, h3 { font-family: var(--sans); font-weight: 650; letter-spacing: -0.018em; line-height: 1.2; }
h1 { font-size: 2.55rem; margin: 0 0 0.5rem; }
h1 + p em { font-size: 1.06rem; color: var(--muted); }
h2 { font-size: 1.52rem; margin: 3.6rem 0 1.1rem; padding-top: 1.3rem; border-top: 1px solid var(--rule); }
h3 { font-size: 1.15rem; margin: 2.4rem 0 0.7rem; color: var(--cool); }
p { margin: 0 0 1.15rem; }
a { color: var(--accent); text-decoration: none; border-bottom: 1px solid rgba(180,83,9,.28); }
a:hover { border-bottom-color: var(--accent); }
strong { font-weight: 650; }
em { }
hr { border: 0; border-top: 1px solid var(--rule); margin: 3rem 0; }
blockquote {
  margin: 1.7rem 0; padding: 0.1rem 0 0.1rem 1.25rem;
  border-left: 3px solid var(--accent); color: #2e2a24;
}
blockquote p { margin: 0.65rem 0; }
code { font-family: var(--mono); font-size: 0.855em; background: #f1ece1; padding: 0.1em 0.32em; border-radius: 3px; }
pre { font-family: var(--mono); font-size: 0.752rem; line-height: 1.55; overflow-x: auto;
      background: #fffefb; border: 1px solid var(--rule); border-radius: 8px;
      padding: 1rem 1.15rem; margin: 1.5rem 0; }
pre code { background: none; padding: 0; font-size: 1em; }
.codehilite { background: #fffefb; }
.codehilite pre { margin: 0; border: 0; background: none; }
.codehilite { border: 1px solid var(--rule); border-radius: 8px; margin: 1.5rem 0; overflow: hidden; }
.codehilite pre { padding: 1rem 1.15rem; }
img.fig { display: block; width: 100%; height: auto; margin: 2rem 0 0.6rem; border-radius: 8px; }
img.fig + p em, p em:only-child { color: var(--muted); }
table { border-collapse: collapse; width: 100%; margin: 1.6rem 0; font-size: 0.9rem; font-family: var(--sans); }
th { text-align: left; font-weight: 650; border-bottom: 1.5px solid var(--rule); padding: 0.5rem 0.6rem; }
td { border-bottom: 1px solid var(--rule); padding: 0.5rem 0.6rem; vertical-align: top; }
td code { font-size: 0.82rem; }
ul, ol { margin: 0 0 1.2rem; padding-left: 1.35rem; }
li { margin: 0.35rem 0; }
hr + h2 { border-top: 0; padding-top: 0; margin-top: 2.2rem; }
.lede { font-style: italic; color: #3a352d; }
"""


def main() -> None:
    text = SRC.read_text()
    # strip YAML front matter (we hard-code the title treatment below)
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)
    text = text.replace("<!--", "<!").replace("-->", ">")  # no html comments in source

    def fence(m: re.Match) -> str:
        lang = (m.group(1) or "").strip().lower()
        body = m.group(2)
        if lang == "math":
            return m.group(0)
        lexer = LEXER_ALIASES.get(lang, lang)
        return f"```{lexer}\n{body}```"

    text = re.sub(r"^```(\w*)\n(.*?)^```$", fence, text, flags=re.S | re.M)

    html = markdown.markdown(
        text,
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "attr_list",
            "sane_lists",
            # NOTE: no `smarty` — it rewrites quotes *inside* code blocks and turns
            # perfectly good Rust into typographic garbage.
            InlineFigures(),
        ],
        extension_configs={
            "codehilite": {"guess_lang": False, "css_class": "codehilite", "noclasses": False}
        },
    )

    # Pygments wraps every token in a colour span, which turns code into confetti and
    # fights the essay palette. Un-wrap the colourless token classes; keep comments,
    # keywords, numbers and strings so there is still structure to read.
    NOPAINT = re.compile(
        r'<span class="(?:n|nv|nf|nn|nb|bp|nc|nd|ne|na|no|nt|vm|vi|vg|err|o|ow|p|w)">(.*?)</span>',
        re.S,
    )

    def calm(m: re.Match) -> str:
        block = m.group(0)
        for _ in range(3):
            new = NOPAINT.sub(r"\1", block)
            if new == block:
                break
            block = new
        return block

    html = re.sub(r"<pre>.*?</pre>", calm, html, flags=re.S)

    pyg = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Algebra always wins</title>
<style>{CSS}\n{pyg}\n.codehilite .c1, .codehilite .c {{ color: #8a8175; font-style: italic; }}
.codehilite .mi, .codehilite .m {{ color: #9f1239; }}
.codehilite .k, .codehilite .kd, .codehilite .kn {{ color: #1d4e6f; }}
.codehilite .s, .codehilite .s1, .codehilite .s2 {{ color: #2f6b3a; }}
.codehilite .nf, .codehilite .nv {{ color: #191714; }}
pre {{ color: #191714; }}
</style>
</head>
<body>
<h1>Algebra always wins</h1>
<p><em>How every architecture dissolves into matmuls, and why the substrate chose them</em></p>
<hr>
{html}
</body>
</html>
"""
    OUT.write_text(out)
    kb = len(out.encode()) / 1024
    print(f"wrote {OUT.name}  ({kb:,.0f} KB, self-contained)")


if __name__ == "__main__":
    main()

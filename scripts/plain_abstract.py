#!/usr/bin/env python3
"""The abstract as arXiv's submission form wants it: plain text, no markup.

The form is a textarea, not a LaTeX document, so `$288$` and `\texttt{bash}` reach the
listing page verbatim if the abstract is pasted straight out of the manuscript. Flattening
it by hand is the kind of step that gets done once correctly and then drifts, which this
repository has spent a session demonstrating: the same figure stated on six surfaces took
four passes to correct.

    ./scripts/plain_abstract.py

Written beside the upload package by `make arxiv`, and gitignored, since it is derived from
`paper/anvil.tex` and regenerating it is cheaper than trusting a copy.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "anvil.tex"
OUT = ROOT / "paper" / "abstract.txt"


def flatten(tex: str) -> str:
    body = tex.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    for macro in ("texttt", "emph", "textbf", "textit"):
        body = re.sub(rf"\\{macro}{{([^}}]*)}}", r"\1", body)
    for escaped, plain in ((r"\%", "%"), (r"\_", "_"), (r"\&", "&"), (r"\$", "$")):
        body = body.replace(escaped, plain)
    body = re.sub(r"\$([^$]*)\$", r"\1", body)
    return re.sub(r"\s+", " ", body).strip()


def main() -> int:
    if not PAPER.is_file():
        print(f"no manuscript at {PAPER}", file=sys.stderr)
        return 1
    text = flatten(PAPER.read_text(encoding="utf-8"))
    # Anything left is markup the form would show as markup, so it is a refusal and not a
    # warning: an abstract is the one part of a submission that cannot be quietly wrong.
    leftover = sorted(set(re.findall(r"\\[a-zA-Z]+|[${}]", text)))
    if leftover:
        print(f"markup this script does not know how to flatten: {leftover}", file=sys.stderr)
        return 1
    OUT.write_text(textwrap.fill(text, width=88) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(text)} characters, paste this into the form")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The manuscript, checked as far as it can be without a TeX installation.

No LaTeX toolchain is assumed here, so these tests do not prove the paper compiles. They
catch the breakages that do not need a compiler to detect: numbers that no longer match
the runs behind them, an `\\input` pointing at nothing, a `\\ref` with no label, an
unbalanced environment, and a citation nobody can follow.

The last of those is the one worth having. A manuscript that quotes a figure the entries
no longer hold is the same failure this project has already published once.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_data import build as build_paper_data  # noqa: E402

PAPER = ROOT / "paper" / "anvil.tex"
DATA = ROOT / "paper" / "data"
BIB = ROOT / "paper" / "anvil.bib"


def _tex() -> str:
    return PAPER.read_text(encoding="utf-8")


def test_the_generated_data_matches_the_leaderboard_entries():
    for name, body in build_paper_data().items():
        path = DATA / name
        assert path.exists(), f"{name} is missing: ./scripts/paper_data.py"
        assert path.read_text(encoding="utf-8") == body, (
            f"{name} is stale: ./scripts/paper_data.py"
        )


def test_every_input_and_data_file_the_paper_names_exists():
    tex = _tex()
    for rel in re.findall(r"\\input\{([^}]+)\}", tex):
        assert (ROOT / "paper" / rel).exists(), rel
    for rel in re.findall(r"table\[[^\]]*\]\s*\{([^}]+)\}", tex):
        assert (ROOT / "paper" / rel).exists(), rel


def test_every_reference_has_a_label():
    tex = _tex()
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    for ref in re.findall(r"\\ref\{([^}]+)\}", tex):
        assert ref in labels, ref


def test_environments_are_balanced():
    tex = _tex()
    opened = Counter(re.findall(r"\\begin\{([a-zA-Z*]+)\}", tex))
    closed = Counter(re.findall(r"\\end\{([a-zA-Z*]+)\}", tex))
    assert opened == closed, {
        name: (opened[name], closed[name])
        for name in set(opened) | set(closed) if opened[name] != closed[name]
    }


def test_every_citation_resolves_to_the_bibliography():
    keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    for cite in re.findall(r"\\cite[tp]?\{([^}]+)\}", _tex()):
        for key in (k.strip() for k in cite.split(",")):
            assert key in keys, key


def test_the_bibliography_holds_only_what_can_be_followed():
    """The related-work section is unwritten on purpose. A plausible reference added to
    fill it would cost this paper the one thing it offers, so the sweep in
    `scripts/litsweep.py` is the only way entries get in."""
    bib = BIB.read_text(encoding="utf-8")
    assert re.findall(r"arXiv:(\d{4}\.\d{4,5})", bib) == ["2107.03374"]
    assert len(re.findall(r"^@\w+\{", bib, re.M)) == 1

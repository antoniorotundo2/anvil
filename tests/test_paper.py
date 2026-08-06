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

from leaderboard import LEVELS  # noqa: E402

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


def test_every_reference_can_be_followed():
    """The one claim this paper has is that its claims can be checked, and a reference
    without an identifier cannot be. Entries come from the sweep in `scripts/litsweep.py`,
    whose output is committed under `sweep/`, and each carries the DOI that sweep found."""
    bib = BIB.read_text(encoding="utf-8")
    entries = re.split(r"^@\w+\{", bib, flags=re.M)[1:]
    assert entries, "the bibliography is empty"
    for entry in entries:
        key = entry.split(",", 1)[0].strip()
        assert re.search(r"^\s*doi\s*=", entry, re.M), f"{key} has no DOI"


def test_the_sweep_behind_the_bibliography_is_committed():
    """Related work says the selection can be audited. That is only true while the records
    it was selected from are in the repository."""
    for name in ("results.csv", "results.md", "by_query.json"):
        assert (ROOT / "sweep" / name).is_file(), name


def test_a_model_the_generator_does_not_know_still_reaches_the_tables(tmp_path):
    """The order was a hardcoded list and also a filter, so a model added to the
    leaderboard would have been dropped from every table and figure without a word. It is
    a preference now: unknown models are appended by ascending strict score."""
    import paper_data as pd  # noqa: PLC0415

    def entry(model, strict):
        return {
            "model": model, "tasks_file": "tasks/t1_slurm.jsonl",
            "scores": {lv: {"mean": strict, "half_range": 0.0} for lv in LEVELS},
        }

    models = {
        "Qwen/Qwen2.5-Coder-7B-Instruct": entry("Qwen/Qwen2.5-Coder-7B-Instruct", 0.667),
        "newcomer/Model-9B": entry("newcomer/Model-9B", 0.5),
    }
    assert pd.order(models)[-1] == "newcomer/Model-9B"
    assert "Model-9B" in pd._table(models)
    assert "Model-9B" in pd._dat(models)

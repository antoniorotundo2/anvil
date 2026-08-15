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


def test_the_patterns_these_checks_read_the_paper_with_still_match():
    """The two checks around this one, and the citation check below, are loops over
    `re.findall` on the sources. A renamed macro or a reformat that puts a brace on the
    next line makes the pattern match nothing, and a loop over nothing passes: the checks
    would go quiet exactly when the paper had changed enough to need them.

    Floors of one, not the current counts, so writing the paper is not an edit here. What
    they catch is a pattern that has stopped seeing the document at all."""
    tex = _tex()
    for pattern, what in (
        (r"\\input\{([^}]+)\}", "included sources"),
        (r"table\[[^\]]*\]\s*\{([^}]+)\}", "data files behind the tables"),
        (r"\\ref\{([^}]+)\}", "cross-references"),
        (r"\\label\{([^}]+)\}", "labels"),
    ):
        assert re.findall(pattern, tex), f"the pattern for {what} matches nothing"


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

    # Keyed by (model, executor, quantization): a task file can be graded and loaded several
    # ways, and each combination is a measurement of its own.
    models = {
        ("Qwen/Qwen2.5-Coder-7B-Instruct", "bash", "4-bit"):
            entry("Qwen/Qwen2.5-Coder-7B-Instruct", 0.667),
        ("newcomer/Model-9B", "bash", "4-bit"): entry("newcomer/Model-9B", 0.5),
    }
    assert pd.order(models)[-1] == ("newcomer/Model-9B", "bash", "4-bit")
    assert "Model-9B" in pd._table(models)
    assert "Model-9B" in pd._dat(models)


def test_a_task_file_graded_twice_keeps_both_arms():
    """`tasks/t1_exec.jsonl` carries a `bash` and an `sbatch` entry per model, and the point
    of the set is the distance between them. Keyed by model alone the second overwrote the
    first and the table published one arm without saying which, which is the shape of every
    silent-drop defect this project has had to correct."""
    import paper_data

    cells = paper_data._by_tasks()["tasks/t1_exec.jsonl"]
    assert {key[1] for key in cells} == {"bash", "sbatch"}
    assert len(cells) == 10

    table = (DATA / "t1_exec_table.tex").read_text(encoding="utf-8")
    assert "(bash)" in table and "(sbatch)" in table
    # The single-arm tables must read exactly as they did, with no executor in the label.
    assert "(bash)" not in (DATA / "t1_slurm_table.tex").read_text(encoding="utf-8")


def test_every_deferred_item_is_reachable_from_the_regeneration_list():
    """Four findings wait on one rebuild of `tasks/t2_repair.jsonl`, which costs a full
    re-generation with every model. They are argued in four different sections, so the list
    that the pass has to follow is the thing that must not drift: a section that says it is
    waiting has to be reachable from it, or the pass happens and one of them is missed.
    """
    doc = (ROOT / "docs" / "OBSERVED_FAILURES.md").read_text(encoding="utf-8")
    listing = doc.split("## What to change when the T2 set is regenerated")[1]
    listing = listing.split("## Next measurements needed")[0]
    for anchor in ("#f9-an-option-this-scheduler-does-not-have",
                   "#f10-a-unit-confusion-the-scheduler-accepts",
                   "#what-f3-actually-measures"):
        assert anchor in listing, anchor
    # And the sections that defer must point back, so neither side can be edited alone.
    assert doc.count("#what-to-change-when-the-t2-set-is-regenerated") >= 2


def test_the_build_epoch_is_the_date_the_title_page_carries():
    """`PAPER_EPOCH` in the Makefile and `\\date` in the manuscript are two literals saying
    the same thing, and nothing but this makes them agree.

    The epoch used to be the commit time of the sources, which reproduced only until the
    next commit: it advances the moment the `.tex` is committed, so the PDF committed
    alongside it never rebuilt identically again. Pinning it closed that, at the cost of a
    second place to change the date.
    """
    import datetime as dt
    import re

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    epoch = re.search(r"^PAPER_EPOCH\s*=\s*(\d+)", makefile, re.M)
    assert epoch, "the Makefile no longer pins a build epoch"

    tex = (ROOT / "paper" / "anvil.tex").read_text(encoding="utf-8")
    stamped = re.search(r"\\date\{([^}]+)\}", tex)
    assert stamped, "the manuscript no longer carries a fixed date"
    assert "today" not in stamped.group(1), "the date floats again, so the epoch cannot pin it"

    named = dt.datetime.strptime(stamped.group(1), "%B %d, %Y").replace(tzinfo=dt.timezone.utc)
    assert int(epoch.group(1)) == int(named.timestamp()), (
        f"the Makefile builds at {epoch.group(1)} and the title page says "
        f"{stamped.group(1)} ({int(named.timestamp())})"
    )

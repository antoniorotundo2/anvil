"""The tables typed into the documents, checked against the entries they claim to report.

`docs/RESULTS.md` and `docs/OBSERVED_FAILURES.md` carry the multi-seed figures as markdown
written by hand, while the same numbers live in `leaderboard/entries/` and reach the paper
through `scripts/paper_data.py`. Two of those three are generated and one is not, so a
regrade updates the generated pair and leaves the prose behind. That happened twice in one
day: the walltime floor and then the memory bound each moved cells that had to be retyped on
four surfaces, and the second time a stale figure survived a commit.

The tables are not generated because they are read as part of an argument and carry a
column, `functional` under both executors side by side, that no single entry holds. Checking
them is the cheaper half of the same guarantee.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ROOT / "leaderboard" / "entries"

# The column order the two documents use, mapped to (level, executor). `strict` appears once
# in RESULTS.md and twice in OBSERVED_FAILURES.md, so it is matched by position rather than
# by name.
COLUMNS = [
    ("syntax", "bash"),
    ("submittability", "bash"),
    ("functional", "bash"),
    ("functional", "sbatch"),
    ("resource_fit", "bash"),
    ("strict_all_levels", "bash"),
    ("strict_all_levels", "sbatch"),
]

TABLES = {
    "docs/RESULTS.md": [("tasks/t1_slurm.jsonl", 6), ("tasks/t2_repair.jsonl", 6)],
    "docs/OBSERVED_FAILURES.md": [("tasks/t1_slurm.jsonl", 7), ("tasks/t2_repair.jsonl", 7)],
}


def _published() -> dict[tuple[str, str, str], tuple[float, float]]:
    """(model suffix, tasks file, level+executor) -> (mean, half range) from the entries."""
    out = {}
    for path in ENTRIES.glob("*__4-bit.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        short = entry["model"].split("/", 1)[-1]
        for level, scores in entry["scores"].items():
            key = (short, entry["tasks_file"], f"{level}/{entry['executor']}")
            out[key] = (scores["mean"], scores["half_range"])
    return out


def _rows(document: str) -> list[tuple[str, str, list[str]]]:
    """Every data row of the multi-seed tables, tagged with the task file it belongs to."""
    lines = (ROOT / document).read_text(encoding="utf-8").splitlines()
    found, expected = [], list(TABLES[document])
    for i, line in enumerate(lines):
        if not line.startswith("| model | syntax |"):
            continue
        tasks_file, width = expected.pop(0)
        for row in lines[i + 2:]:
            if not row.startswith("|"):
                break
            cells = [c.strip() for c in row.strip("|").split("|")]
            assert len(cells) == width + 1, f"{document}: {len(cells) - 1} columns, {width}"
            found.append((tasks_file, cells[0], cells[1:]))
    assert not expected, f"{document}: a multi-seed table is missing"
    return found


def test_the_typed_tables_match_the_entries_they_report():
    published = _published()
    checked = 0
    for document in TABLES:
        for tasks_file, model, cells in _rows(document):
            for value, (level, executor) in zip(cells, COLUMNS, strict=False):
                mean, half = published[(model, tasks_file, f"{level}/{executor}")]
                assert re.fullmatch(r"[\d.]+±[\d.]+", value), f"{document}: {model} {value!r}"
                # Compared as the three decimals both surfaces print, not within a
                # tolerance: an entry keeps six, and 0.0375 stored against 0.037 displayed
                # is agreement rather than drift. Rounding twice is what put 0.004 on this
                # page where the ablation printed 0.005, so the comparison is the format.
                assert value == f"{mean:.3f}±{half:.3f}", (
                    f"{document}: {model} {level}/{executor} says {value}, "
                    f"the entry holds {mean:.3f}±{half:.3f}"
                )
                checked += 1
    # Two documents, two task files, five models, six or seven columns each.
    assert checked >= 120, checked

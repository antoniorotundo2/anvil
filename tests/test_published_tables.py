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


def test_the_verifier_results_md_names_is_the_one_that_graded_the_entries():
    """The page opens by saying which verifier produced everything on it, and that sentence
    is prose: nothing regenerates it. It went stale twice, and the second time it named a
    digest two changes old while also asserting that every leaderboard row was marked
    *stale rules*, which the regrade had just stopped being true. A reader has no way to
    check that claim, so the suite does.
    """
    import re

    from anvil.provenance import verifier_sha

    body = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    claimed = re.search(r"graded by verifier `([0-9a-f]{12})`", body)
    assert claimed, "docs/RESULTS.md no longer says which verifier graded it"

    entries = [json.loads(p.read_text(encoding="utf-8")) for p in ENTRIES.glob("*.json")]
    assert entries
    digests = {e["verifier_sha"] for e in entries}
    assert claimed.group(1) in digests, (
        f"the page says {claimed.group(1)} graded it; the entries carry {sorted(digests)}"
    )

    # Every digest the entries carry has to appear on the page. A regrade lands one arm at a
    # time, since each is a separate run on the machine that holds the generations, so a
    # mixed state is legitimate: the fp16 row sat one verifier behind the other forty for
    # exactly that reason. What is not legitimate is a page that names one of them and lets
    # the reader take it for all.
    unnamed = sorted(d for d in digests if d not in body)
    assert not unnamed, f"entries graded by {unnamed}, which docs/RESULTS.md does not name"

    # And a row graded by anything other than this checkout is not comparable, which the
    # page has to say rather than leave to whoever notices the digests differ.
    if digests != {verifier_sha()}:
        assert verifier_sha() in body, (
            f"this checkout grades with {verifier_sha()}, which docs/RESULTS.md does not "
            f"mention while publishing rows graded by {sorted(digests)}"
        )
        assert "stale rules" in body, "the page does not say those rows are not comparable"


def test_the_figures_the_prose_repeats_are_still_the_facts():
    """Eight T1 tasks, five levels, five models, three seeds. Those four numbers are written
    out in prose across seven documents, thirty-odd times, none of them generated. Changing
    any of the underlying facts leaves every one of those sentences quietly wrong, and the
    only reason this file can check them is that they are facts about the repository rather
    than measurements.

    The check is deliberately failable: adding a sixth model should stop the suite, because
    the documents then need a pass. `docs/DATASET.md` claiming 220 repair tasks where the
    file holds 44 is what this class of drift looks like once it has gone unnoticed.
    """
    from anvil.schema import Level, Task

    tasks = Task.load_jsonl(ROOT / "tasks" / "t1_slurm.jsonl")
    entries = [json.loads(p.read_text(encoding="utf-8")) for p in ENTRIES.glob("*.json")]
    models = {e["model"] for e in entries}
    seeds = {tuple(e["seeds"]) for e in entries}

    assert len(tasks) == 8, f"{len(tasks)} T1 tasks: the documents say eight"
    assert len(list(Level)) == 5, f"{len(list(Level))} levels: the documents say five"
    assert len(models) == 5, f"{len(models)} models on the leaderboard: the documents say five"
    assert seeds == {(0, 1, 2)}, f"seed sets {seeds}: the documents say three seeds, 0 to 2"


def test_the_executor_figure_reads_the_same_on_every_surface_that_states_it():
    """One measurement, five surfaces: the README, `docs/RESULTS.md`,
    `docs/OBSERVED_FAILURES.md`, and both the abstract and the body of the manuscript. All
    typed by hand, three of them in bold, and nothing tied them together.

    Correcting it took three passes because of that. The body went from 297 to 288, the
    abstract kept 297 for an hour, and the other three kept it until a grep found them. A
    figure quoted in five places needs one check, not five careful readers.
    """
    import re

    surfaces = {
        "README.md": r"\*\*(\d+) artifacts of 900 change",
        "docs/RESULTS.md": r"reads \*\*(\d+) artifacts of 900",
        "docs/OBSERVED_FAILURES.md": r"\*\*(\d+) artifacts of 900 change their strict verdict",
        "paper/anvil.tex": r"the same comparison reads \$(\d+)\$ of \$900\$",
    }
    said = {}
    for name, pattern in surfaces.items():
        body = (ROOT / name).read_text(encoding="utf-8")
        found = re.search(pattern, body)
        assert found, f"{name} no longer states the figure where this test looks"
        said[name] = found.group(1)

    # The manuscript's body states it a second time, in bold, beside the OOM breakdown.
    tex = (ROOT / "paper" / "anvil.tex").read_text(encoding="utf-8")
    in_body = re.search(r"\\textbf\{\$(\d+)\$ artifacts of \$900\$\}", tex)
    assert in_body, "the manuscript body no longer states the figure in bold"
    said["paper/anvil.tex (body)"] = in_body.group(1)

    assert len(set(said.values())) == 1, said


def test_the_number_of_sizes_reads_the_same_in_the_page_and_the_paper():
    """`docs/RESULTS.md` said four and the manuscript said five, and the leaderboard has held
    five models at five distinct sizes since the fifth was added. The count of models is pinned
    against the entries above; this pins the phrasing that escaped it, because the sentence
    says sizes rather than models and no grep for one finds the other."""
    import re

    said = {}
    for name in ("docs/RESULTS.md", "paper/anvil.tex"):
        body = (ROOT / name).read_text(encoding="utf-8")
        found = re.search(r"model families at (\w+) sizes", body)
        assert found, f"{name} no longer states how many sizes were measured"
        said[name] = found.group(1)
    assert len(set(said.values())) == 1, said

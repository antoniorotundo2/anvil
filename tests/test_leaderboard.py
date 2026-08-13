"""The published ranking must be the one the entries say.

The page is generated, so the only way it can lie is by being stale. That is exactly how a
wrong table survived here for weeks, so it is a test rather than a habit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_manifest import build as build_manifest  # noqa: E402

from leaderboard import ENTRIES, LEVELS, PAGE, _load_entries, render  # noqa: E402


def test_the_page_matches_its_entries():
    assert PAGE.read_text(encoding="utf-8") == render(), (
        "docs/LEADERBOARD.md is stale: ./scripts/leaderboard.py render"
    )


def test_the_entries_are_found_at_all():
    """Six tests in this file are written `for entry in _load_entries()`, and every one of
    them reports success when the directory yields nothing. `ENTRIES` is monkeypatched to a
    temporary path by two of the tests below, so a restore that fails to run leaves the rest
    of the session looking at an empty directory and passing on it.

    A floor rather than the current count: the entries are a record of measurements and grow,
    but pruning a few stale rows is legitimate and should not fail here."""
    found = _load_entries()
    assert len(found) >= 10, f"{len(found)} entries under {ENTRIES}"


def test_every_entry_was_measured_against_the_current_task_files():
    """An entry graded against a different version of a task file is not comparable with
    the rest of its column. The page marks it stale; this says whether any are."""
    digests = {f["path"]: f["sha256"][:12] for f in build_manifest()["files"]}
    stale = [
        e["model"] for e in _load_entries()
        if e["tasks_sha"] != digests.get(e["tasks_file"])
    ]
    assert not stale, f"entries measured against an older task set: {stale}"


def test_entries_declare_the_conditions_that_make_them_comparable():
    required = {"model", "tasks_file", "tasks_sha", "seeds", "n_per_task",
                "executor", "base_image", "quantization", "scores"}
    for path in ENTRIES.glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        assert required <= set(entry), (path.name, sorted(required - set(entry)))
        assert entry["seeds"], path.name


def test_the_ranking_column_is_present_for_every_entry():
    """A row with no `strict_all_levels` would sort as zero and read as a bad model."""
    for entry in _load_entries():
        assert "strict_all_levels" in entry["scores"], entry["model"]


def test_import_keeps_more_precision_than_it_displays(tmp_path):
    """A half-range of 0.004583 stored as 0.0045 renders as 0.004, while the ablation
    prints 0.005 from the same run. Storing four decimals and displaying three made the
    same measurement show two values."""
    import leaderboard as lb  # noqa: PLC0415

    cells = []
    for pass1 in (0.6595833, 0.6641666, 0.6687499):
        path = tmp_path / f"cell_{pass1}.json"
        path.write_text(json.dumps({
            "model": "m/x",
            "tasks_file": "tasks/t1_slurm.jsonl",
            "environment": {"functional_executor": "bash", "base_image": "ubuntu:24.04"},
            "summary": {"strict_all_levels": {"pass@1": pass1}},
        }), encoding="utf-8")
        cells.append(str(path))

    from argparse import Namespace  # noqa: PLC0415

    entries_dir, lb.ENTRIES = lb.ENTRIES, tmp_path / "entries"
    try:
        lb.cmd_import(Namespace(results=cells, seeds=[0, 1, 2], n=5,
                                quantization="4-bit", source="test"))
        written = json.loads(next((tmp_path / "entries").glob("*.json")).read_text())
    finally:
        lb.ENTRIES = entries_dir

    half = written["scores"]["strict_all_levels"]["half_range"]
    assert f"{half:.3f}" == "0.005", half


def test_an_entry_is_keyed_by_its_executor_too(tmp_path, monkeypatch):
    """Two gradings of one model on one task file are two rows, not one overwriting the
    other. On `tasks/t1_exec.jsonl` the difference between the arms is the whole result, and
    a key without the executor would publish whichever was imported last.
    """
    import leaderboard as lb

    monkeypatch.setattr(lb, "ENTRIES", tmp_path)
    written = []
    for executor in ("bash", "sbatch"):
        cell = {
            "model": "vendor/m", "tasks_file": "tasks/t1_exec.jsonl",
            "environment": {"functional_executor": executor, "base_image": "ubuntu:24.04"},
            "summary": {lv: {"pass@1": 1.0} for lv in LEVELS},
        }
        path = tmp_path / "cell.json"
        path.write_text(json.dumps(cell), encoding="utf-8")
        args = argparse.Namespace(results=[str(path)], seeds=[0], n=5,
                                  quantization="4-bit", source="")
        assert lb.cmd_import(args) == 0
        written.append(sorted(q.name for q in tmp_path.glob("vendor_m__*.json")))

    assert len(written[-1]) == 2, written[-1]


def test_a_cell_measured_under_other_conditions_is_refused(tmp_path, monkeypatch):
    """Base image, samples and seeds are recorded but not in the key, so a cell measured
    under any of them would land on an existing file and replace a published number with one
    taken under other conditions. The import refuses and names the field rather than widening
    the key every time a condition is added. Quantization was the first to trip this and was
    answered by widening the key instead, once there was a measurement showing the two arms
    are a result: the rule is what turned that into a decision."""
    import leaderboard as lb

    monkeypatch.setattr(lb, "ENTRIES", tmp_path)
    cell = {
        "model": "vendor/m", "tasks_file": "tasks/t1_slurm.jsonl",
        "environment": {"functional_executor": "bash", "base_image": "ubuntu:24.04"},
        "summary": {lv: {"pass@1": 1.0} for lv in LEVELS},
    }
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(cell), encoding="utf-8")

    def run(quantization):
        return lb.cmd_import(argparse.Namespace(
            results=[str(path)], seeds=[0], n=5, quantization=quantization, source=""))

    assert run("4-bit") == 0
    assert run("fp16") == 0, "quantization is in the key, so the two arms coexist"
    assert len(list(tmp_path.glob("vendor_m__*.json"))) == 2
    # What the refusal still covers is a condition that is not in the key.
    args = argparse.Namespace(results=[str(path)], seeds=[0], n=9,
                              quantization="4-bit", source="")
    assert lb.cmd_import(args) == 2, "n=9 must not overwrite the n=5 measurement"

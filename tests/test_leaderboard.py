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

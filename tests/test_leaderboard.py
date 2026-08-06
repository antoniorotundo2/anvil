"""The published ranking must be the one the entries say.

The page is generated, so the only way it can lie is by being stale. That is exactly how a
wrong table survived here for weeks, so it is a test rather than a habit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_manifest import build as build_manifest  # noqa: E402

from leaderboard import ENTRIES, PAGE, _load_entries, render  # noqa: E402


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

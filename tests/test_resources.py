"""Where the data files are found, with and without a checkout.

The defaults in the CLI name paths relative to the working directory, which is right
inside a clone and names nothing once the package is installed somewhere else. These tests
pin the order of preference, because getting it wrong is not a crash: it is a run that
silently grades against a different copy of the task set than the caller meant.
"""

from __future__ import annotations

from pathlib import Path

from anvil import resources
from anvil.resources import resolve


def test_an_existing_path_is_used_unchanged(tmp_path):
    """A checkout keeps behaving exactly as before, and an explicit --tasks pointing
    somewhere unusual is never second-guessed."""
    given = tmp_path / "tasks" / "t1_slurm.jsonl"
    given.parent.mkdir()
    given.write_text("{}\n", encoding="utf-8")
    assert resolve(given) == given
    assert resolve(str(given)) == given


def test_a_missing_path_falls_back_to_the_packaged_copy(tmp_path, monkeypatch):
    """The name is one this repository does not have, so the first branch cannot answer:
    the tests run from the checkout, where `tasks/t1_slurm.jsonl` exists and would win."""
    packaged = tmp_path / "data"
    packaged.mkdir()
    (packaged / "t1_installed_only.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(resources, "PACKAGED", packaged)

    assert resolve("tasks/t1_installed_only.jsonl") == packaged / "t1_installed_only.jsonl"


def test_a_path_nobody_can_satisfy_comes_back_as_it_was(tmp_path, monkeypatch):
    """The caller then raises FileNotFoundError naming what the user asked for, rather
    than an internal directory they never mentioned."""
    monkeypatch.setattr(resources, "PACKAGED", tmp_path / "empty")
    assert resolve("nowhere/t1_slurm.jsonl") == Path("nowhere/t1_slurm.jsonl")


def test_the_repository_task_files_resolve_to_the_repository(tmp_path):
    """Inside this checkout the fallback must never fire: every published number was
    measured against `tasks/`, and quietly reading a packaged copy instead would be a
    different task set with the same name."""
    root = Path(__file__).resolve().parents[1]
    for name in ("t1_slurm.jsonl", "t2_repair.jsonl", "retrieval_corpus.jsonl"):
        given = root / "tasks" / name
        assert resolve(given) == given

"""The verifier digest, and what it is allowed to be sensitive to.

The value itself is not asserted: pinning it here would mean editing this file every time
`check_resource_fit` gains a rule, which is the opposite of what the digest is for. What is
asserted is that it moves when the rules move, that it does not depend on the order a
directory happens to list files in, and that a report carries it beside `tasks_sha`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from anvil.provenance import VERDICT_MODULES, verifier_sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from leaderboard import _not_comparable  # noqa: E402


def test_the_digest_is_stable_across_calls():
    assert verifier_sha() == verifier_sha()
    assert len(verifier_sha()) == 12


def test_it_covers_the_modules_a_verdict_depends_on():
    """`verifier.py` decides the levels and `parse.py` reads the directives they decide on.
    A change to either can move a verdict without moving a task file, which is exactly what
    the walltime floor did."""
    assert set(VERDICT_MODULES) == {"verifier.py", "parse.py"}
    for name in VERDICT_MODULES:
        assert (ROOT / "anvil" / name).exists()


def test_it_moves_when_the_rules_move(monkeypatch, tmp_path):
    """Taken over raw bytes, so it moves for a comment too. That is the conservative
    direction: a changed digest asks why, it does not claim the numbers are wrong."""
    import anvil.provenance as prov

    before = verifier_sha()
    fake = tmp_path / "anvil"
    fake.mkdir()
    for name in VERDICT_MODULES:
        (fake / name).write_bytes((ROOT / "anvil" / name).read_bytes() + b"\n# moved\n")
    monkeypatch.setattr(prov, "_HERE", fake)
    assert prov.verifier_sha() != before


def test_a_report_carries_both_digests(tmp_path):
    """The pair is the point: a task set and the rules applied to it. Either moving alone
    makes two reports incomparable, and before this the second one left no trace."""
    from anvil.cli import main

    out = tmp_path / "r.json"
    rc = main(["verify", "--generations", str(ROOT / "results" / "generations.jsonl"),
               "--tasks", str(ROOT / "tasks" / "t1_slurm.jsonl"), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verifier_sha"] == verifier_sha()
    assert len(payload["tasks_sha"]) == 12


def test_an_entry_graded_by_other_rules_is_not_ranked():
    fresh = {"tasks_sha": "abc", "verifier_sha": verifier_sha()}
    assert _not_comparable(fresh, "abc") is None
    assert _not_comparable({**fresh, "verifier_sha": "0" * 12}, "abc") == "stale rules"
    assert _not_comparable({"tasks_sha": "abc"}, "abc") == "unstamped"
    # A row wrong on the task set is reported on that, not on both at once.
    assert _not_comparable({**fresh, "tasks_sha": "zzz"}, "abc") == "stale tasks"

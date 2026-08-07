"""The walltime floor diagnostic, against a report written here.

The number this produces decides whether the published scores get regraded, so the two
readings it could get wrong are pinned. `00:15` is fifteen seconds and not fifteen minutes,
which is the whole reason the gap went unnoticed, and a failing artifact must never be
counted: the question is how many *passes* are wrong, and a fail is already counted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from walltime_floor import ceilings, scan  # noqa: E402


def _report(tmp_path: Path) -> Path:
    def rec(task_id: str, script: str, ok: bool) -> dict:
        return {"task_id": task_id, "script": script, "all_passed": ok, "levels": []}

    results = [
        # 15 seconds where the prompt names 15 minutes, and the verifier passed it.
        rec("t1_array_job", "#SBATCH --time=00:15\n", True),
        # The same slip on a repair task, whose id carries a fault suffix.
        rec("t1_cpus_per_task__F4", "#SBATCH --time=00:30\n", True),
        # Correct, and must not be counted.
        rec("t1_array_job", "#SBATCH --time=00:15:00\n", True),
        # Below the ceiling but already failing for another reason: not a pass to recover.
        rec("t1_hello_serial", "#SBATCH --time=00:01\n", False),
    ]
    path = tmp_path / "model__seed0__bash.json"
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    return path


def test_every_t1_task_declares_a_walltime_ceiling():
    """F4 depends on it and so does this: the constraint is present on all eight, which is
    why the inducer never reaches its `--mem` and `--gpus` candidates."""
    assert len(ceilings()) == 8


def test_counts_only_passing_artifacts_below_the_named_walltime(tmp_path):
    buckets, detail, passed = scan([_report(tmp_path)])
    assert passed == 3
    assert sum(buckets.values()) == 2


def test_a_sub_minute_request_is_read_as_seconds(tmp_path):
    buckets, detail, _ = scan([_report(tmp_path)])
    assert buckets == {"under a minute": 2}
    assert detail == {("t1_array_job", "00:15", 15): 1, ("t1_cpus_per_task", "00:30", 30): 1}

"""The constraint audit, against a report written here.

The numbers this produces decide whether a check gets tightened, and tightening one
regrades everything published, so the readings it could get wrong are pinned. `00:15` is
fifteen seconds and not fifteen minutes, which is the whole reason the walltime gap went
unnoticed for five models. `--mem=8G` against a task naming 2GB has to land under *above*
and not under a bare mismatch, since the two directions are different arguments: an
over-request wastes an allocation and an under-request kills the job. And a failing
artifact must never be counted, because the question is how many *passes* the rules let
through, and a fail is already counted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from constraint_audit import KINDS, declared, scan  # noqa: E402


def _report(tmp_path: Path) -> Path:
    def rec(task_id: str, script: str, ok: bool) -> dict:
        return {"task_id": task_id, "script": script, "all_passed": ok, "levels": []}

    results = [
        # 15 seconds where the task names 15 minutes, which the check used to pass.
        rec("t1_array_job", "#SBATCH --time=00:15\n#SBATCH --mem=1G\n", True),
        # The same slip on a repair task, whose id carries a fault suffix.
        rec("t1_cpus_per_task__F4", "#SBATCH --time=00:30\n#SBATCH --mem=2G\n", True),
        # Four times the memory the task names, which the check still passes.
        rec("t1_array_job", "#SBATCH --time=00:15:00\n#SBATCH --mem=4G\n", True),
        # A GPU count above the declared minimum, the other unenforced direction.
        rec("t1_gpu_single", "#SBATCH --time=02:00:00\n#SBATCH --gres=gpu:4\n", True),
        # Wrong on both, and already failing: not a pass the rules let through.
        rec("t1_hello_serial", "#SBATCH --time=00:01\n#SBATCH --mem=64G\n", False),
    ]
    path = tmp_path / "model__seed0__bash.json"
    path.write_text(json.dumps({"model": "vendor/m", "results": results}), encoding="utf-8")
    return path


def test_every_task_declares_a_walltime_and_a_memory_floor():
    """F4 depends on the first and the audit on both: all eight tasks declare a walltime and
    a memory, which is why the F4 inducer never reaches its `--mem` and `--gpus` candidates.
    Only one task declares a GPU count, so that column is thin by construction."""
    d = declared()
    assert len(d) == 8
    assert sum("time_max_minutes" in c for c in d.values()) == 8
    assert sum("mem_min_mb" in c for c in d.values()) == 8
    assert sum("gpus_min" in c for c in d.values()) == 1


def test_only_passing_artifacts_are_counted(tmp_path):
    sides, _, passed, _ = scan([_report(tmp_path)])
    assert passed == 4
    # The failing record asks 64G against 1024MB and must appear nowhere.
    assert sides[("mem_min_mb", "above")] == 1


def test_the_two_directions_are_separated(tmp_path):
    """Which side a request falls on is the whole argument, so it is never pooled into one
    mismatch count."""
    sides, _, _, _ = scan([_report(tmp_path)])
    assert sides[("time_max_minutes", "below")] == 2
    assert sides[("time_max_minutes", "above")] == 0
    assert sides[("gpus_min", "above")] == 1


def test_a_sub_minute_request_is_read_as_seconds(tmp_path):
    _, written, _, _ = scan([_report(tmp_path)])
    assert written[("time_max_minutes", "below", "vendor/m", "t1_array_job", "00:15", 15)] == 1
    # The repair record keeps its fault suffix, so the two runs stay apart.
    assert written[("time_max_minutes", "below", "vendor/m", "t1_cpus_per_task__F4",
                    "00:30", 30)] == 1


def test_the_table_states_which_direction_each_check_refuses():
    """A populated bucket only matters against what the verifier does about it, so the two
    are printed together and the mapping is not allowed to drift silently."""
    assert KINDS["time_max_minutes"][3] == "both"
    assert KINDS["mem_min_mb"][3] == "both"
    assert KINDS["gpus_min"][3] == "below"


def test_artifacts_of_another_task_file_are_reported_not_silently_dropped(tmp_path):
    """Pointed at a run of a different task set the audit has nothing to say, and saying it
    with zeros in every bucket would read as a clean result. The count of what it could not
    audit is what tells the two apart."""
    path = tmp_path / "other__seed0__bash.json"
    path.write_text(json.dumps({"model": "vendor/m", "results": [
        {"task_id": "t1_memory_bound", "script": "#SBATCH --mem=64M\n", "all_passed": True},
    ]}), encoding="utf-8")
    sides, _, passed, unknown = scan([path])
    assert (passed, unknown) == (1, 1)
    assert not sides


def test_the_model_is_part_of_the_breakdown(tmp_path):
    """A loose bucket that belongs to one model moves a ranking when the check is tightened;
    one spread across all of them shaves every row. The two need different decisions."""
    _, written, _, _ = scan([_report(tmp_path)])
    assert {key[2] for key in written} == {"vendor/m"}

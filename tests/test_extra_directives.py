"""The audit of unrequested directives, and the table that decides what counts as one.

`CARRIERS` maps each constraint to the directive spellings that carry it, so a task
declaring `nodes` is not reported as having an unrequested `--nodes`. A spelling missing
from that table turns a demanded directive into an extra one on every artifact that writes
it, which does not fail: it inflates a count nobody can check by eye, in the direction that
manufactures a finding.

The canonical solutions are the control. They request exactly what each task declares, so
the only directives they may show as extra are ones they genuinely add, and today that is
`--job-name` on the five tasks that do not demand it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extra_directives import CARRIERS, expected, scan  # noqa: E402


def _report(tmp_path: Path, scripts: dict[str, str]) -> Path:
    path = tmp_path / "oracle__bash.json"
    path.write_text(json.dumps({
        "model": "oracle",
        "results": [{"task_id": tid, "all_passed": True, "script": s}
                    for tid, s in scripts.items()],
    }), encoding="utf-8")
    return path


def _references() -> dict[str, str]:
    return {r["id"]: r["script"] for r in
            (json.loads(line) for line in
             (ROOT / "tasks" / "t1_reference.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip())}


def test_every_constraint_the_tasks_declare_has_a_carrier():
    """Read from the task file rather than listed here, so a new constraint key cannot be
    added without the table noticing."""
    declared = set()
    for line in (ROOT / "tasks" / "t1_slurm.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            declared.update(json.loads(line).get("constraints") or {})
    assert declared <= set(CARRIERS), sorted(declared - set(CARRIERS))


def test_the_canonical_solutions_add_only_a_job_name(tmp_path):
    """The control on the table above. If `--mem` fell out of `CARRIERS`, every artifact
    that requests memory would be filed as carrying an unrequested directive, and the
    resulting count would read as a discovery."""
    extra, _, passed, untracked = scan([_report(tmp_path, _references())])
    assert passed == 8
    assert untracked == 0
    assert set(extra) == {"--job-name"}


def test_a_directive_no_task_mentions_is_reported(tmp_path):
    """`--exclusive` changes what is allocated and no level reads it, which is the case this
    audit exists to surface."""
    refs = _references()
    refs["t1_hello_serial"] = refs["t1_hello_serial"].replace(
        "#SBATCH --time", "#SBATCH --exclusive\n#SBATCH --time", 1)
    extra, detail, _, _ = scan([_report(tmp_path, refs)])
    assert extra["--exclusive"] == 1
    assert detail[("--exclusive", "t1_hello_serial", "oracle")] == 1


def test_a_rejected_artifact_is_not_counted(tmp_path):
    """The question is what the rules let through."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "model": "m",
        "results": [{"task_id": "t1_hello_serial", "all_passed": False,
                     "script": "#!/bin/bash\n#SBATCH --exclusive\n"}],
    }), encoding="utf-8")
    extra, _, passed, _ = scan([path])
    assert passed == 0
    assert not extra


def test_the_task_table_is_populated():
    """Every check above passes on an empty map, which is what a moved task file leaves."""
    known = expected()
    assert len(known) >= 8
    assert all(names for names in known.values())

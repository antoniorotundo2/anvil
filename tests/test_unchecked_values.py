"""The audit of values nothing verifies, and the transcription it rests on.

`unchecked_values.py` has to carry the values the prompts name as literals, because they
live in English inside `prompt` and that is the whole reason no level can compare them. A
literal copied out of prose is a transcription, and a transcription can be wrong or go
stale, in which case the audit reports artifacts as deviating when they are correct or,
worse, as correct when they deviate.

So the canonical solutions are the check on it: they are the artifacts that do exactly what
each prompt asks, so every tracked value must read `as asked` on all of them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from unchecked_values import BODY, DIRECTIVES, scan  # noqa: E402


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


def test_every_transcribed_value_matches_the_canonical_solution(tmp_path):
    verdicts, written, _, _ = scan([_report(tmp_path, _references())])
    wrong = sorted((task, label) for (task, label, verdict), n in verdicts.items()
                   if verdict != "as asked" and n)
    assert not wrong, f"the value transcribed for these does not match the reference: {wrong}"
    assert not written


def test_a_changed_value_is_reported_rather_than_counted_as_asked(tmp_path):
    """The direction that matters. An audit that files a deviation under `as asked` would
    report the opening as never taken, which is the answer that closes the question."""
    refs = _references()
    refs["t1_array_job"] = refs["t1_array_job"].replace("--array=1-5", "--array=1-1")
    refs["t1_output_paths"] = refs["t1_output_paths"].replace("logs/out_%j.txt", "out.txt")
    verdicts, written, _, _ = scan([_report(tmp_path, refs)])

    assert verdicts[("t1_array_job", "--array", "other")] == 1
    assert verdicts[("t1_output_paths", "--output", "other")] == 1
    assert {w[4] for w in written} == {"1-1", "out.txt"}


def test_an_artifact_that_failed_is_not_counted(tmp_path):
    """The question is what the rules let through, so a rejected artifact says nothing
    about the opening."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "model": "m",
        "results": [{"task_id": "t1_array_job", "all_passed": False,
                     "script": "#!/bin/bash\n#SBATCH --array=1-1\n"}],
    }), encoding="utf-8")
    verdicts, _, passed, _ = scan([path])
    assert passed == 0
    assert not verdicts


def test_the_audit_still_tracks_the_tasks_it_was_written_for():
    """Both checks above pass on an empty table, which is what a rename in the task file
    would leave behind."""
    assert set(DIRECTIVES) == {"t1_hello_serial", "t1_output_paths",
                               "t1_dependency_chain", "t1_array_job"}
    assert set(BODY) == {"t1_container_apptainer"}
    assert sum(len(v) for v in DIRECTIVES.values()) + sum(len(v) for v in BODY.values()) == 10

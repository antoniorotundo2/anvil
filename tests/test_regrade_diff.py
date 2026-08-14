"""The comparison between two gradings, and the two ways it can lie.

It can miss a flip, which is what the first version did by reading `all_passed`: a
`functional` change on a sample that already fails another level leaves the strict verdict
alone, so the comparison answered "nothing changed" about a run where a level had visibly
moved. And it can report agreement having read nothing, which is what every count-based
check does when its input is empty.

Both are covered here by a positive control as well as a negative one. A test that only
checks the identical case passes just as happily on a comparison that compares nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from regrade_diff import compare, main  # noqa: E402

LEVELS = ("syntax", "submittability", "functional", "resource_fit", "safety")


def _report(path: Path, functional: bool, resource_fit: bool = False) -> None:
    """One sample. `resource_fit` is False by default so the strict verdict stays False
    whatever `functional` does, which is the case the strict-only comparison could not see.
    """
    levels = [{"level": name, "passed": True, "skipped": False, "detail": "ok"}
              for name in LEVELS]
    for lr in levels:
        if lr["level"] == "functional":
            lr["passed"] = functional
            lr["detail"] = "exit 0" if functional else "expected output not found"
        if lr["level"] == "resource_fit":
            lr["passed"] = resource_fit
    path.write_text(json.dumps({
        "model": "m", "tasks_file": "tasks/t1_slurm.jsonl",
        "results": [{"task_id": "t1_output_paths", "all_passed": functional and resource_fit,
                     "script": "#!/bin/bash\n", "levels": levels}],
    }), encoding="utf-8")


def _pair(tmp_path: Path, first_functional: bool, second_functional: bool) -> tuple[Path, Path]:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _report(a / "cell__bash.json", first_functional)
    _report(b / "cell__bash.json", second_functional)
    return a, b


def test_a_level_that_flips_is_reported_even_when_strict_does_not_move(tmp_path):
    a, b = _pair(tmp_path, False, True)
    flips, lines, skipped = compare(a, b)
    assert flips["functional"] == 1
    assert not skipped
    assert "expected output not found" in lines[0]


def test_two_identical_gradings_report_nothing(tmp_path):
    a, b = _pair(tmp_path, True, True)
    flips, lines, skipped = compare(a, b)
    assert not flips and not lines and not skipped


def test_directories_sharing_no_report_are_not_agreement(tmp_path, capsys):
    """Zero flips over zero comparisons is the shape every empty check takes, and it is the
    answer that ends an investigation."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _report(a / "one__bash.json", True)
    _report(b / "another__bash.json", True)

    assert main([str(a), str(b)]) == 1
    assert "nothing was compared" in capsys.readouterr().err


def test_a_cell_whose_samples_do_not_line_up_is_refused_not_compared(tmp_path):
    """Comparing sample 3 of one grading with sample 3 of another only means something if
    they are the same artifact."""
    a, b = _pair(tmp_path, True, True)
    payload = json.loads((b / "cell__bash.json").read_text(encoding="utf-8"))
    payload["results"][0]["task_id"] = "t1_hello_serial"
    (b / "cell__bash.json").write_text(json.dumps(payload), encoding="utf-8")

    flips, _, skipped = compare(a, b)
    assert not flips
    assert skipped and "t1_output_paths against t1_hello_serial" in skipped[0]

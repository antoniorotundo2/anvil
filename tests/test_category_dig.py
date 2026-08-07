"""The category dig, on a report written here so the expected counts are known.

The point of the script is that it is read as evidence about a model: a miscount would be
argued from before it was noticed. Two properties carry that weight and are pinned here.
A level skipped for an environment reason must not appear among the mechanisms, since a
skip is the absence of evidence, and digits must collapse so that two artifacts refused
for the same reason with different numbers are counted once rather than twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from category_dig import collect  # noqa: E402


def _level(name: str, passed: bool, detail: str, skipped: bool = False) -> dict:
    return {"level": name, "passed": passed, "detail": detail, "skipped": skipped}


def _report(tmp_path: Path) -> Path:
    results = [
        # Two failures for the same reason, differing only in the number.
        {"task_id": "t1_hello_serial__F4", "all_passed": False,
         "script": "#SBATCH --time=01:30:00\n", "levels": [
            _level("syntax", True, "ok"),
            _level("submittability", False, "level skipped: sbatch not available", skipped=True),
            _level("resource_fit", False, "--time 90min exceeds maximum 30min"),
        ]},
        {"task_id": "t1_gpu_single__F4", "all_passed": False,
         "script": "#SBATCH --nodes=1\n", "levels": [
            _level("syntax", True, "ok"),
            _level("submittability", False, "level skipped: sbatch not available", skipped=True),
            _level("resource_fit", False, "--time 45min exceeds maximum 30min"),
        ]},
        # A pass, and a failure in another category, both outside the count.
        {"task_id": "t1_array_job__F4", "all_passed": True,
         "script": "#SBATCH --time=00:10:00\n", "levels": [
            _level("resource_fit", True, "ok"),
        ]},
        {"task_id": "t1_array_job__F5", "all_passed": False,
         "script": "#SBATCH --time=99:00:00\n", "levels": [
            _level("syntax", False, "unbalanced quote"),
        ]},
    ]
    path = tmp_path / "repair__model__seed0__bash.json"
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    return path


def test_counts_only_the_requested_category(tmp_path):
    c = collect("F4", [_report(tmp_path)])
    assert c["totals"]["total"] == 3
    assert c["totals"]["failed"] == 2


def test_environment_skips_are_not_mechanisms(tmp_path):
    c = collect("F4", [_report(tmp_path)])
    assert c["levels"] == {"resource_fit": 2}
    assert c["skipped"] == {"submittability": 2}
    assert all(level != "submittability" for level, _ in c["problems"])


def test_digits_collapse_so_one_reason_counts_once(tmp_path):
    c = collect("F4", [_report(tmp_path)])
    assert c["problems"] == {("resource_fit", "--time Nmin exceeds maximum Nmin"): 2}


def test_failures_are_attributed_to_the_base_task(tmp_path):
    c = collect("F4", [_report(tmp_path)])
    assert c["tasks"] == {"t1_hello_serial": 1, "t1_gpu_single": 1}


def test_lines_records_absence_and_keeps_the_passing_artifacts(tmp_path):
    """The value a model wrote is only readable against the one it got right, so `--lines`
    covers passes too, and an artifact with no matching line is counted rather than
    dropped: on this category the absence is the finding."""
    c = collect("F4", [_report(tmp_path)], r"^#SBATCH.*--time")
    assert c["lines"] == {
        ("t1_hello_serial", False, "#SBATCH --time=01:30:00"): 1,
        ("t1_gpu_single", False, "(absent)"): 1,
        ("t1_array_job", True, "#SBATCH --time=00:10:00"): 1,
    }


def test_all_reads_a_report_whose_ids_carry_no_fault_suffix(tmp_path):
    """A T1 report has ids like `t1_array_job`, and stripping at the last `__` would file
    them under `t1`. `all` keeps the id whole, so the same tool answers whether a habit
    seen in one repair category is also there in from-scratch generation."""
    c = collect("all", [_report(tmp_path)])
    assert c["totals"]["total"] == 4
    assert c["tasks"] == {"t1_hello_serial__F4": 1, "t1_gpu_single__F4": 1,
                          "t1_array_job__F5": 1}

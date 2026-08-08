"""The per-category table generator, on two models across two seeds.

What this has to get right is arithmetic nobody would check by eye once the table is in a
document: a mean over seeds and not over samples, and a denominator that is per model
rather than the pooled total. Both were done by hand before, one query at a time, and the
table went out with a missing column twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from category_table import collect, render  # noqa: E402

MODELS = ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen3.5-9B"]


def _write(run: Path, model: str, seed: int, f4: float) -> None:
    report = {
        "model": model,
        "by_category": {
            "F4": {"strict_all_levels": {"pass@1": f4}},
            "F6": {"strict_all_levels": {"pass@1": 1.0}},
        },
        # Three F4 records and one F6, per model per seed.
        "results": [{"task_id": "t1_hello_serial__F4"} for _ in range(3)]
        + [{"task_id": "t1_cpus_per_task__F6"}],
    }
    path = run / f"repair__{model.replace('/', '_')}__seed{seed}__bash.json"
    path.write_text(json.dumps(report), encoding="utf-8")


def _run(tmp_path: Path) -> Path:
    _write(tmp_path, MODELS[0], 0, 0.8)
    _write(tmp_path, MODELS[0], 1, 0.6)
    _write(tmp_path, MODELS[1], 0, 0.2)
    _write(tmp_path, MODELS[1], 1, 0.4)
    return tmp_path


def test_a_cell_is_the_mean_over_seeds(tmp_path):
    scores, _ = collect(_run(tmp_path))
    assert scores[MODELS[0]]["F4"] == [0.8, 0.6]
    table = render(*collect(_run(tmp_path)))
    # PREFERRED puts Qwen3.5 before the 7B, so the columns read in that order.
    assert "| 0.300 | 0.700 |" in table


def test_the_denominator_is_per_model_not_the_pooled_total(tmp_path):
    """Four reports hold 12 F4 records between them, and the table publishes the 6 that
    belong to one model."""
    _, counts = collect(_run(tmp_path))
    assert counts["F4"] == 12
    row = next(ln for ln in render(*collect(_run(tmp_path))).splitlines()
               if ln.startswith("| F4 "))
    assert row.endswith("| 6 |")


def test_only_the_categories_the_reports_hold_become_rows(tmp_path):
    """The row list used to be F1 to F7 written out here, which would have dropped F8 from
    any table of the execution set without a word. It comes from the reports now."""
    table = render(*collect(_run(tmp_path)))
    assert "F4 directive absent" in table
    assert "F6 payload/spec mismatch" in table
    assert "F1" not in table
    assert "F8" not in table


def test_a_category_one_model_lacks_reads_as_not_available(tmp_path):
    """Deriving the rows from the reports must not hide an asymmetry: a category measured
    for one model and not another is a gap in the comparison, not a row to drop."""
    run = _run(tmp_path)
    import json
    path = run / "repair__Qwen_Qwen3.5-9B__seed0__bash.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["by_category"]["F8"] = {"strict_all_levels": {"pass@1": 0.5}}
    report["results"].append({"task_id": "t1_memory_bound__F8"})
    path.write_text(json.dumps(report), encoding="utf-8")

    table = render(*collect(run))
    row = next(ln for ln in table.splitlines() if ln.startswith("| F8 "))
    assert "0.500" in row and "n/a" in row

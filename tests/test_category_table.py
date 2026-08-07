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


def test_categories_absent_from_the_reports_read_as_such(tmp_path):
    table = render(*collect(_run(tmp_path)))
    assert "| F1 omitted default | n/a | n/a |" in table

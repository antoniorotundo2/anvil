#!/usr/bin/env python3
"""The per-fault-category table, emitted from the reports instead of queried by hand.

The table in `docs/OBSERVED_FAILURES.md` was assembled one shell query per model per
category, five models and seven categories, and the last column took three separate passes
to land. That is a lot of transcription for numbers the reports already carry: `anvil
verify-repair` writes `by_category` into every T2 report, so the table is a projection of
files on disk and not a thing to be typed.

    ./scripts/category_table.py results/regrade_floor

Strict pass@1 per category, mean over the seeds present, `bash` arm. Models come out in the
order the paper ranks them and anything unrecognised is appended, the same rule
`paper_data.py` uses, so a new model never silently vanishes from a table again.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_data import PREFERRED, short  # noqa: E402

# Read from the reports rather than listed here. The T1 repair set exercises F1 to F7 and
# the execution set exercises F2, F4, F5, F7 and F8, so a hardcoded list drops a whole
# category from the table without a word, which is the defect this script was written to
# stop being possible by hand.
LABELS = {
    "F1": "F1 omitted default",
    "F2": "F2 directive after the first command",
    "F3": "F3 prose in a value",
    "F4": "F4 directive absent",
    "F5": "F5 no `#SBATCH` at all",
    "F6": "F6 payload/spec mismatch",
    "F7": "F7 malformed value",
    "F8": "F8 memory below what the payload uses",
}


def collect(run: Path) -> tuple[dict[str, dict[str, list[float]]], Counter]:
    """model -> category -> one strict pass@1 per seed, plus the sample count per category."""
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Counter = Counter()
    for path in sorted(glob.glob(str(run / "repair__*__bash.json"))):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        by_category = report.get("by_category")
        if not by_category:
            continue
        for category, summary in by_category.items():
            scores[report["model"]][category].append(summary["strict_all_levels"]["pass@1"])
        # Counted from the records rather than from n_tasks, so the denominator printed is
        # the one the numbers were actually computed over.
        for r in report["results"]:
            counts[r["task_id"].rsplit("__", 1)[1]] += 1
    return scores, counts


def categories(scores: dict) -> list[str]:
    """Sorted, so F10 would land after F9 rather than after F1."""
    seen = {c for per in scores.values() for c in per}
    return sorted(seen, key=lambda c: (len(c), c))


def render(scores: dict, counts: Counter) -> str:
    models = [m for m in PREFERRED if m in scores] + sorted(m for m in scores if m not in PREFERRED)
    header = "| category | " + " | ".join(short(m) for m in models) + " | n per model |"
    lines = [header, "|---" * (len(models) + 2) + "|"]
    for category in categories(scores):
        cells = []
        for model in models:
            values = scores[model].get(category, [])
            cells.append(f"{sum(values) / len(values):.3f}" if values else "n/a")
        # `counts` pools every report, so the per-model denominator the table publishes is
        # that total divided by the models in it.
        label = LABELS.get(category, category)
        lines.append(f"| {label} | " + " | ".join(cells)
                     + f" | {counts[category] // len(models)} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: category_table.py results/<run>", file=sys.stderr)
        return 2
    run = Path(argv[0])
    scores, counts = collect(run)
    if not scores:
        print(f"no T2 report with a by_category block under {run}", file=sys.stderr)
        return 1
    print(render(scores, counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Benchmark metrics.

pass@k uses the unbiased estimator of Chen et al. (Codex, 2021): with n samples
per task of which c are correct, the probability that at least one of k drawn
samples is correct is 1 - C(n-c, k)/C(n, k), computed in a numerically stable form.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .schema import Level, VerificationResult


def pass_at_k(n: int, c: int, k: int) -> float:
    """n = samples generated, c = correct samples, k = budget."""
    if k > n:
        raise ValueError(f"k={k} cannot exceed n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def aggregate(
    results: list[VerificationResult], k: int = 1
) -> dict[str, dict[str, float | int]]:
    """Group by task and compute pass@k for each verification level.

    A skipped level never counts as passed: this is what keeps the metrics honest
    on a machine without a working scheduler.
    """
    by_task: dict[str, list[VerificationResult]] = defaultdict(list)
    for r in results:
        by_task[r.task_id].append(r)

    out: dict[str, dict[str, float | int]] = {}
    for level in Level:
        scores: list[float] = []
        n_skipped = 0
        for _, rs in by_task.items():
            n = len(rs)
            c = sum(1 for r in rs if r.passed(level))
            n_skipped += sum(1 for r in rs if (lr := r.get(level)) and lr.skipped)
            scores.append(pass_at_k(n, c, min(k, n)))
        out[level.value] = {
            f"pass@{k}": round(mean(scores), 4) if scores else 0.0,
            "n_tasks": len(by_task),
            "n_skipped_samples": n_skipped,
        }

    # "strict": every non-skipped level passed
    strict: list[float] = []
    for _, rs in by_task.items():
        n = len(rs)
        c = sum(1 for r in rs if r.all_passed)
        strict.append(pass_at_k(n, c, min(k, n)))
    out["strict_all_levels"] = {
        f"pass@{k}": round(mean(strict), 4) if strict else 0.0,
        "n_tasks": len(by_task),
        "n_skipped_samples": 0,
    }
    return out


def aggregate_by_category(
    results: list[VerificationResult], categories: dict[str, str], k: int = 1
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Partition T2 results by fault category (F1-F7) and compute the same
    pass@k summary as `aggregate()` within each partition.

    `categories` maps a repair task id to its fault category (see
    `RepairTask.fault_category`). Reuses `aggregate()` per partition instead of
    duplicating the pass@k logic: a category is just a task subset.
    """
    by_cat: dict[str, list[VerificationResult]] = defaultdict(list)
    for r in results:
        by_cat[categories.get(r.task_id, "unknown")].append(r)
    return {cat: aggregate(rs, k=k) for cat, rs in sorted(by_cat.items())}

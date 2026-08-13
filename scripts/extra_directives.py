#!/usr/bin/env python3
"""Directives that passing artifacts carry and no task ever asked for.

The companion audit, `unchecked_values.py`, asked whether models wrote something *different*
where nothing looks, and over 11774 passing artifacts the answer was no. This asks the
opposite question, which that run raised: a 7B model wrote `--array=1-5%5` on 229 artifacts,
a concurrency cap nobody requested, harmless on inspection and invisible to every level. If
one unrequested component slipped through unnoticed, the set of them is worth counting.

    ./scripts/extra_directives.py 'results/*/*__bash.json'

A directive is *extra* when it is neither named in the task's `required_directives` nor the
carrier of one of its `constraints`. Extra is not wrong: `--job-name` on a task that does not
demand one is good practice, and `--partition` may be exactly what a site needs. The point is
that nothing in the verifier reads them, so a value that changes what the job is would be
promoted in silence, and the first step is knowing which ones appear at all.

Nothing here changes a verdict. It counts, so that a decision about widening `resource_fit`
rests on which directives real models actually reach for.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anvil.parse import parse_directives  # noqa: E402

TASKS = ROOT / "tasks" / "t1_slurm.jsonl"

# The directive spellings that carry each constraint, so a task declaring `nodes` is not
# reported as having an unrequested `--nodes`. Kept beside `check_resource_fit`, which reads
# the same aliases; a spelling missing here shows up as extra and overstates the count.
CARRIERS: dict[str, tuple[str, ...]] = {
    "nodes": ("--nodes", "-N"),
    "ntasks": ("--ntasks", "-n"),
    "cpus_per_task": ("--cpus-per-task", "-c"),
    "time_max_minutes": ("--time", "-t"),
    "mem_min_mb": ("--mem",),
    "gpus_min": ("--gpus", "-G", "--gres"),
    "array": ("--array", "-a"),
}


def expected() -> dict[str, set[str]]:
    """task id -> every directive that task has a reason to carry."""
    out: dict[str, set[str]] = {}
    for line in TASKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        names = set(task.get("required_directives") or [])
        for constraint in task.get("constraints") or {}:
            names.update(CARRIERS.get(constraint, ()))
        out[task["id"]] = names
    return out


def scan(paths: list[Path]) -> tuple[Counter, Counter, int, int]:
    """directive -> how often it appears unrequested, the same split by task and model,
    artifacts passed, and artifacts on tasks this file does not declare."""
    known = expected()
    extra: Counter = Counter()
    detail: Counter = Counter()
    passed = untracked = 0
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        model = report.get("model", "unknown")
        for r in report["results"]:
            if not r["all_passed"]:
                continue
            passed += 1
            base = r["task_id"].split("__", 1)[0]
            if base not in known:
                untracked += 1
                continue
            for name in parse_directives(r["script"]):
                if name not in known[base]:
                    extra[name] += 1
                    detail[(name, base, model)] += 1
    return extra, detail, passed, untracked


def report(extra: Counter, detail: Counter, passed: int, untracked: int) -> None:
    print(f"{passed} artifacts passed the verifier, {untracked} on tasks not declared here\n")
    if not extra:
        print("no directive appears that its task did not ask for")
        return
    print(f"{'directive':<22} {'artifacts':>9}")
    for name, n in extra.most_common():
        print(f"{name:<22} {n:>9}")
    print("\nwhere each one comes from:")
    for (name, task, model), n in sorted(detail.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {task:<24} {model} x{n}")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: extra_directives.py 'results/<run>/*__bash.json' ...", file=sys.stderr)
        print("       <run> is a directory name, not a literal. Try 'results/*/*__bash.json'",
              file=sys.stderr)
        return 2
    paths = [Path(p) for pattern in argv for p in sorted(glob.glob(pattern))]
    if not paths:
        print(f"no report matched: {' '.join(argv)}", file=sys.stderr)
        return 1
    report(*scan(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

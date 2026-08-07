#!/usr/bin/env python3
"""How many published passes asked for seconds where the prompt said minutes.

`check_resource_fit` compares `--time` against `time_max_minutes` in one direction only:
a request above the ceiling is a problem and a request below it is not. That was a
deliberate reading of `t1_hello_serial`, whose prompt says *at most* 10 minutes, and it is
wrong for the other seven, whose prompts name an exact walltime. Screening five models for
F10 turned up the consequence: `#SBATCH --time=00:15` against a task asking for 15 minutes
is fifteen seconds, and it passes.

    ./scripts/walltime_floor.py 'results/RUN/*__bash.json'

The script counts artifacts the verifier passed whose walltime is below the one the prompt
named, grouped by how far below. Nothing is regraded here and no verdict changes: the
point is to size the gap before deciding what to do about it, because a floor would move
every number this repository has published and that is not a change to make on an
impression.

Why `functional` does not catch it: every T1 payload finishes in under a second, so a job
granted fifteen seconds still prints what the task asked for. The level that would notice
an under-request is the one that runs the work, and this benchmark's work is too small to
run out of time. F8 is the same defect on memory and `functional` does catch that one,
because a payload that allocates more than it requested is killed for it.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anvil.parse import directive_value, parse_directives, parse_time_to_minutes  # noqa: E402

TASKS = ROOT / "tasks" / "t1_slurm.jsonl"


def ceilings() -> dict[str, int]:
    out = {}
    for line in TASKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if "time_max_minutes" in task["constraints"]:
            out[task["id"]] = task["constraints"]["time_max_minutes"]
    return out


def scan(paths: list[Path]) -> tuple[Counter, Counter, int]:
    """Passing artifacts only, bucketed by requested minutes against the declared ceiling."""
    limits = ceilings()
    buckets: Counter = Counter()
    detail: Counter = Counter()
    passed = 0
    for path in paths:
        for r in json.loads(path.read_text(encoding="utf-8"))["results"]:
            if not r["all_passed"]:
                continue
            passed += 1
            base = r["task_id"].split("__", 1)[0]
            limit = limits.get(base)
            if limit is None:
                continue
            raw = directive_value(parse_directives(r["script"]), "--time", "-t")
            mins = parse_time_to_minutes(raw) if raw else None
            if mins is None or mins >= limit:
                continue
            # Integer minutes, so a sub-minute request reads as 0 and needs its own bucket
            # rather than a ratio nobody can divide by.
            share = "under a minute" if mins == 0 else f"{mins}/{limit} of it"
            buckets[share] += 1
            detail[(base, raw, limit)] += 1
    return buckets, detail, passed


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: walltime_floor.py 'results/RUN/*__bash.json' ...", file=sys.stderr)
        return 2
    paths = [Path(p) for pat in argv for p in sorted(glob.glob(pat))]
    if not paths:
        print(f"no report matched: {' '.join(argv)}", file=sys.stderr)
        return 1

    buckets, detail, passed = scan(paths)
    affected = sum(buckets.values())
    print(f"{affected} of {passed} passing artifacts request less walltime than the prompt "
          f"named, over {len(paths)} report(s)")
    if not affected:
        return 0
    print("\nhow far below")
    for share, n in buckets.most_common():
        print(f"  {n:4d}  {share}")
    print("\nwhat was written")
    for (base, raw, limit), n in detail.most_common(15):
        print(f"  {n:4d}  {base:24s} --time={raw:12s} prompt names {limit} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

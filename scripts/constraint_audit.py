#!/usr/bin/env python3
"""Which resource requests the verifier lets through, on every constraint at once.

`check_resource_fit` compared `--time` against the walltime a task names from above only,
so `#SBATCH --time=00:15` against a task naming 15 minutes was fifteen seconds and passed.
123 of 2421 published passes were requests like that, and the whole thing surfaced by
accident while five models were being screened for something else. Two comparisons are
still one-sided, `--mem` and `--gpus`, and the point of this script is that nobody should
have to trip over them the same way.

    ./scripts/constraint_audit.py 'results/RUN/*__bash.json'

For every artifact the verifier passed, the requested value is compared with the one its
task declares and filed under below, exact, or above. The direction the check currently
enforces is printed beside each, so a bucket that is both populated and unenforced is the
thing to look at. `--time` is kept in the output as the control: it should now read zero
outside exact, and a non-zero bucket there means the floor regressed.

This subsumes `walltime_floor.py`, which answered the same question for one constraint and
produced the 123 above. Nothing here changes a verdict; it counts what the rules allow, so
that a decision about tightening them rests on a number rather than on an impression.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anvil.parse import (  # noqa: E402
    directive_value,
    parse_directives,
    parse_mem_to_mb,
    parse_time_to_minutes,
)

TASKS = ROOT / "tasks" / "t1_slurm.jsonl"

# constraint -> (label, directives to read, how to turn the raw value into a number, which
# direction check_resource_fit refuses). "both" means it demands equality.
KINDS = {
    "time_max_minutes": ("--time", ("--time", "-t"), parse_time_to_minutes, "both"),
    "mem_min_mb": ("--mem", ("--mem",), parse_mem_to_mb, "below"),
    "gpus_min": ("--gpus", ("--gpus", "-G", "--gres"), None, "below"),
}


def _gpus(raw: str) -> float | None:
    """`--gres=gpu:2` and `--gpus=2` are the same request, and only the count matters."""
    m = re.search(r"(\d+)\s*$", raw)
    return float(m.group(1)) if m else None


def declared() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in TASKS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = json.loads(line)
            out[task["id"]] = {k: v for k, v in task["constraints"].items() if k in KINDS}
    return out


def scan(paths: list[Path]) -> tuple[Counter, Counter, int, int]:
    """(constraint, side) counts over passing artifacts, what was written on each side, how
    many artifacts passed, and how many of those belong to a task this file does not
    declare."""
    limits = declared()
    sides: Counter = Counter()
    written: Counter = Counter()
    passed = unknown = 0
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for r in report["results"]:
            if not r["all_passed"]:
                continue
            passed += 1
            base = r["task_id"].split("__", 1)[0]
            model = report.get("model", "unknown")
            if base not in limits:
                # A report from another task file. Counted rather than skipped: without
                # this, pointing the audit at the wrong run prints zeros everywhere and
                # reads as a clean result instead of an empty one.
                unknown += 1
                continue
            directives = parse_directives(r["script"])
            for constraint, value in limits.get(base, {}).items():
                _, aliases, parse, _ = KINDS[constraint]
                raw = directive_value(directives, *aliases)
                if raw is None:
                    continue
                got = parse(raw) if parse else _gpus(raw)
                if got is None:
                    continue
                side = "exact" if got == value else ("above" if got > value else "below")
                sides[(constraint, side)] += 1
                if side != "exact":
                    # The model belongs in the key: whether a loose bucket is one model's
                    # habit or everyone's decides whether tightening the check moves a
                    # ranking or shaves every row equally.
                    written[(constraint, side, model, base, raw, value)] += 1
    return sides, written, passed, unknown


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: constraint_audit.py 'results/RUN/*__bash.json' ...", file=sys.stderr)
        return 2
    paths = [Path(p) for pat in argv for p in sorted(glob.glob(pat))]
    if not paths:
        print(f"no report matched: {' '.join(argv)}", file=sys.stderr)
        return 1

    sides, written, passed, unknown = scan(paths)
    print(f"{passed} passing artifacts over {len(paths)} report(s)")
    if unknown:
        print(f"{unknown} of them belong to a task {TASKS.name} does not declare and were "
              f"not audited")
    print()
    print(f"{'constraint':12s} {'below':>7s} {'exact':>7s} {'above':>7s}   refused by the check")
    for constraint, (label, _, _, refuses) in KINDS.items():
        counts = [sides[(constraint, side)] for side in ("below", "exact", "above")]
        note = {"both": "either side", "below": "below only, above is allowed"}[refuses]
        print(f"{label:12s} {counts[0]:7d} {counts[1]:7d} {counts[2]:7d}   {note}")

    if not written:
        print("\nnothing outside exact")
        return 0
    print("\nwhat was written where the value is not the declared one")
    for (constraint, side, model, base, raw, value), n in written.most_common(20):
        label = KINDS[constraint][0]
        print(f"  {n:4d}  {side:5s}  {model.split('/')[-1]:28s} {base:24s} "
              f"{label}={raw:10s} task declares {value:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Why one fault category fails, not just how often.

The per-category table in `docs/OBSERVED_FAILURES.md` says Qwen3.5-9B repairs F4 at 0.242
while four other models sit between 0.742 and 0.750. A number that far off the group is
either a mechanism or a mistake, and the table cannot tell which: it counts verdicts and
throws the reason away. This reads the reason back out of the reports.

    ./scripts/category_dig.py F4 'results/RUN/repair__*Qwen3.5*__bash.json'
    ./scripts/category_dig.py F4 --lines '^#SBATCH.*--time' 'results/RUN/repair__*__bash.json'

The output is three groupings of the failing artifacts: which levels refused them, which
problem strings the verifier emitted with digits collapsed so near-identical messages
count together, and which base task they came from. A category that fails for one reason
everywhere is a capability claim; one that fails for six unrelated reasons on one task is
a task defect, and the two look the same from the table.

With `--lines`, every artifact in the category also has its matching script lines counted
per base task and split by verdict, absence included. The problem strings say a directive
was missing or out of range; this says what stood there instead, which is the difference
between a model that ignores a constraint and one that reads it and answers it wrongly.

Levels skipped for an environment reason are separated out rather than counted as
failures. `strict_all_levels` is right to refuse them, but a skip is the absence of
evidence and has no place among the mechanisms.
"""

from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path


def collect(
    category: str, paths: list[Path], pattern: str | None = None
) -> dict[str, collections.Counter]:
    out = {k: collections.Counter() for k in ("levels", "skipped", "problems", "tasks", "lines")}
    total = failed = 0
    matcher = re.compile(pattern) if pattern else None
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for r in report["results"]:
            if not r["task_id"].endswith("__" + category):
                continue
            total += 1
            base = r["task_id"].rsplit("__", 1)[0]
            if matcher is not None:
                hits = [ln.strip() for ln in r["script"].splitlines() if matcher.search(ln)]
                # An artifact with no matching line is the interesting case, not a gap in
                # the tally: absence is recorded rather than dropped.
                for line in hits or ["(absent)"]:
                    out["lines"][(base, r["all_passed"], line[:56])] += 1
            if r["all_passed"]:
                continue
            failed += 1
            bad = [lv for lv in r["levels"] if not lv["passed"] and not lv["skipped"]]
            out["levels"][" + ".join(lv["level"] for lv in bad) or "skips only"] += 1
            for lv in r["levels"]:
                if lv["skipped"]:
                    out["skipped"][lv["level"]] += 1
            out["tasks"][base] += 1
            for lv in bad:
                for problem in lv["detail"].split("; "):
                    out["problems"][(lv["level"], re.sub(r"\d+", "N", problem)[:64])] += 1
    out["totals"] = collections.Counter(total=total, failed=failed)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: category_dig.py F4 [--lines REGEX] 'results/RUN/repair__*.json' ...",
              file=sys.stderr)
        return 2
    pattern = None
    if "--lines" in argv:
        i = argv.index("--lines")
        pattern = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    category, patterns = argv[0], argv[1:]
    paths = [Path(p) for pat in patterns for p in sorted(glob.glob(pat))]
    if not paths:
        print(f"no report matched: {' '.join(patterns)}", file=sys.stderr)
        return 1

    c = collect(category, paths, pattern)
    t = c["totals"]
    print(f"{category}: {t['total'] - t['failed']}/{t['total']} pass, {t['failed']} fail, "
          f"over {len(paths)} report(s)")
    print("\nfailing level combination")
    for combo, n in c["levels"].most_common():
        print(f"  {n:4d}  {combo}")
    if c["skipped"]:
        print("\nskipped levels")
        for level, n in c["skipped"].most_common():
            print(f"  {n:4d}  {level}")
    print("\nproblem, digits collapsed to N")
    for (level, problem), n in c["problems"].most_common(12):
        print(f"  {n:4d}  [{level}] {problem}")
    print("\nfailures per base task")
    for task, n in c["tasks"].most_common(10):
        print(f"  {n:4d}  {task}")
    if pattern:
        print(f"\nlines matching {pattern!r}, by base task")
        for (task, ok, line) in sorted(c["lines"]):
            print(f"  {c['lines'][(task, ok, line)]:4d}  {'pass' if ok else 'fail'}  "
                  f"{task:24s} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

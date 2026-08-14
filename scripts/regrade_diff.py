#!/usr/bin/env python3
"""Two gradings of the same generations, compared level by level.

`functional` is the only level that executes, so it is the only one that can disagree with
itself, and `verifier_sha` cannot see that: it says which rules produced a verdict, not that
the verdict is repeatable. Verifying one cell twice returned 0.85 and then 0.875, and the
artifact behind it wrote `exec > >(tee logs/out_$$.txt)`, a background process with no `&`
anywhere in the script, whose output the sandbox reaped before reading. See
`docs/OBSERVED_FAILURES.md`.

    ./scripts/regrade_diff.py results/first results/second

Reports are paired by file name and compared per sample and per level. What matters is the
level and not the strict verdict: a `functional` flip on a sample that already fails another
level leaves `strict_all_levels` untouched, which is exactly how this drifted unnoticed
across earlier regrades. The first version of this comparison read `all_passed` and answered
"nothing changed" on a run where a level had visibly moved.

Nothing here changes a verdict. It exists so that the advice to verify twice comes with the
means to read the second answer.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

LEVELS = ("syntax", "submittability", "functional", "resource_fit", "safety")


def _reports(directory: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(directory.glob("*.json"))}


def _level(result: dict, name: str) -> dict | None:
    return next((lr for lr in result["levels"] if lr["level"] == name), None)


def compare(first: Path, second: Path) -> tuple[Counter, list[str], list[str]]:
    """(level -> flips, human-readable flips, cells that could not be compared)."""
    a, b = _reports(first), _reports(second)
    flips: Counter = Counter()
    lines: list[str] = []
    skipped: list[str] = []

    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            skipped.append(f"{name}: present in only one of the two")
            continue
        ra = json.loads(a[name].read_text(encoding="utf-8"))
        rb = json.loads(b[name].read_text(encoding="utf-8"))
        if len(ra["results"]) != len(rb["results"]):
            skipped.append(f"{name}: {len(ra['results'])} results against {len(rb['results'])}")
            continue
        for i, (x, y) in enumerate(zip(ra["results"], rb["results"], strict=True)):
            if x["task_id"] != y["task_id"]:
                skipped.append(f"{name}: sample {i} is {x['task_id']} against {y['task_id']}")
                break
            for level in LEVELS:
                lx, ly = _level(x, level), _level(y, level)
                if lx is None or ly is None:
                    continue
                if (lx["passed"], lx["skipped"]) == (ly["passed"], ly["skipped"]):
                    continue
                flips[level] += 1
                lines.append(
                    f"  {name} sample {i} {x['task_id']} {level}: "
                    f"{lx['passed']} -> {ly['passed']}\n"
                    f"      first : {lx['detail'][:100]}\n"
                    f"      second: {ly['detail'][:100]}"
                )
    return flips, lines, skipped


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: regrade_diff.py <first results dir> <second results dir>", file=sys.stderr)
        return 2
    first, second = Path(argv[0]), Path(argv[1])
    for directory in (first, second):
        if not directory.is_dir():
            print(f"not a directory: {directory}", file=sys.stderr)
            return 1

    flips, lines, skipped = compare(first, second)
    pairs = len(set(_reports(first)) & set(_reports(second)))
    print(f"{pairs} reports compared, {len(skipped)} could not be")
    for note in skipped:
        print(f"  [skipped] {note}")
    if not pairs:
        # Every count below is zero when nothing was read, which reads as agreement.
        print("nothing was compared: the two directories share no report name", file=sys.stderr)
        return 1
    if not flips:
        print("no level changed its verdict between the two gradings")
        return 0
    print("\nlevel verdicts that changed:")
    for level, n in flips.most_common():
        print(f"  {level:<16} {n}")
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

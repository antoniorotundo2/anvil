#!/usr/bin/env python3
"""What does a model ask for when the specification does not say?

`tasks/t1_exec.jsonl` states no memory minimum on purpose: the payload holds 64MB and the
prompt asks for enough memory to fit it, so the only ground truth is what the script
actually needs. That makes the memory request a free choice, and this script reads what
the model chose.

The question it answers is the one F8 is waiting for. F8 is induced mechanically, so it
proves the harness can catch an under-request; it says nothing about whether a model
commits one. Here the requests are real, and the two executors give the verdict: `bash`
ignores the number entirely, real submission holds the job to it.

    ./scripts/memory_request.py results/executor_<stamp>

Reads the per-cell results the executor ablation writes, `<cell>__bash.json` and
`<cell>__sbatch.json`. Nothing here touches the verifier: this is analysis of a finding.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anvil.parse import parse_directives, parse_mem_to_mb  # noqa: E402

# What the payload really needs, measured rather than assumed: 32M is killed and 256M is
# not, so the boundary sits between them. Reported alongside every request, since a
# request is only wrong relative to a need.
PAYLOAD_MB = 64
KNOWN_KILLED_MB = 32
KNOWN_SURVIVED_MB = 256


def terminal_state(detail: str) -> str:
    m = re.search(r"ended (\w+)", detail)
    if m:
        return m.group(1)
    if "COMPLETED" in detail:
        return "COMPLETED"
    if "level skipped" in detail:
        return "skipped"
    if "refused" in detail:
        return "refused at submit"
    if "expected output" in detail:
        return "wrong output"
    return "other"


def main(run_dir: str) -> int:
    run = Path(run_dir)
    pairs: dict[str, dict[str, dict]] = {}
    for f in sorted(run.glob("*__*.json")):
        cell, executor = f.stem.rsplit("__", 1)
        if executor in ("bash", "sbatch"):
            pairs.setdefault(cell, {})[executor] = json.loads(f.read_text(encoding="utf-8"))
    if not pairs:
        print(f"no *__bash.json / *__sbatch.json pairs in {run}", file=sys.stderr)
        return 2

    print(f"payload holds ~{PAYLOAD_MB}MB; {KNOWN_KILLED_MB}M is killed, "
          f"{KNOWN_SURVIVED_MB}M is not\n")
    header = f"{'cell':<44}{'--mem':>10}{'MB':>8}  {'bash':<6}{'sbatch':<10}state"
    print(header)
    print("-" * len(header))

    requests: list[int | None] = []
    stopped = 0
    for cell, per in sorted(pairs.items()):
        if len(per) < 2:
            continue
        # strict: the two arms verified the same generations file in the same order, so a
        # length mismatch means one of them did not finish and the pairing is meaningless.
        for a, b in zip(per["bash"]["results"], per["sbatch"]["results"], strict=True):
            mem = parse_directives(a["script"]).get("--mem")
            mb = parse_mem_to_mb(mem) if mem else None
            requests.append(mb)
            fa = next(x for x in a["levels"] if x["level"] == "functional")
            fb = next(x for x in b["levels"] if x["level"] == "functional")
            if fa["passed"] and not fb["passed"] and not fb["skipped"]:
                stopped += 1
            print(f"{cell[:43]:<44}{str(mem):>10}{str(mb):>8}  "
                  f"{'pass' if fa['passed'] else 'FAIL':<6}"
                  f"{('pass' if fb['passed'] else 'skip' if fb['skipped'] else 'FAIL'):<10}"
                  f"{terminal_state(fb['detail'])}")

    stated = [m for m in requests if m is not None]
    print(f"\n{len(requests)} samples, {len(requests) - len(stated)} without any --mem")
    if stated:
        print(f"  requested MB: min {min(stated)}, median {sorted(stated)[len(stated)//2]}, "
              f"max {max(stated)}")
        below = [m for m in stated if m <= KNOWN_KILLED_MB]
        print(f"  {len(below)} of {len(stated)} at or below the {KNOWN_KILLED_MB}M "
              "known to be killed")
        print("  distribution:", dict(Counter(sorted(stated))))
    print(f"  {stopped} samples pass under bash and are stopped by real submission")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: memory_request.py <executor ablation directory>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))

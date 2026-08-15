#!/usr/bin/env python3
"""The commands a regrade needs, derived from what is in the directories rather than typed.

A change to `verifier.py` moves `verifier_sha`, which marks every leaderboard row until the
same generations are graded again. That regrade is three ablation runs and twenty-one
imports, and handing the sequence over by hand cost two mistakes in one session: the
execution matrix verified against `tasks/t1_slurm.jsonl`, because `executor_ablation.sh`
defaults to it and the override was forgotten, so sixty cells failed on unknown task ids;
and a closing sequence whose `git commit` was not chained to the checks, which pushed a red
`main`.

    ./scripts/regrade_plan.py results/20260802_091236 results/20260808_161623

Each argument is inspected rather than assumed. A directory holding `*.generations.jsonl`
gets an ablation command with `TASKS` and `REPAIR_TASKS` read from the digest the
generations carry, which is the only thing that actually ties them to a task file: the
directory names were chosen by hand at different times and one of them names a model while
holding the whole matrix. A directory holding reports gets its import commands instead,
grouped by model, task file and executor as the entries are keyed.

Commands are printed, not run. Every one of them takes minutes to hours or writes to the
published entries, and reading them first is what caught both mistakes above.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anvil.cli import _file_sha  # noqa: E402
from anvil.provenance import verifier_sha  # noqa: E402

# A T1 task file and the repair file induced from it. `executor_ablation.sh` needs both,
# since one run covers a from-scratch matrix and the repairs beside it.
PAIRS = {
    "tasks/t1_slurm.jsonl": "tasks/t2_repair.jsonl",
    "tasks/t1_exec.jsonl": "tasks/t2_exec_repair.jsonl",
}


def digests() -> dict[str, str]:
    """digest -> repository-relative task file, for every task file that exists."""
    out = {}
    for path in sorted((ROOT / "tasks").glob("*.jsonl")):
        out[_file_sha(path)] = f"tasks/{path.name}"
    return out


def _generation_digests(directory: Path) -> set[str]:
    seen = set()
    for path in sorted(directory.glob("*.generations.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                sha = record.get("tasks_sha") or record.get("repair_tasks_sha")
                if sha:
                    seen.add(sha)
                break
    return seen


def ablation_command(directory: Path, out: str) -> tuple[str | None, str | None]:
    """(command, complaint). One of the two is always None."""
    known = digests()
    found = _generation_digests(directory)
    unknown = sorted(d for d in found if d not in known)
    if unknown:
        return None, (f"{directory}: generations carry {unknown}, which no task file in this "
                      f"checkout matches. They were made against another version of the set.")

    files = {known[d] for d in found}
    # A T1 set, by the naming the repository uses; the reference files hold `{id, script}`
    # and are never verified against.
    t1 = sorted(f for f in files
                if f.startswith("tasks/t1_") and not f.endswith("_reference.jsonl"))
    if len(t1) != 1:
        return None, (f"{directory}: {sorted(files) or 'no digests'} does not name exactly one "
                      f"T1 task file, so the set to verify against is ambiguous.")

    tasks = t1[0]
    # `REPAIR_TASKS` only where induced repairs exist. `tasks/t1_coreutils.jsonl` has none,
    # and naming a repair file that has nothing to do with the run is how the override went
    # wrong in the first place.
    repair = PAIRS.get(tasks)
    prefix = f"TASKS={tasks}" + (f" REPAIR_TASKS={repair}" if repair else "")
    return (f"{prefix} \\\n"
            f"  OUT={out} bash scripts/executor_ablation.sh {directory}"), None


def import_commands(directory: Path, quantization: str, n: int) -> list[str]:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        key = (report["model"], report["tasks_file"],
               report["environment"]["functional_executor"])
        groups[key].append(str(path))

    lines = []
    for (_model, _tasks_file, _executor), files in sorted(groups.items()):
        seeds = ",".join(str(i) for i in range(len(files)))
        lines.append(f"./scripts/leaderboard.py import {' '.join(files)} "
                     f"--seeds {seeds} --n {n} --quantization {quantization} "
                     f"--source {directory}")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--quantization", default="4-bit")
    parser.add_argument("--n", type=int, default=5, help="samples per task in the run")
    parser.add_argument("--out-prefix", default="results/regrade",
                        help="OUT for each ablation, with the verifier digest and the source "
                             "directory name appended")
    args = parser.parse_args(argv)

    complaints, planned, ablations = [], 0, 0
    for name in args.directories:
        directory = Path(name)
        if not directory.is_dir():
            complaints.append(f"{directory}: not a directory")
            continue
        if any(directory.glob("*.generations.jsonl")):
            # The digest belongs in the name. Without it a second regrade lands on the
            # first one's directory, and `executor_ablation.sh` resumes rather than
            # overwrites: every cell is skipped and the reports it returns are the old
            # ones, graded by the rules the regrade exists to replace. It also makes a
            # directory say which verifier produced it, which is the question that took a
            # session to answer for the directories named before this existed.
            out = f"{args.out_prefix}_{verifier_sha()}_{directory.name}"
            if Path(out).exists():
                complaints.append(f"{out} already exists: this grading has been run, and "
                                  f"rerunning resumes it rather than redoing it")
                continue
            command, complaint = ablation_command(directory, out)
            if complaint:
                complaints.append(complaint)
            else:
                print(f"# {directory}: generations, verify them again")
                print(command)
                planned += 1
                ablations += 1
        elif any(directory.glob("*.json")):
            print(f"# {directory}: reports, import them")
            for line in import_commands(directory, args.quantization, args.n):
                print(line)
            planned += 1
        else:
            complaints.append(f"{directory}: holds neither generations nor reports")

    for complaint in complaints:
        print(f"[skipped] {complaint}", file=sys.stderr)
    if not planned:
        # Printing nothing and exiting 0 would read as a regrade with no work in it.
        print("nothing to plan", file=sys.stderr)
        return 1
    # Keyed on what was actually planned, not on what was passed: a directory whose
    # generations were refused leaves nothing to follow up on.
    if ablations:
        print("\n# then run this again on the OUT directories above to get the imports,")
        print("# and finish with:")
        print("#   ./scripts/leaderboard.py render && ./scripts/paper_data.py && \\")
        print("#     make test && make lint && make guards && make guards-t2")
    return 0 if not complaints else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

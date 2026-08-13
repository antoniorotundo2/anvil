#!/usr/bin/env python3
"""What models wrote where the prompt names a value and no level checks it.

`required_directives` asks whether a directive is written, never what it says, which is the
refusal of surface-form matching argued in `docs/DESIGN.md`. The measured consequence is
that four T1 tasks accept an artifact that does not do what their prompt asked: the log
paths, the dependency target, the container image and bind mount, and the size of the job
array all pass unread, under `bash` and under real submission alike.

    ./scripts/unchecked_values.py 'results/*/*__bash.json'

Only artifacts the verifier passed are counted, since the question is what the rules let
through. Nothing here changes a verdict. It exists because the fix costs a regeneration of
`tasks/t1_slurm.jsonl`, and a decision that expensive should rest on how often the opening
was actually taken rather than on the demonstration that it exists.

The expected values are the ones the prompts name, transcribed here rather than derived:
they live in English inside `prompt`, which is exactly why no level can compare them.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anvil.parse import directive_value, parse_directives  # noqa: E402


def _index_set(raw: str) -> str:
    """`--array=1-5%5` and `--array=1-5` name the same five tasks. The `%N` suffix caps how
    many run at once and leaves the index set alone, so comparing the raw string filed 229
    artifacts as deviating when none of them was: every one of the five tasks the prompt asks
    for is there. A step, `1-5:2`, does change the set and is deliberately left in place.
    """
    return raw.split("%", 1)[0].strip()


# Values that need reading before comparing, because more than one spelling means the same
# request. Everything else is compared as written.
NORMALISE = {"--array": _index_set}

# task -> label -> (directives to read, the value the prompt names)
DIRECTIVES: dict[str, dict[str, tuple[tuple[str, ...], str]]] = {
    "t1_hello_serial": {"--job-name": (("--job-name", "-J"), "hello")},
    "t1_output_paths": {
        "--output": (("--output", "-o"), "logs/out_%j.txt"),
        "--error": (("--error", "-e"), "logs/err_%j.txt"),
        "--job-name": (("--job-name", "-J"), "io_test"),
    },
    "t1_dependency_chain": {
        "--dependency": (("--dependency", "-d"), "afterok:12345"),
        "--job-name": (("--job-name", "-J"), "stage2"),
    },
    "t1_array_job": {"--array": (("--array", "-a"), "1-5")},
}

# task -> label -> the string the prompt names, looked for anywhere in the script. These
# are arguments to a command rather than directives, so there is nothing to parse.
BODY: dict[str, dict[str, str]] = {
    "t1_container_apptainer": {
        "image": "/opt/images/app.sif",
        "bind source": "/scratch/data",
        "bind target": "/data",
    },
}


def scan(paths: list[Path]) -> tuple[Counter, Counter, int, int]:
    """(task, label, verdict) counts, what was written when it differed, artifacts passed,
    and artifacts belonging to a task this script says nothing about."""
    verdicts: Counter = Counter()
    written: Counter = Counter()
    passed = untracked = 0
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        model = report.get("model", "unknown")
        for r in report["results"]:
            if not r["all_passed"]:
                continue
            passed += 1
            base = r["task_id"].split("__", 1)[0]
            if base not in DIRECTIVES and base not in BODY:
                # Counted rather than ignored: a run pointed at the wrong task file would
                # otherwise print zeros everywhere, which reads as a clean result.
                untracked += 1
                continue
            directives = parse_directives(r["script"])
            for label, (aliases, want) in DIRECTIVES.get(base, {}).items():
                raw = directive_value(directives, *aliases)
                if raw is None:
                    verdict = "absent"
                else:
                    got = NORMALISE.get(label, str)(raw.strip())
                    verdict = "as asked" if got == want else "other"
                verdicts[(base, label, verdict)] += 1
                if verdict == "other":
                    # The raw string, not the normalised one: the point of this list is to
                    # be read, and a value stripped of what made it differ cannot be.
                    written[(base, label, model, r["task_id"], raw.strip())] += 1
            for label, want in BODY.get(base, {}).items():
                verdict = "as asked" if want in r["script"] else "absent"
                verdicts[(base, label, verdict)] += 1
    return verdicts, written, passed, untracked


def report(verdicts: Counter, written: Counter, passed: int, untracked: int) -> None:
    print(f"{passed} artifacts passed the verifier, {untracked} on tasks not tracked here\n")
    tracked = sorted({(task, label) for task, label, _ in verdicts})
    width = max((len(f"{t} {lab}") for t, lab in tracked), default=20)
    print(f"{'task and value':<{width}}  {'as asked':>9} {'other':>7} {'absent':>7}")
    for task, label in tracked:
        counts = {v: verdicts[(task, label, v)] for v in ("as asked", "other", "absent")}
        print(f"{task + ' ' + label:<{width}}  {counts['as asked']:>9} "
              f"{counts['other']:>7} {counts['absent']:>7}")
    if written:
        print("\nwhat was written instead:")
        # The task is in the key so two tasks cannot merge their rows, and printed through
        # `task_id`, which carries it plus the repair suffix where there is one.
        for (_task, label, model, task_id, raw), n in sorted(written.items()):
            print(f"  {task_id:<26} {label:<12} {raw:<28} {model} x{n}")


def _runs_that_exist(limit: int = 8) -> list[str]:
    """Directories under `results/` holding reports this script can read. Printed when a
    pattern matches nothing, because `RUN` in the usage line is a placeholder and reads as
    a name: it was pasted literally twice before this existed, on a machine where the
    answer was one `ls` away and the message did not say so."""
    runs = {p.parent for p in (ROOT / "results").glob("*/*__bash.json")}
    return sorted(str(p.relative_to(ROOT)) for p in runs)[:limit]


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: unchecked_values.py 'results/<run>/*__bash.json' ...", file=sys.stderr)
        print("       <run> is a directory name, not a literal. Try 'results/*/*__bash.json'",
              file=sys.stderr)
        return 2
    paths = [Path(p) for pattern in argv for p in sorted(glob.glob(pattern))]
    if not paths:
        print(f"no report matched: {' '.join(argv)}", file=sys.stderr)
        found = _runs_that_exist()
        if found:
            print("runs that do hold reports:", file=sys.stderr)
            for run in found:
                print(f"  {run}", file=sys.stderr)
        else:
            print("no directory under results/ holds a *__bash.json report either: "
                  "run scripts/run_experiments.sh first", file=sys.stderr)
        return 1
    report(*scan(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

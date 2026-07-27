#!/usr/bin/env python3
"""Why does retrieved context cost `resource_fit`, and nothing else?

The ablation found one clean effect: `resource_fit` falls from 0.49 zero-shot to 0.19
vectorless while the other four levels barely move. This script tests the mechanisms
that could produce that shape. Two have been tested and both are refuted; the outcome
is recorded in `docs/DESIGN.md` under Retrieval ablation.

**Copying.** The corpus states concrete values, so the model might reproduce them
instead of deriving the ones the task asks for. Refuted: the arm that collapses
reproduces them least, and no sample used a retrieved value where it was wrong.

**Omission.** Retrieval does suppress directives, and `check_resource_fit` passes only
on an empty problem list, so one absent directive sinks a sample. Refuted as the
mechanism: the share of failures that are omissions does not rise with the damage.

The design point that makes any of this readable is the **zero-shot arm as a control**.
A value the model would have written anyway is evidence of nothing, and a count that
moves must be compared against the arm that saw no documents at all.

    ./scripts/retrieval_copying.py results/retrieval_20260727_160402

Reads the generations and the per-cell results that `retrieval_ablation.sh` saves.
Nothing here touches the verifier: this is analysis of a finding, not part of the
bracket.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anvil.parse import parse_directives  # noqa: E402

CORPUS = ROOT / "tasks" / "retrieval_corpus.jsonl"
TASKS = ROOT / "tasks" / "t1_slurm.jsonl"

# A literal worth tracking is a directive written out with a concrete value. Documents
# also spell out shapes like `--gres=gpu:N` and `--dependency=afterok:JOBID`, which no
# model can copy into a working script; counting them would add rows that are zero
# everywhere. Every placeholder in this corpus is marked by an uppercase run, and no
# real value in it has one, so that is the test.
LITERAL = re.compile(r"--[a-z][a-z-]*=[^\s,;.`'\"]+")
PLACEHOLDER = re.compile(r"[A-Z]")


def corpus_literals() -> dict[str, set[str]]:
    """Concrete `--key=value` strings each document states, keyed by document id."""
    out: dict[str, set[str]] = {}
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        found = set()
        for hit in LITERAL.findall(doc["text"]):
            value = hit.split("=", 1)[1]
            if not PLACEHOLDER.search(value):
                found.add(hit)
        out[doc["id"]] = found
    return out


def load_tasks() -> dict[str, dict]:
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


# `check_resource_fit` collects problems and passes only when the list is empty, so a
# single omission sinks the whole sample. That makes the level's fall much larger than
# the fall in directive count, and it also means the two causes can be told apart: a
# problem either says a directive is absent or says its value is wrong.
OMISSION = re.compile(r"missing|not requested|not declared")


def resource_fit_problems(run: Path) -> dict[str, dict[str, int]]:
    """Per arm, how many resource_fit problems were omissions and how many wrong values."""
    per_arm: dict[str, dict[str, int]] = defaultdict(lambda: {"omitted": 0, "wrong value": 0})
    for f in sorted(run.glob("*.json")):
        if f.name.endswith(".generations.jsonl"):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        arm = data.get("retrieval", "zero-shot")
        for result in data.get("results", []):
            for level in result["levels"]:
                if level["level"] != "resource_fit" or level["passed"] or level["skipped"]:
                    continue
                for problem in level["detail"].split("; "):
                    key = "omitted" if OMISSION.search(problem) else "wrong value"
                    per_arm[arm][key] += 1
    return per_arm


def main(run_dir: str) -> int:
    run = Path(run_dir)
    gens = sorted(run.glob("*.generations.jsonl"))
    if not gens:
        print(f"no *.generations.jsonl in {run}", file=sys.stderr)
        return 2

    lits = corpus_literals()
    tasks = load_tasks()
    every_literal = {lit for s in lits.values() for lit in s}
    if not every_literal:
        print("the corpus states no concrete directive values: nothing to copy")
        return 0

    # arm -> literal -> [times it appears, times it appears when that doc was retrieved]
    seen: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    scripts_per_arm: dict[str, int] = defaultdict(int)
    per_arm_directives: dict[str, list[int]] = defaultdict(list)
    wrong_and_copied = []

    for path in gens:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            arm = g.get("retrieval", "zero-shot")
            scripts_per_arm[arm] += 1
            shown = set(g.get("retrieved_docs") or [])
            shown_literals = {lit for d in shown for lit in lits.get(d, ())}

            script = g["script"]
            for lit in every_literal:
                if lit in script:
                    seen[arm][lit][0] += 1
                    if lit in shown_literals:
                        seen[arm][lit][1] += 1

            # A copied value only matters if it is also wrong for this task.
            task = tasks.get(g["task_id"], {})
            constraints = task.get("constraints", {})
            directives = parse_directives(script)
            per_arm_directives[arm].append(len(directives))
            for key, expected in (("nodes", "nodes"), ("ntasks", "ntasks")):
                want = constraints.get(expected)
                got = directives.get(key)
                if want is None or got is None:
                    continue
                if str(want) != str(got).strip() and f"--{key}={got}" in shown_literals:
                    wrong_and_copied.append(
                        (arm, g["task_id"], f"--{key}={got}", f"task asks {want}")
                    )

    arms = sorted(scripts_per_arm, key=lambda a: (a != "zero-shot", a))
    if "zero-shot" not in arms:
        print("WARNING: no zero-shot cells here, so there is no control arm and the")
        print("         counts below cannot separate copying from the model's own priors.\n")

    print("scripts per arm:", {a: scripts_per_arm[a] for a in arms})
    print("\nHow often each corpus literal appears in a generated script.")
    print("'retrieved' counts only the scripts whose own prompt carried that document.\n")
    width = max(len(x) for x in every_literal)
    header = f"{'literal':<{width}}" + "".join(f"{a:>24}" for a in arms)
    print(header)
    print("-" * len(header))
    base_arm = "zero-shot" if "zero-shot" in arms else None
    for lit in sorted(every_literal):
        cells = []
        for a in arms:
            total, retrieved = seen[a][lit]
            mark = ""
            if base_arm and a != base_arm:
                delta = total - seen[base_arm][lit][0]
                mark = " up" if delta > 0 else (" down" if delta < 0 else " same")
            cells.append(f"{total:>3} ({retrieved} retr){mark}".rjust(24))
        print(f"{lit:<{width}}" + "".join(cells))

    print("\nValues that were both copied from a retrieved document and wrong for the task:")
    if not wrong_and_copied:
        print("  none found")
    else:
        for arm, task_id, lit, why in sorted(set(wrong_and_copied)):
            print(f"  {arm:<12} {task_id:<24} {lit:<16} {why}")

    print("\nDirectives written per script, by arm.")
    print("A resource never requested fails resource_fit exactly as a wrong value does,")
    print("so omission is the other way retrieval could move that level.\n")
    for a in arms:
        n = len(per_arm_directives[a])
        mean = sum(per_arm_directives[a]) / n if n else 0.0
        empty = sum(1 for c in per_arm_directives[a] if c == 0)
        print(f"  {a:<12} mean {mean:5.2f} directives   {empty:>3} of {n} scripts had none")

    problems = resource_fit_problems(run)
    if problems:
        print("\nWhy resource_fit failed, problem by problem.")
        print("The level passes only with an empty problem list, so one omission sinks a")
        print("whole sample: that is how a modest drop in directives becomes a large drop here.\n")
        for a in arms:
            counts = problems.get(a)
            if not counts:
                continue
            total = sum(counts.values())
            share = (counts["omitted"] / total * 100) if total else 0.0
            print(f"  {a:<12} {counts['omitted']:>3} omitted, "
                  f"{counts['wrong value']:>3} wrong value   "
                  f"({share:.0f}% of problems are omissions)")

    print(
        "\nReading this: a literal is evidence of copying when its count under an arm that\n"
        "retrieved it clearly exceeds its zero-shot count. Equal counts mean the model was\n"
        "going to write that value anyway, and the document had nothing to do with it."
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-4], file=sys.stderr)
        print("usage: retrieval_copying.py <run directory>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))

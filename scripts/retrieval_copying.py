#!/usr/bin/env python3
"""Did the model copy directive values out of the documents it was shown?

The retrieval ablation found one clean effect: `resource_fit` falls from 0.49
zero-shot to 0.19 vectorless, while the other four levels barely move. A uniform
penalty, such as reference text crowding out the instructions, does not predict that
shape. Copying does: the corpus states concrete values (`--nodes=2`, `--array=1-5`,
`--output=logs/out_%j`), and a model that reproduces them instead of deriving the
ones the task asks for would fail exactly the level that scores the effective
request against the spec, and no other.

This script measures that, and its whole design rests on one point. Finding
`--nodes=2` in vectorless scripts proves nothing on its own: the value may simply be
a common prior for a small code model. What carries the argument is the **zero-shot
arm as a control**, where the same model saw no documents at all. Only a literal that
appears far more often once it has been retrieved is evidence of copying.

    ./scripts/retrieval_copying.py results/retrieval_20260727_160402

Reads the generations that `retrieval_ablation.sh` saves beside every cell. Nothing
here touches the verifier: this is analysis of a finding, not part of the bracket.
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
    header = f"{'literal':<{width}}" + "".join(f"{a:>22}" for a in arms)
    print(header)
    print("-" * len(header))
    for lit in sorted(every_literal):
        cells = []
        for a in arms:
            total, retrieved = seen[a][lit]
            cells.append(f"{total:>3} ({retrieved} retrieved)".rjust(22))
        base = seen["zero-shot"][lit][0] if "zero-shot" in arms else None
        flag = ""
        if base is not None:
            worst = max(seen[a][lit][0] for a in arms if a != "zero-shot")
            if worst > base:
                flag = "   <-- more common once retrieved"
        print(f"{lit:<{width}}" + "".join(cells) + flag)

    print("\nValues that were both copied from a retrieved document and wrong for the task:")
    if not wrong_and_copied:
        print("  none found")
    else:
        for arm, task_id, lit, why in sorted(set(wrong_and_copied)):
            print(f"  {arm:<12} {task_id:<24} {lit:<16} {why}")

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

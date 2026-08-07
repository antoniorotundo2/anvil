#!/usr/bin/env python3
"""The leaderboard, generated from entries rather than typed.

A table of numbers maintained by hand drifts from the runs it claims to summarise. This
project has already paid for that once, with a published table that had been graded
against the wrong scheduler and read as a result for weeks. So the page is not edited:
`leaderboard/entries/*.json` are the record, this script renders them, and `--check` fails
if the rendered page and the entries disagree.

    ./scripts/leaderboard.py import results/<cell>.json --seeds 0,1,2   # add or update
    ./scripts/leaderboard.py render                                     # write the page
    ./scripts/leaderboard.py --check                                    # fail on drift

An entry carries the figures and the conditions they were obtained under: which task file,
its digest, the base image, the executor, how many seeds and samples. The digests are the
ones that matter. Two entries for the same model against different task sets are not
comparable and must not sit in the same ranking, so an entry whose `tasks_sha` is not the
current one is rendered as stale rather than ranked. The same holds for `verifier_sha`, and
for the same reason: the walltime floor moved 123 verdicts of 2421 without touching a
single task file, and rows graded on either side of it are two series, not one.

Averaging across seeds happens at import: a cell is one seed, and an entry is the mean and
half-range over the seeds given. Half-range, not a confidence interval, and the page says
so, because three draws do not support a significance claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_FOR_ANVIL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOR_ANVIL))

from anvil.provenance import verifier_sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_manifest import build as build_manifest  # noqa: E402

ENTRIES = ROOT / "leaderboard" / "entries"
PAGE = ROOT / "docs" / "LEADERBOARD.md"
LEVELS = ["syntax", "submittability", "functional", "resource_fit", "safety", "strict_all_levels"]


def _current_digests() -> dict[str, str]:
    return {f["path"]: f["sha256"][:12] for f in build_manifest()["files"]}


def _slug(text: str) -> str:
    return text.replace("/", "_").replace(":", "-")


def _not_comparable(entry: dict, current_tasks_sha: str | None) -> str | None:
    """Why this row cannot be ranked against the others, or None if it can.

    Task set first: a row measured on different tasks answers different questions, and that
    is the older and more obvious of the two. Rules second, and only reported when the task
    set is current, because a row that is wrong on both is not made clearer by saying so
    twice.
    """
    if entry["tasks_sha"] != current_tasks_sha:
        return "stale tasks"
    recorded = entry.get("verifier_sha", "unstamped")
    if recorded == "unstamped":
        return "unstamped"
    if recorded != verifier_sha():
        return "stale rules"
    return None


def cmd_import(args: argparse.Namespace) -> int:
    cells = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.results]
    models = {c["model"] for c in cells}
    tasks = {c["tasks_file"] for c in cells}
    if len(models) != 1 or len(tasks) != 1:
        print(f"one entry is one model on one task file, got {models} and {tasks}", file=sys.stderr)
        return 2

    scores: dict[str, list[float]] = {level: [] for level in LEVELS}
    for cell in cells:
        for level in LEVELS:
            if level in cell["summary"]:
                scores[level].append(cell["summary"][level]["pass@1"])

    entry = {
        "model": cells[0]["model"],
        "tasks_file": cells[0]["tasks_file"],
        # Taken from the report when it carries one. Before reports were stamped there was
        # nothing to take, and the digest of the task file as it stands now was the only
        # value available: that is a guess, and it reads as fresh even for a cell graded
        # against an older task set. Reports written from now on settle it.
        "tasks_sha": cells[0].get(
            "tasks_sha", _current_digests().get(cells[0]["tasks_file"], "unknown")
        ),
        "verifier_sha": cells[0].get("verifier_sha", "unstamped"),
        "seeds": args.seeds,
        "n_per_task": args.n,
        "executor": cells[0]["environment"].get("functional_executor", "bash"),
        "base_image": cells[0]["environment"].get("base_image", "unknown"),
        "quantization": args.quantization,
        "source": args.source or "",
        # Six decimals stored, three displayed. Rounding at import and rounding again at
        # render are not the same operation: a half-range of 0.004583 stored as 0.0045
        # renders as 0.004, while `executor_ablation.sh` prints 0.005 from the same run.
        # The same measurement showing two values in one repository is the defect this
        # whole page exists to prevent, and it appeared the first time the entries were
        # imported rather than transcribed.
        "scores": {
            level: {
                "mean": round(sum(v) / len(v), 6),
                "half_range": round((max(v) - min(v)) / 2, 6),
                "cells": len(v),
            }
            for level, v in scores.items() if v
        },
    }
    # The paper's tables and figures are a projection of these entries, and `tests/test_paper.py`
    # fails when the two disagree. Importing without regenerating them broke main once.
    print("entries changed: run ./scripts/paper_data.py and rebuild the paper", file=sys.stderr)
    ENTRIES.mkdir(parents=True, exist_ok=True)
    path = ENTRIES / f"{_slug(entry['model'])}__{Path(entry['tasks_file']).stem}.json"
    path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return 0


def _load_entries() -> list[dict]:
    if not ENTRIES.exists():
        return []
    return sorted(
        (json.loads(p.read_text(encoding="utf-8")) for p in ENTRIES.glob("*.json")),
        key=lambda e: (e["tasks_file"], -e["scores"].get("strict_all_levels", {}).get("mean", 0)),
    )


def _cell(entry: dict, level: str) -> str:
    s = entry["scores"].get(level)
    return "n/a" if s is None else f"{s['mean']:.3f}±{s['half_range']:.3f}"


def render() -> str:
    current = _current_digests()
    entries = _load_entries()
    out = [
        "# Leaderboard",
        "",
        "Generated from `leaderboard/entries/`, never edited by hand: run",
        "`./scripts/leaderboard.py render`. Every figure is `pass@1`, the mean across seeds with",
        "half the range beside it. Half the range is not a confidence interval, and with three",
        "seeds no significance is claimed anywhere on this page.",
        "",
        "`strict_all_levels` is the ranking column: it requires every level either to pass or to",
        "be out of the machine's reach, and a skipped level is never a passed one.",
        "",
    ]
    by_tasks: dict[str, list[dict]] = {}
    for entry in entries:
        by_tasks.setdefault(entry["tasks_file"], []).append(entry)

    for tasks_file, group in by_tasks.items():
        out.append(f"## `{tasks_file}`")
        out.append("")
        out.append("| model | " + " | ".join(f"`{lv}`" for lv in LEVELS) + " | conditions |")
        out.append("|---" * (len(LEVELS) + 2) + "|")
        for entry in group:
            why = _not_comparable(entry, current.get(tasks_file))
            name = f"{entry['model']}{f' ({why})' if why else ''}"
            conditions = (
                f"{len(entry['seeds'])} seeds, n={entry['n_per_task']}, "
                f"{entry['executor']}, {entry['quantization']}"
            )
            out.append(
                f"| {name} | " + " | ".join(_cell(entry, lv) for lv in LEVELS)
                + f" | {conditions} |"
            )
        out.append("")
        reasons = {_not_comparable(e, current.get(tasks_file)) for e in group} - {None}
        if reasons:
            out.append(
                "An entry marked *stale tasks* was measured against a different version of this "
                "task file; one marked *stale rules* was graded by a different verifier, and "
                "*unstamped* predates the digest being recorded at all. Any of the three means "
                "the row is not comparable with the rest of the column. They are shown rather "
                "than deleted."
            )
            out.append("")

    out += [
        "## Getting on it",
        "",
        "Generate where the accelerator is, grade where the scheduler is, then import the cells:",
        "",
        "```",
        "anvil run --model <model id> --tasks tasks/t1_slurm.jsonl -n 5 --seed 0 \\",
        "  --save-generations gen_seed0.jsonl",
        "./scripts/executor_ablation.sh <the directory holding the generations>",
        "./scripts/leaderboard.py import <the per-cell result files> --seeds 0,1,2 --n 5",
        "./scripts/leaderboard.py render",
        "```",
        "",
        "Grading outside the container is not accepted: the levels that depend on the scheduler",
        "are the ones a wrong environment silently changes, which is recorded in",
        "[OBSERVED_FAILURES.md](OBSERVED_FAILURES.md#a-table-measured-against-the-wrong-cluster).",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    i = sub.add_parser("import", help="turn per-seed result files into one entry")
    i.add_argument("results", nargs="+")
    i.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], required=True)
    i.add_argument("--n", type=int, required=True, help="samples per task")
    i.add_argument("--quantization", default="4-bit")
    i.add_argument("--source", default="", help="where the result files came from")
    i.set_defaults(func=cmd_import)

    r = sub.add_parser("render", help="write docs/LEADERBOARD.md")
    r.set_defaults(func=lambda a: (PAGE.write_text(render(), encoding="utf-8"), print(PAGE))[1])

    parser.add_argument("--check", action="store_true", help="fail if the page is out of date")
    args = parser.parse_args(argv)

    if args.check:
        if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != render():
            print("docs/LEADERBOARD.md is out of date: ./scripts/leaderboard.py render",
                  file=sys.stderr)
            return 1
        print("leaderboard page matches its entries")
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

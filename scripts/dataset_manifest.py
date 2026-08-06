#!/usr/bin/env python3
"""The digest of every file the benchmark is defined by.

`docs/RESULTS.md` says where three models stand. That sentence only means something while
the task files are the ones the models were graded against, and a task file is a text file
that anybody can edit without noticing what they have invalidated. The harness already
takes this seriously in one place: `anvil verify` refuses generations whose `tasks_sha`
does not match the task set in front of it. This is the same idea for the release as a
whole, one digest per file, written down.

    ./scripts/dataset_manifest.py            # write dataset/MANIFEST.json
    ./scripts/dataset_manifest.py --check    # fail if the files no longer match it

The check is a test as well (`tests/test_dataset.py`), so editing a task without
regenerating the manifest fails `make test` rather than being discovered later by someone
comparing numbers that were never comparable.

Regenerating it is the right move when a change to the task set is intended. What must not
happen is regenerating it *and* leaving the published figures in place: the numbers in
`docs/RESULTS.md` belong to the digests recorded here, and a new digest means they need
measuring again.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset" / "MANIFEST.json"

# Everything a third party needs to reproduce a published number, and nothing that is a
# result rather than an input. The reference solutions are in: without them the oracle
# cannot be run, and an upper bound nobody can reproduce is an assertion.
RELEASED = [
    "tasks/t1_slurm.jsonl",
    "tasks/t1_reference.jsonl",
    "tasks/t2_repair.jsonl",
    "tasks/t3_apptainer.jsonl",
    "tasks/t3_reference.jsonl",
    "tasks/t1_exec.jsonl",
    "tasks/t1_exec_reference.jsonl",
    "tasks/t2_exec_repair.jsonl",
    "tasks/t1_coreutils.jsonl",
    "tasks/t1_coreutils_reference.jsonl",
    "tasks/retrieval_corpus.jsonl",
]

VERSION = "0.1.0"


def _records(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build() -> dict:
    files = []
    for rel in RELEASED:
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"{rel} is in the release list and not on disk")
        data = path.read_bytes()
        files.append({
            "path": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "records": _records(path),
        })
    return {"dataset": "anvil", "version": VERSION, "files": files}


def main(argv: list[str]) -> int:
    current = build()
    if "--check" not in argv:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        total = sum(f["records"] for f in current["files"])
        print(f"{MANIFEST.relative_to(ROOT)}: {len(current['files'])} files, {total} records")
        return 0

    if not MANIFEST.exists():
        print(f"{MANIFEST.relative_to(ROOT)} does not exist: run this script without --check",
              file=sys.stderr)
        return 1

    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if recorded == current:
        print("manifest matches the files on disk")
        return 0

    was = {f["path"]: f["sha256"] for f in recorded.get("files", [])}
    now = {f["path"]: f["sha256"] for f in current["files"]}
    for path in sorted(set(was) | set(now)):
        if was.get(path) != now.get(path):
            print(f"  changed: {path}", file=sys.stderr)
    print(
        "\nthe released files no longer match dataset/MANIFEST.json.\n"
        "If the change was intended, regenerate it, and treat every figure in\n"
        "docs/RESULTS.md as needing measurement again: they belong to the old digests.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

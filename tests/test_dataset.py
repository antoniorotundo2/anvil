"""The released files must be the ones the published figures were measured against.

A task file is a text file, and editing one is easy to do without noticing which numbers it
invalidates. `anvil verify` already refuses generations whose task digest does not match;
this is the same guard aimed at the release, so the drift is caught by `make test` rather
than by somebody comparing figures that were never comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_manifest import MANIFEST, RELEASED, build  # noqa: E402


def test_the_manifest_matches_the_files_on_disk():
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert recorded == build(), (
        "dataset/MANIFEST.json is stale: re-run ./scripts/dataset_manifest.py, and treat "
        "every figure in docs/RESULTS.md as needing measurement again"
    )


def test_every_released_file_exists_and_carries_records():
    for rel in RELEASED:
        path = ROOT / rel
        assert path.is_file(), rel
        assert any(line.strip() for line in path.read_text(encoding="utf-8").splitlines()), rel


def test_the_loaders_the_guards_iterate_over_return_the_task_sets():
    """Nearly every guard in this suite is written `for task in load(...)`, and a loop over
    an empty list reports success. The manifest above catches a file that changed on disk
    and cannot see a loader that stopped reading one, which is the way the count actually
    reaches zero: a constant pointed somewhere else, a resolver aimed at an empty directory,
    a schema that quietly skips lines it no longer recognises.

    Floors rather than counts, so adding a task is not an edit here. They are the sizes the
    published figures were measured against, so a set that shrinks fails and asks why.

    The reference files are absent because they are not task files: they hold `{id, script}`
    and are read by the oracle's own loader, whose coverage is what
    `test_oracle_passes_every_task` measures.
    """
    from anvil.retrieval import Document  # noqa: PLC0415
    from anvil.schema import RecipeTask, RepairTask, Task  # noqa: PLC0415

    counts = {
        "tasks/t1_slurm.jsonl": (Task, 8),
        "tasks/t1_exec.jsonl": (Task, 2),
        "tasks/t2_repair.jsonl": (RepairTask, 44),
        "tasks/t2_exec_repair.jsonl": (RepairTask, 10),
        "tasks/t3_apptainer.jsonl": (RecipeTask, 3),
        "tasks/retrieval_corpus.jsonl": (Document, 8),
    }
    for name, (kind, floor) in counts.items():
        loaded = kind.load_jsonl(ROOT / name)
        assert len(loaded) >= floor, f"{name}: {len(loaded)} records, expected at least {floor}"


def test_the_digest_the_harness_writes_is_the_manifest_digest(tmp_path):
    """`tasks_sha` in a generations file is the first twelve characters of the same
    SHA-256. One number identifies the dataset in the manifest, in saved generations and
    in the refusal, or it identifies nothing."""
    from anvil.cli import _file_sha  # noqa: PLC0415

    recorded = {f["path"]: f["sha256"] for f in json.loads(
        MANIFEST.read_text(encoding="utf-8"))["files"]}
    for rel, digest in recorded.items():
        assert _file_sha(ROOT / rel) == digest[:12], rel


def test_every_task_file_is_covered_by_the_package_data_pattern():
    """The claim in the README, that the task files travel with `pip install`, rests on one
    glob in `pyproject.toml`. Verified by installing the package from the repository URL into
    a clean environment: all eleven files arrive and are byte-identical to `tasks/`. What
    that check cannot do is run from a checkout, where `anvil/data` does not exist until a
    build creates it, so what is pinned here is the declaration the claim depends on.

    A task file with any other extension would be dropped from the wheel without a word, and
    `anvil check --task ...` would work for whoever has the repository and fail for whoever
    followed the README.
    """
    import re

    # Read with a regex rather than by splitting on the next `[`, which is the opening
    # bracket of the list itself and leaves nothing to match. No `tomllib`: it arrived in
    # 3.11 and this package supports 3.10.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    body = pyproject.split("[tool.setuptools.package-data]", 1)[1]
    line = re.search(r'"anvil\.data"\s*=\s*\[([^\]]*)\]', body)
    assert line, "the package-data entry for anvil.data is gone"
    patterns = re.findall(r'"([^"]+)"', line.group(1))
    assert patterns == ["*.jsonl"], patterns

    suffixes = {p.suffix for p in (ROOT / "tasks").iterdir() if p.is_file()}
    assert suffixes == {".jsonl"}, f"not covered by {patterns}: {sorted(suffixes - {'.jsonl'})}"


def test_the_counts_the_dataset_page_states_are_the_counts_on_disk():
    """`DATASET.md` said the repair file holds 220 tasks. It holds 44, and 220 is what a run
    at `-n 5` produces from them, which is the figure `RESULTS.md` reports. One document
    describing the file and another describing a pass over it, with the same word for both,
    is how a reader ends up citing a dataset five times its size.
    """
    import re

    body = (ROOT / "docs" / "DATASET.md").read_text(encoding="utf-8")
    claims = re.findall(r"`(tasks/[a-z0-9_]+\.jsonl)` holds (\d+)", body)
    assert claims, "docs/DATASET.md no longer states the size of any task file"
    for path, stated in claims:
        records = sum(1 for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
                      if line.strip())
        assert records == int(stated), f"{path}: page says {stated}, file holds {records}"

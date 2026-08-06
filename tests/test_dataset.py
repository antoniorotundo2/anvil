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


def test_the_digest_the_harness_writes_is_the_manifest_digest(tmp_path):
    """`tasks_sha` in a generations file is the first twelve characters of the same
    SHA-256. One number identifies the dataset in the manifest, in saved generations and
    in the refusal, or it identifies nothing."""
    from anvil.cli import _file_sha  # noqa: PLC0415

    recorded = {f["path"]: f["sha256"] for f in json.loads(
        MANIFEST.read_text(encoding="utf-8"))["files"]}
    for rel, digest in recorded.items():
        assert _file_sha(ROOT / rel) == digest[:12], rel

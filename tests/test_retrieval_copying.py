"""The shell-expansion count in `scripts/retrieval_copying.py`.

The published figure (0, 1 and 2 scripts of 72) was taken by hand from one regrade's
`sbatch` errors, and no definition was kept, so a fourth arm could not be counted the same
way. These tests pin the definition that replaced it: which lines count, and which only
look as if they should.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_copying import expanding_directive, main  # noqa: E402


def test_the_idiom_the_corpus_teaches_counts_in_the_directive_block():
    script = "#!/bin/bash\n#SBATCH --ntasks=${SLURM_NTASKS:-4}\nsrun hostname\n"
    assert expanding_directive(script) == "#SBATCH --ntasks=${SLURM_NTASKS:-4}"


def test_every_form_bash_would_expand_counts():
    for value in ("$SLURM_NTASKS", "${N}", "$(nproc)"):
        assert expanding_directive(f"#!/bin/bash\n#SBATCH --ntasks={value}\necho x\n"), value


def test_a_late_directive_does_not_count():
    """After the first command `sbatch` reads `#SBATCH` as a comment, so the value never
    reaches it and cannot produce the error the published count was read from."""
    script = "#!/bin/bash\necho start\n#SBATCH --ntasks=${SLURM_NTASKS:-4}\n"
    assert expanding_directive(script) is None


def test_the_payload_and_trailing_comments_do_not_count():
    """The corpus teaches the idiom for the payload, where it is correct, and a `$` in a
    comment after the options is prose."""
    script = (
        "#!/bin/bash\n"
        "#SBATCH --ntasks=4 # the payload reads $SLURM_NTASKS\n"
        "#SBATCH --output=logs/out_%j.log\n"
        "srun -n ${SLURM_NTASKS:-4} ./app\n"
    )
    assert expanding_directive(script) is None


def test_the_count_is_reported_per_arm(tmp_path, capsys):
    rows = {
        "zero_shot__seed0": ("zero-shot", "#!/bin/bash\n#SBATCH --ntasks=4\nsrun x\n"),
        "dense__seed0": ("dense", "#!/bin/bash\n#SBATCH --ntasks=${SLURM_NTASKS:-4}\nsrun x\n"),
    }
    for name, (arm, script) in rows.items():
        record = {"task_id": "t1_mpi_multinode", "retrieval": arm, "retrieved_docs": [],
                  "script": script}
        (tmp_path / f"{name}.generations.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8")

    assert main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    section = out.split("Shell expansion inside the directive block")[1]
    assert "zero-shot      0 of 1 scripts" in section
    assert "dense          1 of 1 scripts" in section
    assert "t1_mpi_multinode" in section

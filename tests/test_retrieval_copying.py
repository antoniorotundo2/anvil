"""The shell-expansion count in `scripts/retrieval_copying.py`.

The published figure was taken by hand and no definition was kept, so a fourth arm could
not be counted the same way. These tests pin the definition that replaced it: which lines
count, and which only look as if they should. The late-directive case is the one that
corrected the hand count, which had included a `vector` script whose expansion sat after
the first command.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_copying import (  # noqa: E402
    copied_and_wrong,
    corpus_literals,
    expanding_directive,
    main,
)

from anvil.schema import Task  # noqa: E402


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


def test_the_curated_corpus_still_yields_the_published_literals():
    """The copying rows in DESIGN.md are these three. Teaching the reader a second corpus
    must not change what it finds in the first."""
    found = {lit for lits in corpus_literals().values() for lit in lits}
    assert found == {"--array=1-5", "--nodes=2", "--output=logs/out_%j"}


def test_a_generated_corpus_is_read_past_its_header_and_its_placeholders(tmp_path):
    """The man-page corpus opens with a `//` provenance line and writes placeholders as
    `<time>`; neither is a document, and neither is a value a model could copy."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "// generated\n"
        + json.dumps({"id": "d", "text": "--time=<time> or, say, #SBATCH --time=1", "tags": []})
        + "\n", encoding="utf-8")
    assert corpus_literals(corpus) == {"d": {"--time=1"}}


def _t1(task_id: str) -> Task:
    return next(t for t in Task.load_jsonl(ROOT / "tasks" / "t1_slurm.jsonl") if t.id == task_id)


def test_a_copied_value_the_verifier_rejects_is_reported():
    """`--nodes=2` on a one-node task is the case the old comparison existed for and never
    caught: it looked up `nodes` where the parser returns `--nodes`."""
    script = "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --time=10\n#SBATCH --mem=512M\necho x\n"
    assert copied_and_wrong(script, _t1("t1_hello_serial"), {"--nodes=2"}) == ["--nodes=2"]


def test_every_key_the_verifier_checks_is_covered():
    """A minute of walltime, copied from the man page's first example, against a task that
    declares ten."""
    script = "#!/bin/bash\n#SBATCH --time=1\n#SBATCH --mem=512M\necho x\n"
    assert copied_and_wrong(script, _t1("t1_hello_serial"), {"--time=1"}) == ["--time=1"]


def test_a_copied_value_that_is_right_or_not_used_is_not_reported():
    task = _t1("t1_mpi_multinode")
    script = ("#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --ntasks=4\n#SBATCH --time=30\n"
              "#SBATCH --mem=4G\nsrun hostname\n")
    assert copied_and_wrong(script, task, {"--nodes=2"}) == []
    assert copied_and_wrong(script, task, {"--time=1"}) == []

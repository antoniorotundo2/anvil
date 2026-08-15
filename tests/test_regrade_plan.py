"""The regrade planner, and the mistake it exists to make impossible.

`executor_ablation.sh` defaults to `tasks/t1_slurm.jsonl`, so pointing it at execution-set
generations without the override verifies them against a task file whose ids they do not
share: sixty cells failed on unknown ids in one session, and the digest that would have
explained it was sitting inside the generations the whole time. The planner reads that
digest instead of trusting the directory's name, which was chosen by hand and, in one case,
names a model while holding the whole matrix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from regrade_plan import PAIRS, ablation_command, import_commands, main  # noqa: E402

sys.path.insert(0, str(ROOT))
from anvil.cli import _file_sha  # noqa: E402


def _generations(directory: Path, name: str, tasks_file: str, repair: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    key = "repair_tasks_sha" if repair else "tasks_sha"
    record = {"task_id": "t", "sample": 0, "model": "m", "seed": 0,
              key: _file_sha(ROOT / tasks_file), "script": "#!/bin/bash\n"}
    (directory / f"{name}.generations.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8")


def test_the_task_file_comes_from_the_digest_not_the_directory_name(tmp_path):
    """The name says `slurm` and the contents say otherwise; the contents win."""
    directory = tmp_path / "20260808_misleading_slurm_name"
    _generations(directory, "a", "tasks/t1_exec.jsonl")
    _generations(directory, "repair__a", "tasks/t2_exec_repair.jsonl", repair=True)

    command, complaint = ablation_command(directory, str(tmp_path / "out"))
    assert complaint is None, complaint
    assert "TASKS=tasks/t1_exec.jsonl" in command
    assert "REPAIR_TASKS=tasks/t2_exec_repair.jsonl" in command
    assert "tasks/t1_slurm.jsonl" not in command


def test_the_main_matrix_is_planned_with_its_own_pair(tmp_path):
    directory = tmp_path / "main"
    _generations(directory, "a", "tasks/t1_slurm.jsonl")
    _generations(directory, "repair__a", "tasks/t2_repair.jsonl", repair=True)

    command, complaint = ablation_command(directory, str(tmp_path / "out"))
    assert complaint is None, complaint
    assert "TASKS=tasks/t1_slurm.jsonl REPAIR_TASKS=tasks/t2_repair.jsonl" in command


def test_generations_from_another_version_of_the_set_are_refused(tmp_path):
    """`verify` would refuse them cell by cell after the image had started. Refusing to plan
    the run at all says the same thing before it costs anything."""
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "a.generations.jsonl").write_text(
        json.dumps({"task_id": "t", "tasks_sha": "deadbeef1234",
                    "script": "#!/bin/bash\n"}) + "\n", encoding="utf-8")

    command, complaint = ablation_command(directory, str(tmp_path / "out"))
    assert command is None
    assert "deadbeef1234" in complaint


def test_reports_are_planned_as_imports_grouped_as_the_entries_are_keyed(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    for seed in (0, 1, 2):
        for executor in ("bash", "sbatch"):
            (directory / f"m__seed{seed}__{executor}.json").write_text(json.dumps({
                "model": "acme/m", "tasks_file": "tasks/t1_slurm.jsonl",
                "environment": {"functional_executor": executor},
                "summary": {}, "results": [],
            }), encoding="utf-8")

    lines = import_commands(directory, "4-bit", 5)
    assert len(lines) == 2, lines
    assert all("--seeds 0,1,2 --n 5 --quantization 4-bit" in line for line in lines)
    assert sum("__bash.json" in line for line in lines) == 1


def test_a_directory_with_neither_is_not_silently_a_plan(tmp_path, capsys):
    """Printing nothing and exiting 0 would read as a regrade that needs no work, which is
    the answer that ends the task rather than starting it."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)]) == 1
    assert "nothing to plan" in capsys.readouterr().err


def test_the_pairs_cover_the_task_files_that_have_a_repair_set():
    """A T1 file whose repairs exist but which is missing here would be planned without its
    `REPAIR_TASKS`, which is the original mistake wearing a different hat."""
    for t1, repair in PAIRS.items():
        assert (ROOT / t1).is_file(), t1
        assert (ROOT / repair).is_file(), repair
    induced = {p.name for p in (ROOT / "tasks").glob("t2_*.jsonl")}
    assert induced == {Path(r).name for r in PAIRS.values()}, induced


def test_a_t1_set_with_no_induced_repairs_is_planned_without_one(tmp_path):
    """`tasks/t1_coreutils.jsonl` has no `t2_` counterpart. Naming a repair file anyway would
    point the run at a set that has nothing to do with it, which is the original mistake
    with the arguments the other way round. Found by running the planner on the directories
    this repository actually has rather than on the fixtures above."""
    directory = tmp_path / "coreutils"
    _generations(directory, "a", "tasks/t1_coreutils.jsonl")

    command, complaint = ablation_command(directory, str(tmp_path / "out"))
    assert complaint is None, complaint
    assert "TASKS=tasks/t1_coreutils.jsonl" in command
    assert "REPAIR_TASKS" not in command


def test_the_follow_up_note_appears_only_when_a_run_was_planned(tmp_path, capsys):
    """A directory whose generations were refused leaves nothing to follow up on, and
    telling the reader to rerun on OUT directories that will never exist is instructions for
    work that is not going to happen."""
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "a.generations.jsonl").write_text(
        json.dumps({"task_id": "t", "tasks_sha": "deadbeef1234"}) + "\n", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "m__seed0__bash.json").write_text(json.dumps({
        "model": "acme/m", "tasks_file": "tasks/t1_slurm.jsonl",
        "environment": {"functional_executor": "bash"},
    }), encoding="utf-8")

    main([str(directory), str(reports)])
    out = capsys.readouterr().out
    assert "leaderboard.py import" in out
    assert "run this again on the OUT directories" not in out


def test_the_out_directory_carries_the_verifier_that_will_grade_it(tmp_path, capsys, monkeypatch):
    """Without the digest a second regrade lands on the first one's directory, and
    `executor_ablation.sh` resumes rather than overwrites: every cell is skipped and the
    reports handed back are the old ones, graded by the rules the regrade exists to replace.
    Caught by the planner printing the same OUT twice on consecutive verifier changes."""
    from anvil.provenance import verifier_sha

    directory = tmp_path / "run"
    _generations(directory, "a", "tasks/t1_slurm.jsonl")
    monkeypatch.chdir(tmp_path)

    assert main([str(directory), "--out-prefix", "out/regrade"]) == 0
    printed = capsys.readouterr().out
    assert f"OUT=out/regrade_{verifier_sha()}_run" in printed


def test_a_grading_that_already_exists_is_refused(tmp_path, capsys, monkeypatch):
    """Resuming is the right behaviour for an interrupted run and the wrong one for a new
    grading, and the two are told apart only by whether the directory is already there."""
    from anvil.provenance import verifier_sha

    directory = tmp_path / "run"
    _generations(directory, "a", "tasks/t1_slurm.jsonl")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / f"regrade_{verifier_sha()}_run").mkdir()

    assert main([str(directory), "--out-prefix", "out/regrade"]) == 1
    assert "already exists" in capsys.readouterr().err

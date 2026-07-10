"""T2 (diagnose-and-repair) tests.

Guiding principle, same as test_verifier.py: broken must mean broken, and the
oracle must mean solvable. An inducer that produces an accidentally-valid
script, or a repair verifier that is accidentally permissive, is a bug in the
harness — not a property of the model under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.inducer import FAULT_CATEGORIES, INDUCERS, induce
from anvil.parse import extract_script, misplaced_directives, parse_time_to_minutes
from anvil.repair import (
    RepairBrokenModel,
    RepairOracleModel,
    build_repair_prompt,
    induce_t2_tasks,
    verify_repair,
)
from anvil.schema import RepairTask, Task
from anvil.verifier import verify

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks" / "t1_slurm.jsonl"
REFS = ROOT / "tasks" / "t1_reference.jsonl"
T2_TASKS = ROOT / "tasks" / "t2_repair.jsonl"


def _t1_tasks() -> list[Task]:
    return Task.load_jsonl(TASKS)


def _reference() -> dict[str, str]:
    import json

    out = {}
    with open(REFS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec["script"]
    return out


# ---------------------------------------------------------------- induction
def test_every_category_has_a_description():
    assert set(INDUCERS) == set(FAULT_CATEGORIES)


def test_induction_guard_broken_means_broken():
    """Every variant induce_t2_tasks keeps must actually fail the verifier.
    This is the T2 equivalent of the oracle/broken bracket in DESIGN.md."""
    tasks = _t1_tasks()
    reference = _reference()
    repair_tasks, _warnings = induce_t2_tasks(tasks, reference)
    assert repair_tasks, "no T2 tasks were induced"

    by_base = {t.id: t for t in tasks}
    for rt in repair_tasks:
        base_task = by_base[rt.base_task_id]
        res = verify(rt.broken_script, base_task)
        assert not res.all_passed, (
            f"{rt.id} ({rt.fault_category}) verifies clean — the inducer did not "
            "actually break anything"
        )


def test_induction_covers_several_categories():
    tasks = _t1_tasks()
    repair_tasks, _ = induce_t2_tasks(tasks, _reference())
    categories = {rt.fault_category for rt in repair_tasks}
    assert len(categories) >= 5, f"only {categories} were exercised"


@pytest.mark.parametrize("task_id", [t.id for t in Task.load_jsonl(TASKS)])
def test_f2_misplaced_directive_is_actually_misplaced(task_id):
    tasks = {t.id: t for t in _t1_tasks()}
    reference = _reference()
    task, good = tasks[task_id], reference[task_id]
    broken = INDUCERS["F2"](good, task)
    if broken is None:
        pytest.skip("F2 not applicable to this task")
    assert misplaced_directives(broken)
    assert not misplaced_directives(good)


def test_f7_malformed_time_is_unparsable():
    tasks = {t.id: t for t in _t1_tasks()}
    reference = _reference()
    task, good = tasks["t1_hello_serial"], reference["t1_hello_serial"]
    broken = INDUCERS["F7"](good, task)
    assert broken is not None
    assert parse_time_to_minutes("aa:bb") is None
    assert "--time=aa:bb" in broken


def test_f5_strips_every_sbatch_line():
    tasks = {t.id: t for t in _t1_tasks()}
    reference = _reference()
    task, good = tasks["t1_hello_serial"], reference["t1_hello_serial"]
    broken = INDUCERS["F5"](good, task)
    assert broken is not None
    assert "#SBATCH" not in broken


def test_f1_drops_cpus_per_task_not_ntasks_when_both_absent_defaults_hold():
    tasks = {t.id: t for t in _t1_tasks()}
    reference = _reference()
    task, good = tasks["t1_cpus_per_task"], reference["t1_cpus_per_task"]
    broken = induce(good, task)["F1"]
    assert "--cpus-per-task" not in broken


def test_f6_breaks_the_payload_derivation():
    tasks = {t.id: t for t in _t1_tasks()}
    reference = _reference()
    task, good = tasks["t1_cpus_per_task"], reference["t1_cpus_per_task"]
    broken = INDUCERS["F6"](good, task)
    assert broken is not None
    assert "SLURM_CPUS_PER_TASK" not in broken.split("export OMP_NUM_THREADS=")[1].split("\n")[0]


# ---------------------------------------------------------------- prompt
def test_build_repair_prompt_embeds_spec_and_broken_script():
    task = Task(id="x", prompt="Write a script.")
    prompt = build_repair_prompt(task, "#!/bin/bash\necho hi\n")
    assert prompt.startswith(task.prompt)
    assert "#!/bin/bash\necho hi" in prompt


# ---------------------------------------------------------------- oracle / broken repair
def test_repair_oracle_passes_every_induced_task():
    """If this fails, either a T2 task is unsolvable by the T1 canonical
    solution, or the repair verifier is too strict."""
    tasks = _t1_tasks()
    by_base = {t.id: t for t in tasks}
    repair_tasks, _ = induce_t2_tasks(tasks, _reference())
    oracle = RepairOracleModel(REFS, tasks)

    for rt in repair_tasks:
        base_task = by_base[rt.base_task_id]
        prompt = build_repair_prompt(base_task, rt.broken_script)
        raw = oracle.generate(prompt, n=1)[0]
        res = verify_repair(extract_script(raw), rt, base_task)
        failures = [
            f"{lr.level.value}: {lr.detail}"
            for lr in res.levels
            if not lr.passed and not lr.skipped
        ]
        assert not failures, f"oracle repair failed on {rt.id}: {failures}"


def test_repair_broken_identity_fails_every_induced_task():
    """A 'repair' that changes nothing must score 0.0 strict everywhere."""
    tasks = _t1_tasks()
    by_base = {t.id: t for t in tasks}
    repair_tasks, _ = induce_t2_tasks(tasks, _reference())
    identity = RepairBrokenModel()

    for rt in repair_tasks:
        base_task = by_base[rt.base_task_id]
        prompt = build_repair_prompt(base_task, rt.broken_script)
        raw = identity.generate(prompt, n=1)[0]
        assert extract_script(raw) == rt.broken_script.strip("\n")
        res = verify_repair(extract_script(raw), rt, base_task)
        assert not res.all_passed, f"no-op repair unexpectedly passed on {rt.id}"


# ---------------------------------------------------------------- committed t2 file
def test_t2_repair_file_is_in_sync_with_current_inducers():
    """Guards against tasks/t2_repair.jsonl drifting from t1_reference.jsonl
    or from the inducer implementation without being regenerated."""
    committed = {rt.id for rt in RepairTask.load_jsonl(T2_TASKS)}
    fresh = {rt.id for rt in induce_t2_tasks(_t1_tasks(), _reference())[0]}
    assert committed == fresh, (
        "tasks/t2_repair.jsonl is stale: re-run `anvil induce` "
        "(anvil induce --out tasks/t2_repair.jsonl)"
    )

"""The execution-only task set, checked as far as a suite without a scheduler can.

F8 is the class no static check and no `bash` run can see: the memory value stays well
formed, the scheduler accepts it, and only an enforced allocation refuses the job. That
makes the tasks in `tasks/t1_exec.jsonl` the ones whose properties are least visible to
this suite, and the most worth pinning where they are visible. What execution decides is
`make docker-guards-enforcement`, which asserts that the oracle solves both tasks under
real submission and that both under-requests come back OUT_OF_MEMORY.

What is asserted here is the structure that makes that guard meaningful: that no task in
the file states a memory minimum, since one would let `resource_fit` refuse the fault
statically and F8 would never reach execution, and that every task therefore has an F8
variant in the induced set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anvil.inducer import INDUCERS, NEEDS_ENFORCEMENT, decidable  # noqa: E402
from anvil.schema import Task  # noqa: E402
from anvil.verifier import (  # noqa: E402
    check_functional,
    check_resource_fit,
    check_syntax,
    sandbox_mem_mb,
)

ROOT = Path(__file__).resolve().parents[1]
EXEC_TASKS = ROOT / "tasks" / "t1_exec.jsonl"
EXEC_REFS = ROOT / "tasks" / "t1_exec_reference.jsonl"
EXEC_REPAIRS = ROOT / "tasks" / "t2_exec_repair.jsonl"


def _records(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _tasks() -> list[Task]:
    return [Task(**r) for r in _records(EXEC_TASKS)]


def test_no_execution_task_states_a_memory_minimum():
    """The whole point of the set. With `mem_min_mb` declared, cutting `--mem` fails
    `resource_fit` before anything runs, which is F4's territory and says nothing about
    enforcement. Here the payload's real need is the only ground truth."""
    for task in _tasks():
        assert "mem_min_mb" not in task.constraints, task.id


def test_every_execution_task_has_an_f8_variant():
    ids = {r["id"] for r in _records(EXEC_REPAIRS)}
    for task in _tasks():
        assert f"{task.id}__F8" in ids, task.id


def test_the_f8_under_request_survives_every_static_check():
    """It has to, or the class would be a duplicate of F4. The cut value is well formed and
    inside no declared bound, so `syntax` and `resource_fit` both pass it and the artifact
    reaches the executor looking correct."""
    refs = {r["id"]: r["script"] for r in _records(EXEC_REFS)}
    for task in _tasks():
        broken = INDUCERS["F8"](refs[task.id], task)
        assert broken is not None, task.id
        assert check_syntax(broken).passed, task.id
        assert check_resource_fit(broken, task).passed, task.id


def test_the_two_tasks_underspend_for_different_reasons():
    """Two tasks that fail F8 by the same mechanism would be one task measured twice. The
    first hides its cost in a command substitution, which holds the pipe and the variable at
    once; the second in concurrency, where `--mem` covers the node and four workers are
    resident together. A model can reason correctly about one and not the other."""
    refs = {r["id"]: r["script"] for r in _records(EXEC_REFS)}
    assert set(refs) == {"t1_memory_bound", "t1_memory_workers"}
    assert "&" not in refs["t1_memory_bound"]
    assert "wait" in refs["t1_memory_workers"]


def test_only_f8_needs_an_enforced_allocation():
    """Every other induced fault is decidable from the text or from a sandbox run, so a guard
    without a scheduler can still conclude something about it. Adding a class here weakens
    what the pre-GPU guards check, so the set is pinned."""
    assert NEEDS_ENFORCEMENT == {"F8"}


def test_a_guard_under_bash_may_not_conclude_anything_about_f8():
    """It passes all five levels under bash by construction, which is the property F8 exists
    to demonstrate. A guard that read that as a permissive verifier refused to run the whole
    execution task set, which is how this was found."""
    assert not decidable("F8", "bash")
    assert decidable("F8", "sbatch")
    for category in ("F2", "F4", "F5", "F7"):
        assert decidable(category, "bash"), category


def test_the_execution_repair_set_leaves_the_guard_something_to_decide():
    """If every record needed enforcement the pre-GPU guard would have nothing to say, and
    saying nothing must be a hard stop rather than a pass. This set does not hit that, and the
    test records why: four of its five categories are decidable under bash."""
    categories = {r["id"].rsplit("__", 1)[-1] for r in _records(EXEC_REPAIRS)}
    assert categories - NEEDS_ENFORCEMENT


def test_the_sandbox_ceiling_is_not_the_requested_allocation():
    """The `bash` executor must keep ignoring `--mem`, or F8 stops being a class. A script
    declaring 16M and allocating far more still completes, because the ceiling protects the
    machine and says nothing about the artifact."""
    script = (
        "#!/bin/bash\n#SBATCH --mem=16M\n#SBATCH --time=00:15:00\n"
        "chunk=$(head -c 67108864 /dev/zero | tr '\\0' 'x')\n"
        'echo "ALLOCATED=${#chunk}"\necho ANVIL_OK\n'
    )
    task = Task(id="t", prompt="p", constraints={"nodes": 1, "ntasks": 1},
                expects_in_body=["ANVIL_OK", "ALLOCATED="])
    result = check_functional(script, task)
    assert result.passed, result.detail


@pytest.mark.skipif(sandbox_mem_mb() is None,
                    reason="ulimit -v does nothing on this platform, so nothing is capped")
def test_an_unbounded_allocation_is_stopped_by_the_ceiling_and_named_as_such():
    """Pointing the experiment matrix at the execution set killed the host's virtual machine
    before this existed. The detail has to name the ceiling, or the next reader takes a
    machine-protection failure for a verdict on the artifact's resource request."""
    huge = 4 * sandbox_mem_mb() * 1024 * 1024
    script = (
        "#!/bin/bash\n#SBATCH --mem=64G\n#SBATCH --time=00:15:00\n"
        f"chunk=$(head -c {huge} /dev/zero | tr '\\0' 'x')\n"
        "echo ANVIL_OK\n"
    )
    task = Task(id="t", prompt="p", constraints={"nodes": 1, "ntasks": 1},
                expects_in_body=["ANVIL_OK"])
    result = check_functional(script, task)
    assert not result.passed
    assert "sandbox ceiling" in result.detail, result.detail

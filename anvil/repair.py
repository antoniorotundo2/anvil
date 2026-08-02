"""T2: diagnose-and-repair.

Given a broken script (mechanically induced, see inducer.py) and the original
task spec, the model must diagnose the problem and produce a corrected
script. A repair is correct if and only if the repaired script clears the
SAME verifier used to grade a from-scratch T1 solution. This module adds no
separate notion of "close enough".

Two models bracket the task, mirroring `OracleModel`/`BrokenModel` in
models.py:

  * RepairOracleModel  - ignores the broken script and returns the T1
                         canonical solution. Proves every T2 task is
                         solvable and the verifier is not too strict.
  * RepairBrokenModel  - returns the broken script UNCHANGED (a no-op
                         "repair"). Proves the induced faults actually fail
                         verification and repair scoring is not too
                         permissive: a model that changes nothing must score
                         0.0 strict.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .inducer import FAULT_CATEGORIES, induce
from .models import Model
from .schema import RepairTask, Task
from .verifier import verify

REPAIR_SYSTEM_PROMPT = (
    "You are an expert HPC user support engineer. A user submitted the SLURM "
    "batch script below for the job described. It contains a bug: diagnose it, "
    "then output ONLY the corrected script inside one ```bash code block. "
    "No explanation outside the code block."
)


def build_repair_prompt(task: Task, broken_script: str) -> str:
    return (
        f"{task.prompt}\n\n"
        "The following script was submitted for this job. It is broken:\n\n"
        f"```bash\n{broken_script}```\n\n"
        "Diagnose the bug and fix it."
    )


class RepairOracleModel(Model):
    """The T1 canonical solution, regardless of what was broken.

    `build_repair_prompt` always starts the prompt with the base task's
    original NL spec, so (exactly like `OracleModel` for T1) the correct
    reference script is recovered by matching that prefix, with no need to
    thread repair-task ids through `generate`'s prompt-only interface.
    """

    name = "oracle"

    def __init__(self, reference_path: str | Path, t1_tasks: list[Task]):
        by_id: dict[str, str] = {}
        with open(reference_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    by_id[rec["id"]] = rec["script"]
        self._prompt_to_script = {
            task.prompt: by_id.get(task.id, "") for task in t1_tasks
        }

    def generate(self, prompt: str, n: int = 1, seed: int | None = None) -> list[str]:
        script = next(
            (s for p, s in self._prompt_to_script.items() if prompt.startswith(p)), ""
        )
        return [f"```bash\n{script}```" for _ in range(n)]


class RepairBrokenModel(Model):
    """Identity 'repair': returns the broken script unchanged.

    A negative control. If this ever scores above 0.0 strict, `t2_repair.jsonl`
    contains a fault that does not actually fail verification.
    """

    name = "broken"

    def generate(self, prompt: str, n: int = 1, seed: int | None = None) -> list[str]:
        # The broken script is always the last fenced block in the prompt we
        # build ourselves (see build_repair_prompt); recover it verbatim.
        marker = "```bash\n"
        start = prompt.rfind(marker)
        end = prompt.find("```", start + len(marker))
        broken = prompt[start + len(marker):end] if start != -1 and end != -1 else ""
        return [f"```bash\n{broken}```" for _ in range(n)]


def build_repair_model(spec: str, reference_path: str | Path, t1_tasks: list[Task], **kw):
    if spec == "oracle":
        return RepairOracleModel(reference_path, t1_tasks)
    if spec == "broken":
        return RepairBrokenModel()
    from .models import HFModel  # noqa: PLC0415

    return HFModel(spec, **kw)


def verify_repair(
    script: str, repair_task: RepairTask, base_task: Task, run_functional: bool = True
):
    """A repair is graded by the T1 verifier applied to the base task, with
    the result's task_id set to the REPAIR task id so pass@k groups samples
    per induced-fault instance, not per base task."""
    res = verify(script, base_task, run_functional=run_functional)
    res.task_id = repair_task.id
    return res


def induce_t2_tasks(
    t1_tasks: list[Task], reference: dict[str, str], run_functional: bool = True
) -> tuple[list[RepairTask], list[str]]:
    """Build the T2 task set from T1 references, keeping only induced variants
    that actually fail the verifier: `induce()` is a pure string transform
    and cannot guarantee that on its own.

    Returns (repair_tasks, warnings), where the warnings list categories that an
    inducer produced but that turned out to still verify clean, and were
    therefore dropped.
    """
    repair_tasks: list[RepairTask] = []
    warnings: list[str] = []
    for task in t1_tasks:
        good = reference.get(task.id)
        if good is None:
            warnings.append(f"{task.id}: no reference solution, skipped")
            continue
        for category, broken in induce(good, task).items():
            probe = replace(task, id=f"__induce_probe__{task.id}")
            res = verify(broken, probe, run_functional=run_functional)
            if res.all_passed:
                warnings.append(
                    f"{task.id}/{category}: induced variant still verifies clean, dropped"
                )
                continue
            repair_tasks.append(
                RepairTask(
                    id=f"{task.id}__{category}",
                    base_task_id=task.id,
                    fault_category=category,
                    fault_detail=FAULT_CATEGORIES[category],
                    broken_script=broken,
                )
            )
    return repair_tasks, warnings

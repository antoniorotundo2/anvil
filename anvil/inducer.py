"""Fault injection for T2 (diagnose-and-repair).

T2 needs broken scripts with KNOWN ground truth: what is wrong, and what a
correct repair looks like. Hand-writing these does not scale past a handful of
tasks, so this module induces them mechanically from the T1 reference
solutions, anchored to the failure classes observed on a real model
(docs/OBSERVED_FAILURES.md, F1-F8).

F9, an option the scheduler does not have, is observed on a real model and
deliberately not induced: registering it would regenerate tasks/t2_repair.jsonl,
which test_repair.py holds equal to what the inducers produce, and that file is
the denominator of every published T2 number.

Each inducer takes a KNOWN-GOOD script and its Task and returns a broken
variant, or `None` if it does not apply (e.g. F6 needs a derived-value payload
that most tasks do not have). Whether an induced variant actually breaks
verification is NOT decided here: these functions are pure string transforms,
deliberately kept free of subprocess/SLURM dependencies so they stay fast and
hermetic to unit-test. The caller that builds tasks/t2_repair.jsonl is
responsible for running the real verifier and discarding any variant that
does not fail (an inducer that produces an accidentally-valid script is a bug
in the inducer, not a fault worth teaching a model to repair, the same
"broken must mean broken" bracketing DESIGN.md applies to T1's broken model).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .parse import parse_directives
from .schema import Task

FAULT_CATEGORIES: dict[str, str] = {
    "F1": "silent under-request through an omitted default",
    "F2": "directive after the first command (silently ignored by SLURM)",
    "F3": "prose leaking into a directive value",
    "F4": "missing directive with no universal default",
    "F5": "no #SBATCH directive at all",
    "F6": "payload/spec mismatch",
    "F7": "malformed directive value rejected by the scheduler",
    "F8": "memory request below what the payload actually uses",
}


def _line_declares(line: str, directive: str) -> bool:
    """Does this single line carry the given #SBATCH directive?"""
    if "#SBATCH" not in line:
        return False
    return directive in parse_directives(line)


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _drop_directive(script: str, directive: str) -> str | None:
    lines = script.splitlines(keepends=True)
    kept = [ln for ln in lines if not _line_declares(ln, directive)]
    if len(kept) == len(lines):
        return None
    return "".join(kept)


# --------------------------------------------------------------------------
# F1 - silent under-request through an omitted default
# --------------------------------------------------------------------------
def inject_f1_silent_underrequest(script: str, task: Task) -> str | None:
    """Drop --cpus-per-task or --ntasks: SLURM's default masks the drop.

    The scheduler still accepts the job and it still runs, at a fraction of
    the requested parallelism. Only `resource_fit` catches this.
    """
    c = task.constraints
    for key, directive in (("cpus_per_task", "--cpus-per-task"), ("ntasks", "--ntasks")):
        if key not in c:
            continue
        out = _drop_directive(script, directive)
        if out is not None:
            return out
    return None


# --------------------------------------------------------------------------
# F2 - directive after the first command
# --------------------------------------------------------------------------
def inject_f2_misplaced_directive(script: str, task: Task) -> str | None:
    """Move the last #SBATCH directive to after the first real command.

    SLURM stops reading directives at the first command; the moved directive
    is silently ignored. `sbatch --test-only` still accepts the job.
    """
    lines = script.splitlines()
    directive_lines = [ln for ln in lines if ln.strip().startswith("#SBATCH")]
    if len(directive_lines) < 2:
        return None  # must leave at least one directive before the first command

    victim = directive_lines[-1]
    remaining: list[str] = []
    removed = False
    for ln in lines:
        if not removed and ln.strip().startswith("#SBATCH") and ln == victim:
            removed = True
            continue
        remaining.append(ln)

    insertion = None
    for i, ln in enumerate(remaining):
        s = ln.strip()
        if s and not s.startswith("#"):
            insertion = i + 1
            break
    if insertion is None:
        return None

    new_lines = remaining[:insertion] + [victim] + remaining[insertion:]
    return "\n".join(new_lines) + ("\n" if script.endswith("\n") else "")


# --------------------------------------------------------------------------
# F3 - prose leaking into a directive value
# --------------------------------------------------------------------------
def inject_f3_prose_leak(script: str, task: Task) -> str | None:
    """Replace a --mem value with a degenerate bare number plus prose.

    Mirrors the observed artifact `#SBATCH --mem=2 referencing GB`: it parses
    cleanly (the prose is a stray token) but the number is far below the
    minimum requested.
    """
    if "mem_min_mb" not in task.constraints:
        return None
    lines = script.splitlines()
    out: list[str] = []
    changed = False
    for ln in lines:
        if not changed and _line_declares(ln, "--mem"):
            out.append(f"{_leading_ws(ln)}#SBATCH --mem=2 referencing the requested memory")
            changed = True
            continue
        out.append(ln)
    if not changed:
        return None
    return "\n".join(out) + ("\n" if script.endswith("\n") else "")


# --------------------------------------------------------------------------
# F4 - missing directive with no universal default
# --------------------------------------------------------------------------
def inject_f4_missing_no_default(script: str, task: Task) -> str | None:
    """Drop --time, --mem or --gpus: unlike --nodes/--ntasks/--cpus-per-task,
    none of these has a SLURM-wide default. Omitting them means the resource
    was never requested."""
    c = task.constraints
    candidates = []
    if "time_max_minutes" in c:
        candidates.append("--time")
    if "mem_min_mb" in c:
        candidates.append("--mem")
    if "gpus_min" in c:
        candidates.append("--gpus")
    for directive in candidates:
        out = _drop_directive(script, directive)
        if out is not None:
            return out
    return None


# --------------------------------------------------------------------------
# F5 - no #SBATCH directive at all
# --------------------------------------------------------------------------
def inject_f5_no_sbatch(script: str, task: Task) -> str | None:
    """Strip every #SBATCH line: the artifact stops being a job script."""
    lines = script.splitlines(keepends=True)
    kept = [ln for ln in lines if not ln.strip().startswith("#SBATCH")]
    if len(kept) == len(lines):
        return None
    return "".join(kept)


# --------------------------------------------------------------------------
# F6 - payload/spec mismatch
# --------------------------------------------------------------------------
_EXPORT_DEFAULT = re.compile(r"(export\s+\w+=)\$\{[A-Z_]+:-(\d+)\}")


def inject_f6_payload_mismatch(script: str, task: Task) -> str | None:
    """Hardcode a wrong constant over a value that should derive from SLURM's
    environment, e.g. `export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}`
    becomes `export OMP_NUM_THREADS=5`. The printed payload no longer matches
    the spec, and no longer reacts to the actual allocation."""
    m = _EXPORT_DEFAULT.search(script)
    if not m:
        return None
    wrong = int(m.group(2)) + 1
    return _EXPORT_DEFAULT.sub(lambda mm: f"{mm.group(1)}{wrong}", script, count=1)


# --------------------------------------------------------------------------
# F7 - malformed directive value rejected by the scheduler
# --------------------------------------------------------------------------
def inject_f7_malformed_value(script: str, task: Task) -> str | None:
    """Corrupt --time into a value the scheduler itself rejects."""
    lines = script.splitlines()
    out: list[str] = []
    changed = False
    for ln in lines:
        if not changed and _line_declares(ln, "--time"):
            out.append(f"{_leading_ws(ln)}#SBATCH --time=aa:bb")
            changed = True
            continue
        out.append(ln)
    if not changed:
        return None
    return "\n".join(out) + ("\n" if script.endswith("\n") else "")


# --------------------------------------------------------------------------
# F8 - memory request below what the payload uses
# --------------------------------------------------------------------------
def inject_f8_memory_underrequest(script: str, task: Task) -> str | None:
    """Cut `--mem` to a value the payload cannot fit in.

    The first fault class no static check can see and `bash` cannot either: the value
    stays well formed, the scheduler accepts it, and the script runs to completion
    everywhere except where the allocation is enforced, which is the sbatch executor on a
    cluster with cgroup constraints. There the job comes back OUT_OF_MEMORY.

    Applies only where the spec does not pin the memory. A task that states a minimum
    already has this covered statically: cutting the value below it fails `resource_fit`,
    which is F3's and F4's territory. The interesting case is the one where the payload's
    real need is the only ground truth, and no check that reads the text can reach it.
    """
    if "mem_min_mb" in task.constraints:
        return None
    lines = script.splitlines()
    out: list[str] = []
    changed = False
    for ln in lines:
        if not changed and _line_declares(ln, "--mem"):
            out.append(f"{_leading_ws(ln)}#SBATCH --mem=16M")
            changed = True
            continue
        out.append(ln)
    if not changed:
        return None
    return "\n".join(out) + ("\n" if script.endswith("\n") else "")


# Fault classes that no static check and no sandbox can decide, so a guard run without an
# enforcing executor cannot tell a permissive verifier from an environment that simply
# cannot see the fault. F8 is the one: the memory value stays well formed, the scheduler
# accepts it, the script completes under bash, and only an enforced allocation refuses the
# job. A no-op repair of an F8 therefore passes every level under bash *by construction*,
# which is the property the class exists to demonstrate and not a hole in the verifier.
NEEDS_ENFORCEMENT = frozenset({"F8"})


def decidable(category: str, executor: str) -> bool:
    """Can this environment decide whether a repair of this fault succeeded?

    Called by the guards rather than by the verifier: a level the executor cannot judge is
    already reported as skipped, and skipped is never passed. What this answers is the
    different question of whether a *guard* may draw a conclusion from the result.
    """
    return executor == "sbatch" or category not in NEEDS_ENFORCEMENT


INDUCERS: dict[str, Callable[[str, Task], str | None]] = {
    "F1": inject_f1_silent_underrequest,
    "F2": inject_f2_misplaced_directive,
    "F3": inject_f3_prose_leak,
    "F4": inject_f4_missing_no_default,
    "F5": inject_f5_no_sbatch,
    "F6": inject_f6_payload_mismatch,
    "F7": inject_f7_malformed_value,
    "F8": inject_f8_memory_underrequest,
}


def induce(script: str, task: Task) -> dict[str, str]:
    """Return {fault_category: broken_script} for every inducer applicable to
    this task. Categories whose inducer returns None (or a no-op) are absent:
    not every task exhibits every fault class."""
    out: dict[str, str] = {}
    for category, fn in INDUCERS.items():
        broken = fn(script, task)
        if broken is not None and broken != script:
            out[category] = broken
    return out

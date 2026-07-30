# The reference cluster

`submittability` validates artifacts against a **declared** topology, not against the hardware
of whichever machine runs the benchmark.

## Why

`sbatch --test-only` compares the requested resources with the node configuration. If that
configuration were inherited from the host, a script asking for two GPU nodes would be rejected
on a laptop and accepted on a server: **scores would depend on who runs the benchmark**.

## The topology (v1)

| | |
|---|---|
| Nodes | 4 (`node1`–`node4`) |
| Cores per node | 16 |
| Memory per node | 64000 MB |
| GPUs per node | 4 |
| Partition | `debug`, no time limit |
| `FirstJobId` | 12345 |

Overridable via `ANVIL_NODES`, `ANVIL_CPUS`, `ANVIL_MEM_MB`, `ANVIL_GPUS`.

**Changing the topology changes the results.** It is part of the benchmark definition, like the
tasks: version it and cite it in the paper.

## Non-obvious details, learned empirically

- **`SlurmdParameters=config_overrides`** — slurmctld trusts the configuration instead of
  interrogating the hardware. Without it, a single-core container rejects a job asking for eight.
- **`RealMemory` is in MB.** A task requesting `--mem=16G` (16384 MB) does **not** fit a node
  declared with 16000. The first reference cluster was too tight and rejected `t1_gpu_single`:
  the canonical solution was correct, the cluster was not.
- **`FirstJobId=12345` plus a held placeholder job.** Tasks declaring
  `--dependency=afterok:12345` otherwise fail with *"Job dependency problem"* — the job does not
  exist. The placeholder deterministically takes that id.
- **`cgroup.conf` with `IgnoreSystemd=yes`.** Containers have no systemd: without it slurmd dies
  on *"can't stat /sys/fs/cgroup/systemd/"*, never registers, nodes stay `idle*`, and
  `sbatch --test-only` rejects **every** script.
- **Declaring nodes `State=IDLE` and skipping slurmd is not enough.** It works for a few seconds,
  then the *not responding* flag appears and allocations start failing. A benchmark that works for
  thirty seconds and then silently rots is worse than one that never works.

## The preflight (the defence that matters)

Before scoring anything, the verifier submits a **canary** to the scheduler: a minimal,
certainly-valid script. If the canary fails, `submittability` is marked **skipped**, not *failed*.

Without this, a misconfigured cluster produces eight zeros that are **indistinguishable from a
terrible model**. It happened during development: the canonical solutions were correct and the
harness was broken. A benchmark that executes code must be able to tell a failure of its subject
from a failure of itself.

## Two executors for `functional`

The level has two executors, and the environment report declares which one produced a given
number in `functional_executor`.

`bash` (the default) runs the script in a temporary sandbox. It ignores every `#SBATCH` line and
injects `SLURM_NTASKS`, `SLURM_CPUS_PER_TASK` and `SLURM_NNODES` derived from the task
constraints, so it needs no scheduler at all and every published number was measured with it.

`sbatch` (`--executor sbatch`, or `ANVIL_FUNCTIONAL_EXECUTOR=sbatch`) submits the script for real
with `--chdir` pointed at the sandbox, polls `scontrol` until the job reaches a terminal state, and
reads the output from the files the script's own `--output`/`--error` name. It needs a scheduler
that actually runs jobs, not merely one that accepts them, which is what its own canary checks
before anything is graded. `sacct` is deliberately not used: it needs accounting storage, which
this cluster does not configure ("Slurm accounting storage is disabled").

Three details of the reference cluster show up only under real submission:

- **A held placeholder job is not a satisfiable dependency.** The placeholder that gives
  `--dependency=afterok:12345` something to point at is held, so it never completes. Under
  `--test-only` the script is accepted; submitted for real the job stays PENDING with
  `Reason=Dependency` forever, so `functional` is **skipped** for `t1_dependency_chain`, not failed.
- **`--output` directories must exist before submission.** slurmstepd opens that file before the
  script's first command, so a `mkdir -p logs` inside the script comes too late. The harness creates
  them.
- **`MinJobAge` bounds how long the record survives.** `scontrol` forgets a finished job after it,
  and a record that vanishes before the poll reads it is skipped rather than guessed at.

`make guards-sbatch` is the bracket for this executor: no oracle sample may *fail* `functional`
under real submission, at least one must actually have run, and the broken model must still score
0.0 strict.

## What still is not observed (Phase 3, stage 2)

OOM kills and CPU/GPU binding. Both need cgroup enforcement, which the reference cluster does not
configure, and enabling it collides with the declared topology: a job asking for 64000 MB on a
4-node reference cluster cannot be held to the memory of whatever machine runs the benchmark
without making the score a property of that machine again. That tension is the actual open design
question, not a missing line of configuration.

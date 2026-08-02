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

### One partition, and what that measures

The cluster declares a single partition, `debug`, and it stays that way. A script naming any other
is rejected at submit, and models do it often: in the multi-seed run 106 samples of 1560 came back
`invalid partition specified`, naming `gpu`, `small`, and in several cases the placeholder
`your_partition_name` left in from a template.

Declaring a `gpu` partition would accept most of those and would look more like a real centre. It
is left undeclared deliberately. No task asks for a partition and no reference solution sets one,
so a script that volunteers a name is assuming something about a cluster it has never seen, and on
a real system that assumption is precisely what gets the job rejected. One partition makes the
benchmark measure the assumption instead of forgiving it.

## Non-obvious details, learned empirically

- **`SlurmdParameters=config_overrides`**: slurmctld trusts the configuration instead of
  interrogating the hardware. Without it, a single-core container rejects a job asking for eight.
- **`RealMemory` is in MB.** A task requesting `--mem=16G` (16384 MB) does **not** fit a node
  declared with 16000. The first reference cluster was too tight and rejected `t1_gpu_single`:
  the canonical solution was correct, the cluster was not.
- **`FirstJobId=12345` plus a held placeholder job.** Tasks declaring
  `--dependency=afterok:12345` otherwise fail with *"Job dependency problem"*: the job does not
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
0.0 strict. `make docker-guards-sbatch SCHED_IMAGE=<image>` runs it in a container, which needs
`--privileged --cgroupns=host` (see below) and an image whose scheduler can execute.

## Making the container execute, and where it stops

Nothing in the image had ever run a job: `sbatch --test-only` needs no `slurmd`, so no `slurmd` had
to work, and four separate faults sat there undisturbed until the day one was asked to. In order:

- **The cgroup slice above the stepd scope does not exist.** slurmd creates
  `<base>/system.slice/nodeN_slurmstepd.scope` but not the `system.slice` above it, and under
  Docker that parent is missing, so cgroup/v2 fails to initialise and the daemon exits. The
  entrypoint now creates it, which needs a writable `/sys/fs/cgroup`, hence `--privileged
  --cgroupns=host` on the container. `sbatch --test-only` needs neither.
- **Multi-slurmd needs one port per virtual node.** All four were declared with the range form and
  no `Port=`, so three died on "Address already in use" and one survived.
- **A registering slurmd reports the GPUs it can see.** With none, the controller answers
  "gres/gpu count reported lower than configured (0 < 4)" and drains the node. The entrypoint
  backs the declared GPUs with device files, and falls back to the count-only form where it
  cannot create them.
- **Ubuntu 24.04's SLURM refuses every job it accepts.** With those three fixed, all four nodes
  register and stay IDLE, and every job still sits at `PENDING` with `Reason=InvalidAccount`. It is
  not this configuration: a minimal stock `slurm.conf` in the same image behaves identically. The
  package (23.11.4-1.2ubuntu5) ships `accounting_storage_slurmdbd.so` and nothing else, so with no
  `slurmdbd` the association manager has no entries and the scheduler rejects each job. The same
  base image with Ubuntu 26.04 (SLURM 25.11.2) runs the same job to `COMPLETED`, so it is this
  package rather than SLURM.

## The accounting image

Switching the base to 26.04 would trade a scheduler that cannot execute for coreutils that are not
GNU, which is the one thing this image exists to avoid. So accounting is added instead, opt-in for
the same reason apptainer is, and on the same 24.04 base:

```
make docker-build-sched          # docker build --build-arg WITH_SLURMDBD=1
make docker-guards-sbatch        # the strict bracket, submitted for real
```

It adds `slurmdbd` and a local MariaDB, and the entrypoint brings up the database, writes
`slurmdbd.conf` on port 6899 (6819 would collide with the virtual nodes), and registers the
cluster, one account and the submitting user before slurmctld starts. `AccountingStorageEnforce`
stays unset: the association manager needs something to find, not a policy to apply, and limits
would make a job's fate a property of this file.

Where it stands: the oracle scores `strict_all_levels` 1.0 with seven of eight tasks executed for
real, `t1_dependency_chain` skipped for the held placeholder it depends on, and the broken model
0.0 strict.

## Enforcement, and why it does not import the host

With the cgroup controllers delegated, `TaskPlugin=task/cgroup` holds every job to what it asked
for: `ConstrainRAMSpace`, `ConstrainSwapSpace` and `ConstrainCores`. A script that requests less
memory than its payload uses now comes back `OUT_OF_MEMORY` with `ExitCode=0:125` instead of
completing. Swap has to be constrained alongside RAM, and that is not a detail: with `memory.max`
alone the pages over the limit are simply paged out and the job finishes as if nothing had
happened, which is what the first measurement showed.

The tension with the declared topology is narrower than it looks. A cgroup limit is a ceiling, not
a reservation: a job that requests 64000 MB and touches 10 MB runs on a machine with 8 GB, because
nothing is set aside. What the limit compares is the job's own request against the job's own usage,
and both travel with the artifact. The host only enters if a task's payload really allocates, which
is why the one task that does asks for tens of megabytes and not gigabytes.

Enforcement is detected, never assumed. Without a writable `/sys/fs/cgroup` the delegation fails,
and the configuration falls back to the plugins that need none: the entrypoint prints which of the
two is in force, since the same script scores differently under each.

## What still is not observed

CPU and GPU binding. `ConstrainCores` is on, so a job is confined to the cores it was allocated,
but no task in the set reads its own affinity, and the GPUs are device files with nothing behind
them, so a task that checked which one it was given would be checking the harness. Both are
task-set work rather than configuration.

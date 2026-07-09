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

## Known limitation (Phase 2)

The `functional` level executes the script with **bash in a temporary sandbox**; it does not
submit it to `sbatch`. This is declared in the environment report as `functional_executor: "bash"`.

Consequences not to be hidden in the paper:
- OOM kills, walltime overruns and CPU/GPU binding are not observed — precisely the failure modes
  most interesting for the T2 repair task;
- the execution environment lacks the runtime variables SLURM injects beyond those simulated from
  the task constraints.

Phase 2: real `functional` via `sbatch`, waiting for completion and reading the exit code from
`sacct`. This unlocks the induced-failure taxonomy (OOM, walltime) on the reference cluster.

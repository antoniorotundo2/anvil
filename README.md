# Anvil

The project aims to measure whether the operational artifacts an LLM writes for a supercomputer —
SLURM job scripts, and later container recipes — are actually correct: verified by submission,
execution and resource fit, not by textual similarity. Beyond writing scripts from scratch (T1),
Anvil also measures whether a model can **diagnose and repair** a broken one (T2): see
[Diagnose-and-repair (T2)](#diagnose-and-repair-t2) below.

## Requirements

### Hardware

- Any machine, to develop and to verify
- [Optional] An NVIDIA GPU or Apple Silicon, to generate scripts with a real model
- [Optional] A machine with SLURM installed, if you do not want to use Docker

### Software

- Python >= 3.10
- Docker Community Edition (recommended: it provides SLURM and GNU coreutils)
- [Optional] PyTorch and Transformers, only to generate with a model
- [Optional] `bitsandbytes`, for 4-bit quantization on CUDA

## Configuration

The benchmark validates artifacts against a **declared** reference cluster, not against the
hardware of the machine that runs it. Otherwise scores would depend on who runs the benchmark.
The topology is defined by environment variables read by the container:

```
ANVIL_NODES=4        # virtual nodes
ANVIL_CPUS=16        # cores per node
ANVIL_MEM_MB=64000   # memory per node, in MB
ANVIL_GPUS=4         # GPUs per node
```

Changing the topology changes the results: it is part of the benchmark definition. See
[`docs/REFERENCE_CLUSTER.md`](docs/REFERENCE_CLUSTER.md).

Tasks live in `tasks/t1_slurm.jsonl`; their canonical solutions in `tasks/t1_reference.jsonl`.
Induced repair tasks (T2) live in `tasks/t2_repair.jsonl` — see
[Diagnose-and-repair (T2)](#diagnose-and-repair-t2).

## Install

```
make install
```

To generate scripts with a real model, also install the model extras:

```
make install-models
```

## Run

To run the project you can use two methods. The first one (recommended) uses Docker, which ships a
SLURM reference cluster and GNU coreutils, so every verification level is active. The second one is
the manual way, on your own machine, where `submittability` is skipped unless you install SLURM and
where `functional` runs under whatever `bash` and coreutils you happen to have.

### Docker (Recommended)

```
make docker-run
```

Verify what the environment can actually check:

```
make doctor
```

### Manually

```
make run
```

### Guards

The oracle returns canonical solutions and must score 1.0; the broken model returns deliberately
faulty artifacts and must score 0.0. If the oracle drops, the benchmark is broken — not the model.

```
make guards
```

## Development

### Tests

```
make test
make lint
```

### Generate here, verify there

Generation needs the machine with the accelerator. Faithful verification needs the machine with the
scheduler and GNU coreutils. They are rarely the same machine.

```
make generate MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
make docker-verify
```

`verify` records `bash`, `coreutils`, `base_image` and `functional_executor` in its JSON output, so
every number carries the environment that produced it. This also enables the cross-distribution
ablation: generate once, verify against several base images.

### Base image

The container defaults to `ubuntu:24.04`, not the newest LTS: Ubuntu 26.04 replaces GNU coreutils
with `uutils`, which no HPC centre runs. Fidelity beats freshness.

```
docker build -t anvil:2604 --build-arg BASE_IMAGE=ubuntu:26.04 docker/
```

### Adding tasks

A task declares a natural-language prompt, the resource constraints to check, the directives that
must be written out explicitly, and the strings its output must contain. Every new task needs a
canonical solution in `tasks/t1_reference.jsonl`, or `make guards` will fail.

## Diagnose-and-repair (T2)

T1 asks a model to write a script from scratch. T2 hands it a **broken** one — one of seven fault
classes anchored to failures observed on a real model (`docs/OBSERVED_FAILURES.md`, F1–F7) — and
asks it to diagnose and fix it. A repair is correct if and only if it clears the exact same
verifier used to grade a from-scratch T1 solution: repair is not a softer notion of correctness.

`tasks/t2_repair.jsonl` is not hand-written: it is induced mechanically from the T1 canonical
solutions (`anvil/inducer.py`), and only variants that actually fail verification are kept.
Rebuild it after changing a T1 task or its reference solution:

```
make induce-t2
```

Run a model against it and verify, same shape as T1:

```
make repair MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
```

```
anvil repair --model oracle --repair-tasks tasks/t2_repair.jsonl --tasks tasks/t1_slurm.jsonl -v
anvil repair --model <hf-model-id> --repair-tasks tasks/t2_repair.jsonl --save-generations results/repair_generations.jsonl
anvil verify-repair --generations results/repair_generations.jsonl --repair-tasks tasks/t2_repair.jsonl -v
```

### Guards

The oracle repair (returns the T1 canonical solution, ignoring the diagnosis) must pass every
induced fault; a no-op "repair" that returns the broken script unchanged must pass none. If either
fails, `t2_repair.jsonl` or the repair verifier is broken — not a model.

```
make guards-t2
```

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — why execution-based verification, the five levels, the preflight
- [`docs/REFERENCE_CLUSTER.md`](docs/REFERENCE_CLUSTER.md) — the declared topology and its non-obvious details
- [`docs/OBSERVED_FAILURES.md`](docs/OBSERVED_FAILURES.md) — failure classes observed on a real model
- [`docs/HARDWARE.md`](docs/HARDWARE.md) — development machine vs experiment machine

## License

MIT.

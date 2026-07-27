<h1 align="center"><a id="anvil"></a><img src="docs/assets/logo.svg" alt="Anvil" width="420"></h1>

The project aims to measure whether the operational artifacts an LLM writes for a supercomputer —
SLURM job scripts and Apptainer container recipes — are actually correct: verified by submission,
execution and resource fit, not by textual similarity. Beyond writing scripts from scratch (T1),
Anvil also measures whether a model can **diagnose and repair** a broken one (T2, see
[Diagnose-and-repair (T2)](#diagnose-and-repair-t2)) and whether it can write a correct
[Apptainer recipe (T3)](#apptainer-recipes-t3).

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

Both commands break the summary down **per fault category** (F1–F7) in addition to the overall
one, on screen and under `"by_category"` in `--out`'s JSON: an aggregate pass@k can hide a category
a model never manages to repair.

### Guards

The oracle repair (returns the T1 canonical solution, ignoring the diagnosis) must pass every
induced fault; a no-op "repair" that returns the broken script unchanged must pass none. If either
fails, `t2_repair.jsonl` or the repair verifier is broken — not a model.

```
make guards-t2
```

## Apptainer recipes (T3)

A third artifact type: a model writes an Apptainer definition file (`.def`) instead of a SLURM
script. Same shape as T1, different vocabulary: `syntax`, `buildable` (does `apptainer build`
succeed), `functional`, `resource_fit` (header and section set against the spec), `safety`.

Apptainer is opt-in and not part of the default image, since most anvil work never touches it:

```
make docker-build-apptainer
```

Run a model against `tasks/t3_apptainer.jsonl` and verify:

```
make recipe MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
```

```
anvil recipe --model oracle --tasks tasks/t3_apptainer.jsonl -v
anvil recipe --model <hf-model-id> --tasks tasks/t3_apptainer.jsonl --save-generations results/recipe_generations.jsonl
anvil verify-recipe --generations results/recipe_generations.jsonl --tasks tasks/t3_apptainer.jsonl -v
```

Apptainer runs unprivileged inside the container, so no capability is granted: what it
needs is exemptions from Docker's confinement. `make docker-guards-t3` applies them and
needs no argument. The same set works on every verified host, so there is nothing to
select per environment (see the Makefile's `DOCKER_RUN_APPTAINER` and `docs/DESIGN.md`
for what each one unlocks):

```
docker run --rm --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --security-opt systempaths=unconfined --device /dev/fuse \
    -v "$PWD":/work -w /work anvil:apptainer ...
```

`--privileged` also works but grants far more than these actually need. Verified on
GitHub-hosted runners and on WSL2, where the strict bracket returns identical per-level
scores. On Docker Desktop for Mac, `build` succeeded but `run` failed
(`exec ... failed: invalid argument`), untested since the AppArmor findings.

### Guards

`buildable` and `functional` both need a real `apptainer` binary, much less commonly available
than the `bash` that T1/T2's `functional` relies on. Without it, both are skipped, not failed:

```
make guards-t3
```

checks only what `syntax`/`resource_fit`/`safety` can prove. The full oracle-1.0/broken-0.0
bracket needs the opt-in image:

```
make docker-guards-t3
```

## Retrieval ablation

Three ways to prompt a model for T1: `zero-shot` (the default, no change from the rest of this
README), `vector` (TF-IDF similarity against `tasks/retrieval_corpus.jsonl`), `vectorless` (exact
tag match, no scoring). `anvil run --retrieval` selects the arm:

```
anvil run --model oracle --tasks tasks/t1_slurm.jsonl --retrieval vector -v
anvil run --model <hf-model-id> --tasks tasks/t1_slurm.jsonl --retrieval vectorless
```

`OracleModel` still recognises the task regardless of which arm is active (it matches on
`prompt.startswith(task.prompt)`, since retrieved context is always appended after the original
prompt, never before it), so `make guards` stays valid for any `--retrieval` value.

Compare all three arms on the same model, seeds and tasks:

```
./scripts/retrieval_ablation.sh
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct SEEDS="0 1 2" N=5 ./scripts/retrieval_ablation.sh
```

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — why execution-based verification, the five levels, the preflight
- [`docs/REFERENCE_CLUSTER.md`](docs/REFERENCE_CLUSTER.md) — the declared topology and its non-obvious details
- [`docs/OBSERVED_FAILURES.md`](docs/OBSERVED_FAILURES.md) — failure classes observed on a real model
- [`docs/HARDWARE.md`](docs/HARDWARE.md) — development machine vs experiment machine

## License

MIT.

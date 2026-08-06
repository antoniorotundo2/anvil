<h1 align="center"><a id="anvil"></a><img src="docs/assets/logo.svg" alt="Anvil" width="420"></h1>

The project aims to measure whether the operational artifacts an LLM writes for a supercomputer
(SLURM job scripts and Apptainer container recipes) are actually correct: verified by submission,
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
Induced repair tasks (T2) live in `tasks/t2_repair.jsonl`, see
[Diagnose-and-repair (T2)](#diagnose-and-repair-t2).

## Install

To work on the benchmark, from a checkout:

```
make install
```

To generate scripts with a real model, also install the model extras:

```
make install-models
```

To use the verifier without a checkout, install the package:

```
pip install git+https://github.com/antoniorotundo2/anvil
```

That gives you the `anvil` command with no dependencies beyond the standard library. The task
files travel with it, so `anvil check job.sh --task t1_gpu_single` works from any directory. A
checkout still reads `tasks/` from the working directory, which is where every published number
was measured, and the packaged copy is only consulted when that path does not exist.

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
faulty artifacts and must score 0.0. If the oracle drops, the benchmark is broken, not the model.

```
make guards
```

### Real submission

`functional` executes the script with `bash` in a sandbox by default. `--executor sbatch` submits
it to the scheduler for real instead, waits for the job and reads its outcome from `scontrol`, so
the walltime the script requested is enforced and the payload sees every variable SLURM injects:

```
anvil run --model oracle --tasks tasks/t1_slurm.jsonl --executor sbatch -v
ANVIL_FUNCTIONAL_EXECUTOR=sbatch anvil verify --generations results/generations.jsonl
```

It is opt-in on purpose. Every published number was measured under `bash`, and switching the
default would make later ones incomparable with them; the executor is recorded in each result
file as `functional_executor`. It also needs a scheduler that genuinely *runs* jobs, not one that
merely accepts them: its own canary checks that first, and `functional` is skipped, never failed,
when the check does not hold. Same for a job the scheduler can never place, such as the dependency
task pointing at the held placeholder job.

```
make guards-sbatch
make docker-guards-sbatch
```

The container form builds its own image: the default one accepts jobs and never runs them, since
Ubuntu 24.04's SLURM has no accounting plugin other than `slurmdbd` and refuses each job with
`Reason=InvalidAccount`. `make docker-build-sched` adds `slurmdbd` and a local database, opt-in
like the apptainer image and on the same base, so the coreutils stay GNU. The run also needs
`--privileged --cgroupns=host`, which the Makefile supplies: `slurmd` creates its cgroup scope
under `/sys/fs/cgroup`, which a plain `docker run` mounts read-only. See
[`docs/REFERENCE_CLUSTER.md`](docs/REFERENCE_CLUSTER.md).

### What only execution can catch

With `slurmd` running, the container also enforces the allocation through cgroups, so a job that
uses more memory than it requested is killed instead of finishing. Nothing in the eight T1 tasks
allocates enough to notice, so `tasks/t1_exec.jsonl` adds one that holds 64MB and asks for enough
memory to fit it, with no number in the specification: the payload's real need is the ground truth.
The fault induced from it (F8, `--mem` cut to 16M) is well formed, within spec, accepted by the
scheduler, and passes under `bash`. It fails only here:

```
make docker-guards-enforcement
```

The set lives in its own file so that adding it changes no digest: `tasks/t1_slurm.jsonl` and
`tasks/t2_repair.jsonl` are untouched, and every published number stays comparable.

CPU and GPU binding remain outside this level: a job is confined to its cores, but no task asks
what it was given.

## Checking a script you already have

Everything above measures a model. If instead you have a job script that an assistant wrote and
you want to know whether it will hold up, `anvil check` answers that with no task file, no model
and no benchmark run:

```
anvil check job.sh
```

`syntax`, `safety` and `submittability` need nothing but the script and, for the last one, a
scheduler to ask. `resource_fit` and `functional` compare against a spec, so without one they are
reported as not checked rather than passed. Give the script a task to be graded against and all
five run:

```
anvil check job.sh --task t1_mpi_multinode
```

The exit code is 0 when every level that ran is satisfied and 1 otherwise, which is what makes it
usable from a pre-submission hook or a CI step. `--json` prints the same verdict for a machine.

### Against a site policy

A cluster has rules that no task file knows: how long a job may run, how many nodes it may take,
which partitions exist, which directives are required. `--policy` checks a script against them:

```
anvil check job.sh --policy policies/reference_cluster.json
```

```
  policy           FAIL   anvil reference cluster
                          partition 'gpu' is not one of normal
                          nodes 9 exceeds the site maximum 4
                          --time 2880min exceeds the site maximum 1440min
```

The comparison runs the opposite way from `resource_fit`, which is the distinction worth keeping
straight: a task fails a script that asks for too little, a policy fails one that asks for too
much. Ceilings apply to the *effective* request, so a script with no `--nodes` is judged as asking
for one node and one with no `--ntasks` as asking for one task per node. A missing `--time` is a
violation rather than a pass, because SLURM would apply a partition limit the file does not state
and the site cannot conclude the job fits.

Every field is optional and an absent field is not a rule. A field that is not recognised is an
error rather than a silence: a misspelled `max_mem_gb` would otherwise read as a site with no
memory limit at all. See [`policies/reference_cluster.json`](policies/reference_cluster.json).

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

T1 asks a model to write a script from scratch. T2 hands it a **broken** one, from one of seven
fault classes anchored to failures observed on a real model (`docs/OBSERVED_FAILURES.md`, F1–F7),
and asks it to diagnose and fix it. A repair is correct if and only if it clears the exact same
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
fails, `t2_repair.jsonl` or the repair verifier is broken, not a model.

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

## Cross-distribution ablation

Verify one set of generations inside several base images and report where the verdicts
diverge, per sample and per level. It reads the `*.generations.jsonl` files that
`scripts/run_experiments.sh` saves beside every cell, so it inherits that run's seeds
without spending inference time again:

```
./scripts/crossdist_ablation.sh results/<run>
BASES="ubuntu:24.04 ubuntu:26.04" ./scripts/crossdist_ablation.sh results/<run>
```

## Executor ablation

Same shape, with the executor as the varying factor instead of the base image: it verifies each
cell twice, under `bash` and under real submission, inside the one image so nothing else changes.

```
make docker-build-sched
./scripts/executor_ablation.sh results/<run>
```

The number it reports is not either pass@k but the disagreement: how many scripts the sandbox
promotes and the scheduler stops, grouped by what stopped them (`OUT_OF_MEMORY`, `TIMEOUT`, a
refused submission, missing output). It also counts the samples real submission cannot judge, such
as the dependency task waiting on a job that never completes, which are skipped rather than
charged to the model.

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md): why execution-based verification, the five levels, the preflight
- [`docs/REFERENCE_CLUSTER.md`](docs/REFERENCE_CLUSTER.md): the declared topology and its non-obvious details
- [`docs/RESULTS.md`](docs/RESULTS.md): every measured number, with what it is and what it is not
- [`docs/DATASET.md`](docs/DATASET.md): the task files, their schemas, and how to tell which copy you have
- [`docs/OBSERVED_FAILURES.md`](docs/OBSERVED_FAILURES.md): failure classes observed on a real model
- [`docs/HARDWARE.md`](docs/HARDWARE.md): development machine vs experiment machine

## Sponsorship

Anvil is developed on one desktop with a single consumer GPU, which is what bounds the
measurements: two model families at two sizes, three seeds, and a roadmap item that reads "more
families, on borrowed hardware". Sponsorship goes to compute, so those arms stop being roadmap
items. [Sponsor this project](https://github.com/sponsors/antoniorotundo2).

## License

MIT.

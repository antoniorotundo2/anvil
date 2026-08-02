# Observed failure classes

Not invented: **observed**, on the first run of a real model against the T1 tasks.
This is the empirical seed of the T2 (diagnose-and-repair) taxonomy, and the answer to the
reviewer question *"are your induced failures representative?"*

Generated with `Qwen/Qwen2.5-Coder-1.5B-Instruct`, fp16 on MPS, n=1, seed=0, 8 tasks.
Verified inside the container (Ubuntu 24.04, bash 5.2, GNU coreutils, live `slurmctld`).
`strict_all_levels` pass@1 = 0.25; `submittability` = 0.625.
**Single sample, single seed: an observation, not a result.** Numbers predate the parser fix below.

---

## F1: Silent under-request through an omitted default
**The headline failure.** On a task asking for 2 nodes and 4 tasks, the model wrote `--nodes=2`
and omitted `--ntasks`. SLURM defaults to one task per node: the job requests **2 tasks, not 4**.

```
ntasks expected 4, effective 2 (SLURM default, not declared)
```

The script is valid. `sbatch --test-only` accepts it. It runs. It uses **half the requested
parallelism**, and nobody is told. No textual-similarity metric detects this; no dry-run rejects
it. The user discovers it when the job takes twice as long.

This is the thesis of the benchmark in a single line: *a wrong resource request does not look
wrong.*

## F2: Directives after the first command
On the I/O task the model emitted `mkdir -p logs` and *then* four `#SBATCH` lines. SLURM stops
reading directives at the first command: job name, walltime and output paths are **silently
ignored**. `sbatch` accepts the job.

Caught by `check_syntax`, not by `sbatch --test-only`. This is the check that justifies going
beyond dry-run validation.

## F3: Prose leaking into directive values
```
#SBATCH --mem=2 referencing GB
#SBATCH --time=20:00: referencing the time in hours, minutes, and seconds
```
The model narrates inside the directive. Degenerate small-model behaviour, but it produces an
artifact that parses as a script and fails only at the semantic level.

### F3 note: the diagnosis is parser-dependent
Before the multi-option parser fix, `#SBATCH --mem=2 referencing GB` was reported as
`--mem unparsable`. After it, `shlex` splits the line, the value becomes `2`, which parses
cleanly as 2 MB, so the same artifact is now reported as `--mem 2MB below minimum 2048MB`.

The error is still caught, but **its category changed**: degenerate prose now masquerades as an
under-request. For a taxonomy built on these categories this matters, and the two must be told
apart before T2 induction relies on them.

## F4: Missing directive with no universal default
`--time` and `--mem` omitted where the spec demanded them. Unlike F1 there is no defensible
default: the resource is simply never requested.

## F5: No `#SBATCH` at all
On the container task the model returned a plain shell script. The artifact is not a job script.

## F6: Payload/spec mismatch
`THREADS=4` expected, not printed: the script exports `OMP_NUM_THREADS` from a variable that its
own (missing) `--cpus-per-task` never set. The failure cascades from F4.

## F7: Malformed directive values rejected by the scheduler
```
sbatch: error: Invalid directive found in batch script: referencing
sbatch: error: Invalid --time specification
```
F3's prose leakage surfaces at `submittability`: the scheduler itself refuses the artifact.
Three of the eight tasks failed here. This is the class where dry-run validation *does* work.

---

## F8: Memory request below what the payload uses

The class the verifier had to grow a new ability to see: with real submission and cgroup
enforcement, a script that asks for less memory than it uses comes back `OUT_OF_MEMORY` instead of
completing. It cannot be induced from the eight T1 tasks, because each of them states a memory
minimum and a value below it fails `resource_fit` before anything runs, which is F3's and F4's
territory. `tasks/t1_exec.jsonl` therefore states no minimum: the payload holds 64MB and the prompt
asks for enough memory to fit it, so the ground truth is what the script actually needs.

**Observed, on the same 1.5B model.** 15 samples, 3 seeds, n=5, verified twice inside the same
image, once per executor (`results/executor_20260802_081626` on the development machine,
`scripts/memory_request.py` for the breakdown). Fourteen of the fifteen request exactly
`--mem=64M`: the size of the data, with nothing left for the process that builds it. The fifteenth
asks for 6M.

One sample is the whole point:

```
#SBATCH --mem=64M
string=$(head -c 67108864 /dev/zero | tr '\0' '\n')
```

It passes `syntax`, `submittability`, `resource_fit` and `safety`, it runs to completion under the
`bash` executor, and it is counted as a fully correct artifact by everything the benchmark could do
before this. Submitted for real it is OOM-killed: a command substitution holds the pipe buffer and
the variable at once, so the peak is at least twice the 64MB the model reasoned about. Strict
pass@1 on that seed goes from 0.20 under `bash` to 0.00 under real submission.

Two caveats, both structural. First, 64M sits on the boundary: three other samples requesting the
same value completed, because how the payload is written moves the peak, which is precisely why
only execution can decide the outcome. Second, the executor stopped three samples in total, but the
other two had already failed `submittability`; for those, real execution only aligned `functional`
with what the level above had said. The new information is the one sample that everything else
called correct.

**One model, one task, 15 samples: an observation, not a rate.**

---

## Two verifications that dry-run cannot do

Both observed in the container, against a live `slurmctld`:

| Task | `submittability` | Anvil |
|---|---|---|
| `t1_mpi_multinode` | **PASS** | `resource_fit` FAIL: ntasks effective 2, expected 4 |
| `t1_output_paths` | **PASS** | `syntax` FAIL: 4 directives after the first command |

The scheduler accepts both. In the first the job silently runs at half the requested parallelism;
in the second, job name, walltime, stdout and stderr paths are silently discarded.

**This is the benchmark's reason to exist, demonstrated on a real model and a real scheduler.**

---

## A harness bug the run exposed

`sbatch` reported *"Invalid --time specification"* on `t1_gpu_single` while Anvil reported
*"--time missing"*. Both cannot be right. SLURM parses a `#SBATCH` line like a command line, so
several options may share one line: the parser read only the first and swallowed the rest.

Same class as the `resource_fit` defaults bug: **surface-form parsing producing false negatives.**
Fixed; short options (`-t`, `-N`, `-c`, ...) are now normalised to their long forms, because `-t`
*is* `--time` and demanding the long spelling would measure style, not correctness.

---

## What this changes

1. **T2 induction can be grounded.** F1–F7 are real, and each maps to an inducer in
   `anvil/inducer.py`, mechanically applied to the T1 canonical solutions to build
   `tasks/t2_repair.jsonl` (see [DESIGN.md](DESIGN.md#diagnose-and-repair-t2)).
2. **F1 is the class to build the paper around.** It is invisible to every proxy metric, invisible
   to the scheduler, and expensive in practice.
3. **`resource_fit` must reason about effective requests.** An earlier version compared string
   presence and reported `found None`, which would have hidden F1 entirely behind a generic
   "missing directive". The bug fix *produced* the finding.

## Multi-seed validation (T1 and T2)

3 seeds (0/1/2), n=5, two model sizes, 4-bit on an RTX 3060. Generated on the experiment
machine, **graded inside the container**, which the first version of this table was not: see
[A table measured against the wrong cluster](#a-table-measured-against-the-wrong-cluster) below.
Generations in `results/20260802_091236/`, verdicts in `results/executor_20260802_140437/`.

**T1 (from scratch), pass@1, mean and half-range across seeds:**

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict (bash) | strict (sbatch) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.575±0.025 | 0.842±0.013 | 0.533±0.037 | 0.375±0.025 | 0.442±0.013 | 0.308±0.025 | 0.308±0.025 |
| Qwen2.5-Coder-7B-Instruct | 1.000±0.000 | 0.792±0.025 | 0.875±0.000 | 0.667±0.025 | 1.000±0.000 | 0.667±0.025 | 0.667±0.025 |

**T2 (diagnose-and-repair), same protocol:**

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict (bash) | strict (sbatch) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.792±0.016 | 0.886±0.005 | 0.664±0.005 | 0.595±0.009 | 0.412±0.014 | 0.291±0.000 | 0.292±0.002 |
| Qwen2.5-Coder-7B-Instruct | 0.983±0.002 | 0.977±0.000 | 0.870±0.002 | 0.847±0.002 | 0.965±0.007 | 0.824±0.002 | 0.824±0.002 |

`safety` is 1.000±0.000 everywhere and is left out of both tables.

### `submittability` does not scale with model size

It falls: 0.842 at 1.5B against 0.792 at 7B on T1, while every other level improves and two of
them reach 1.000. The mechanism is visible in the refusals. Of 1560 verdicts, 106 were
`invalid partition specified`, naming `gpu`, `small`, or the placeholder `your_partition_name`
left in from a template. The reference cluster declares one partition and no task asks for one,
so a script that volunteers a name is asserting something about a cluster it has not seen, which
is exactly what gets a job rejected on a real system. The larger model writes better formed
scripts and volunteers more of them.

### Real submission moves `functional` and barely touches the verdict

`functional` drops under real submission in every cell, by 16 points at 1.5B and 21 at 7B on T1.
`strict_all_levels` does not move at all: 0.308 against 0.308, 0.667 against 0.667, and the two
T2 rows agree to within 0.001. Of 1560 artifacts, exactly **one** changes verdict between the two
executors.

The reason is that the scripts real submission stops were already failing another level. The 106
partition refusals fail `submittability` in both arms; the executor only propagates the verdict
into `functional`. So on these eight tasks the extra strictness of real execution is almost
entirely redundant with the static levels, which is worth saying plainly rather than claiming an
executor earns its keep by itself. Where it does earn it is on a task built to need it: F8 below
is invisible to every static check and to bash.

### The one artifact the two executors disagree about

```bash
if [ -z "$OMP_NUM_THREADS" ]; then
    export OMP_NUM_THREADS=$(nproc)
fi
```

A repair of `t1_cpus_per_task__F6`. Under `bash` it prints the host's core count and fails; under
real submission `ConstrainCores` confines the job to the four cores it was allocated, `nproc`
answers 4, and it passes. The sandbox is the one that is wrong here, and it is wrong in the
direction that matters least in aggregate and most in kind: it produces a **false negative** on an
idiomatic script. Deriving threads from the allocation is normal practice on a cluster that
enforces binding.

It also settles a question the roadmap had left open. Binding was listed as unobservable until a
task existed that reads its own affinity. No such task was needed: a model wrote one.

### F1 is the hardest repair category

At 1.5B, `resource_fit` for F1 repairs is 0.0 on all three seeds: the small model never restores
the missing directive. At 7B `resource_fit` reaches 1.0 while `strict_all_levels` caps at 0.33,
with `submittability` at 0.33 as the bottleneck: the model now understands what was missing, and
the resulting script still often fails `sbatch --test-only`. F6 reaches `strict_all_levels` 1.0
for both models on almost every seed, and is the easy category of the taxonomy.

### A table measured against the wrong cluster

The first version of these numbers was verified on the experiment machine's own SLURM rather than
inside the container. Every result file recorded it, `base_image: "n/a (outside the container)"`,
and nobody read that field. That scheduler has no GPUs, one node, and no job 12345, so it rejects
three of the eight canonical solutions: the oracle itself scored `submittability` 0.625 there.

The published numbers were 0.68 and 0.62; graded against the declared topology they are 0.842 and
0.792, and `strict_all_levels` moves from 0.18 and 0.50 to 0.308 and 0.667. T2 moves further
still, the 7B from 0.46 to 0.824. `syntax`, `functional` and `resource_fit` are unchanged to the
digit, which is the expected signature: they do not depend on the scheduler.

The distortion was not a uniform ceiling either. The 1.5B scored 0.68 where the oracle scored
0.625, because on a cluster without GPUs a script that forgets `--gpus` is accepted and the
correct one is refused. That environment rewarded the omission F1 and F4 exist to catch.

The canary could not have caught this. It submits a minimal script and asks whether the scheduler
accepts it, and any scheduler does. What was missing is a check that the scheduler in front of us
implements the topology this benchmark declares, and it is now the first item under
[Next measurements needed](#next-measurements-needed).

## Next measurements needed

The three gaps named above (multiple seeds, a scheduler-faithful environment, a larger model)
are resolved; see [Multi-seed validation](#multi-seed-validation-t1-and-t2). What remains open:

- a preflight that checks the scheduler in front of us against the *declared* topology, not only
  that it accepts a minimal script. The canary passes on any working SLURM, which is how a whole
  table came to be measured against a one-node cluster with no GPUs. A second canary requesting
  what `ANVIL_NODES` and `ANVIL_GPUS` promise would have refused to score at all;
- more than two model sizes and families, to see where the `submittability` plateau breaks, if
  it breaks;
- a genuine outlier check on F3, to separate small-model degeneracy from a stable semantic
  error as model scale keeps increasing;
- the mechanism behind the `vectorless` `resource_fit` collapse (0.19 against 0.49 zero-shot, see
  [DESIGN.md](DESIGN.md#retrieval-ablation)). Two candidates were measured with
  `scripts/retrieval_copying.py` and both are refuted: the arm that collapses reproduces the
  corpus values *least*, and the share of failures that are omissions falls instead of rising.
  What moves the level is therefore still unidentified;
- the same ablation on a larger model, and a variant that prepends context instead of
  appending it;
- the T1 and T2 matrices measured again under `--executor sbatch`, to see how far `functional`
  moves once the requested walltime is enforced and the payload receives the scheduler's own
  environment instead of three simulated variables. The tooling is in place
  (`scripts/executor_ablation.sh` against the accounting image), and the one task measured that
  way so far is the execution set below; the matrices themselves need generations, which the
  earlier multi-seed run did not save;
- F8 beyond one task and one model: the observation below is 15 samples of a 1.5B model on a
  single task whose payload sits on the boundary of what it requests. Whether larger models leave
  headroom, and whether the error survives a payload whose need is unambiguous, is unmeasured;
- a T1 task that depends on a coreutils corner where `uutils` and GNU are known to differ
  (`stat`, `sort`, `date` formatting, flag-level behaviour). The cross-distribution ablation
  now agrees across 3 seeds and 360 level comparisons, but none of the eight current tasks
  reaches those corners, so the agreement measures the tasks as much as the toolchains.

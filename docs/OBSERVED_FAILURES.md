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

## F9: An option this scheduler does not have

```
sbatch: unrecognized option '--walltime=30'
```

Granite 4.1 3B on `t1_mpi_multinode`, all fifteen samples, all three seeds, and nowhere else in
its 780 verdicts. `--walltime` is not a SLURM option. `walltime` is what PBS calls the same
resource, requested as `-l walltime=hh:mm:ss`, so the artifact carries SLURM's long-option form
around PBS's parameter name: a blend of two schedulers rather than an error in either. It appears
on the one multi-node task, which is where published examples are most likely to be PBS.

F7 is a real directive with a value the parser rejects, and a better value would have saved it.
Here no value exists that would: the option itself is not in the vocabulary. The distinction is
worth keeping because it is also the distinction between a portable failure and a local one. The
partition refusals that dominate the Qwen rows depend on which cluster is asked; this one does
not.

It has no inducer, and that is a decision rather than an omission. `tasks/t2_repair.jsonl` is held
equal to what the inducers produce by a test, so registering F9 regenerates it, and that file is
the denominator of every T2 number published here, including the three-model table below. The
fault it would teach is also close to F7's: both are refused at `submittability`, and the models
that clear F7 above 0.83 would most likely clear this too. It stays an observed class, and joins
the induced ones if the T2 set is ever regenerated for an independent reason.

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

3 seeds (0/1/2), n=5, three models across two families, 4-bit on an RTX 3060. Generated on the experiment
machine, **graded inside the container**, which the first version of this table was not: see
[A table measured against the wrong cluster](#a-table-measured-against-the-wrong-cluster) below.
Generations in `results/20260802_091236/`, verdicts in `results/executor_20260802_140437/`.

**T1 (from scratch), pass@1, mean and half-range across seeds:**

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict (bash) | strict (sbatch) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.575±0.025 | 0.842±0.013 | 0.533±0.037 | 0.375±0.025 | 0.442±0.013 | 0.308±0.025 | 0.308±0.025 |
| Qwen2.5-Coder-7B-Instruct | 1.000±0.000 | 0.792±0.025 | 0.875±0.000 | 0.667±0.025 | 1.000±0.000 | 0.667±0.025 | 0.667±0.025 |
| granite-4.1-3b | 1.000±0.000 | 0.875±0.000 | 0.842±0.037 | 0.717±0.037 | 0.625±0.000 | 0.500±0.000 | 0.500±0.000 |

**T2 (diagnose-and-repair), same protocol:**

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict (bash) | strict (sbatch) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.792±0.016 | 0.886±0.005 | 0.664±0.005 | 0.595±0.009 | 0.412±0.014 | 0.291±0.000 | 0.292±0.002 |
| Qwen2.5-Coder-7B-Instruct | 0.983±0.002 | 0.977±0.000 | 0.870±0.002 | 0.847±0.002 | 0.965±0.007 | 0.824±0.002 | 0.824±0.002 |
| granite-4.1-3b | 0.862±0.002 | 1.000±0.000 | 0.750±0.000 | 0.750±0.000 | 0.753±0.009 | 0.661±0.009 | 0.661±0.009 |

`safety` is 1.000±0.000 everywhere and is left out of both tables.

### `submittability` does not track model size

Ordered by the level itself, T1 reads 0.875 for Granite at 3B, 0.842 for Qwen at 1.5B, 0.792 for
Qwen at 7B; T2 reads 1.000, 0.977 and 0.886, with the 7B second and the 1.5B last. The smallest
model of the second family leads both, and inside the Qwen family the level falls as size rises
while every other level improves and two of them reach 1.000. Whatever `submittability` measures,
parameter count does not order it.

The refusals show why, and they are not the same failure in the two families. Qwen invents
*values*: of 1560 verdicts, 106 were `invalid partition specified`, naming `gpu`, `small`, or the
placeholder `your_partition_name` left in from a template. The reference cluster declares one
partition and no task asks for one, so a script that volunteers a name is asserting something
about a cluster it has not seen. The larger model writes better formed scripts and volunteers more
of them.

Granite invents *syntax*. Not one of its refusals names a partition, in T2 it has none at all
across 660 samples, and its entire T1 deficit is one task failing all fifteen times on an option
SLURM does not have: F9 below. That difference matters for what the level is worth. A script
asking for `--partition=gpu` would be accepted on a cluster that happens to have a `gpu`
partition, so those refusals are site-dependent and a reader may fairly discount them;
`--walltime` is refused by every SLURM installation there is.

So the level does not rank models by capability. It measures how much each one adds that the
prompt never asked for, and which scheduler's vocabulary it reaches for when it does. Two
families, two habits, and a 3B ahead of a 7B.

### Real submission moves `functional` and barely touches the verdict

`functional` drops under real submission in every cell, by 16 points at 1.5B, 21 at 7B and 12 at
Granite 3B on T1. `strict_all_levels` does not move at all: 0.308 against 0.308, 0.667 against
0.667, 0.500 against 0.500, and the T2 rows agree to within 0.001. Of 2340 artifacts, exactly
**one** changes verdict between the two executors.

The reason is that the scripts real submission stops were already failing another level. All 121
refusals, the 106 partition names and Granite's 15 unknown options, fail `submittability` in both
arms; the executor only propagates the verdict into `functional`. So on these eight tasks the
extra strictness of real execution is almost
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

### Which fault is hardest depends on the model, and on the level

Per fault category, `bash` arm, three seeds pooled. F1 applies to three tasks and F6 to one, hence
the smaller denominators.

| category | 1.5B | 7B | Granite 3B | n per model |
|---|---|---|---|---|
| F1 omitted default | 0.000 | 1.000 | 0.356 | 45 |
| F2 directive after the first command | 0.342 | 0.875 | 0.750 | 120 |
| F3 prose in a value | 0.542 | 0.875 | 0.800 | 120 |
| F4 directive absent | 0.000 | 0.750 | 0.742 | 120 |
| F5 no `#SBATCH` at all | 0.050 | 0.658 | 0.250 | 120 |
| F6 payload/spec mismatch | 0.933 | 1.000 | 1.000 | 15 |
| F7 malformed value | 0.550 | 0.875 | 0.833 | 120 |

`strict_all_levels` pass@1. **F5 is the lowest category for two of the three models and never
comfortable for any of them**, which no single-model view showed: it is the fault that leaves the
artifact furthest from a job script, and restoring every directive from the prompt alone is closer
to writing one than to repairing one. F6 stays the easy category, as it was with two models.

F1 is the one that separates them: 0.000, 0.356, 1.000, the widest spread in the table, on the
fault this document opens with. A benchmark wants categories like that, and a claim about model
capability made on F6 would be worth nothing.

An earlier version of this section called F1 the hardest category outright, on the strength of the
7B capping at 0.33 there with `submittability` as the bottleneck. That number came from the run
graded against the wrong cluster, and `submittability` collapsing is its signature. Regraded in the
container, F1 at 7B is 1.000. The correction is recorded rather than quietly dropped because the
retracted claim is the more interesting one: F1 is not hard for a model that has understood it,
it is hard to *tell* whether a model has.

Two structural facts hold across all 2340 verdicts. First, `syntax` fails only ever on F2 and F5:
F1, F3, F4, F6 and F7 are 1.000 for every model without one exception. Those five leave a
well-formed script behind and the fault surfaces higher up, at `resource_fit` or
`submittability`; F2 and F5 are the only two that concern whether directives exist and where they
sit, which is what `syntax` is able to look at. A repair fails at the level its fault lives on.

Second, Granite at 3B is within twelve points of the 7B on F2, F3, F4 and F7, and the whole gap
between its 0.661 and the 7B's 0.824 comes from F1 and F5. Two categories out of seven account for
a model of less than half the size trailing.

### Writing a directive and noticing it is missing are different abilities

Granite scores `syntax` 1.000 on T1 and 0.367 on F5 repairs. It writes correct `#SBATCH` blocks
whenever a prompt asks for a script, 120 samples out of 120, and hands back a plain shell script
still missing every directive on 75 of the 120 repairs where the fault is that they were removed.

The comparison rules out the obvious explanation. The 1.5B scores 0.575 and 0.375, low in both:
it is simply weak. The 7B scores 1.000 and 0.908, high in both. Granite is the only cell where the
two diverge, and the divergence is not a general weakness, since its `syntax` is 1.000 on five of
the seven repair categories and 0.875 on the sixth.

This is the clearest evidence so far that T2 measures something T1 does not, and the reason it is
a separate task set rather than a variant. Until now that separation rested on the scores moving;
here one model has the generation ability at ceiling and the recognition ability at floor, on the
same level and the same metric.

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
implements the topology this benchmark declares. That check is `_topology_healthy`, and the first
model run after it landed hit it on this same machine: `submittability` was skipped, with
`Invalid generic resource (gres) specification` given as the reason, rather than scored against a
cluster that is not the reference one.

## Next measurements needed

The gaps named above are resolved: multiple seeds, a scheduler-faithful environment, a larger
model, a second family, and both matrices measured under real submission; see
[Multi-seed validation](#multi-seed-validation-t1-and-t2). So is the topology preflight that the
wrong-cluster table called for. `_topology_healthy` submits a job asking for the `ANVIL_NODES` and
`ANVIL_GPUS` the benchmark declares, and on the experiment machine's own SLURM it does what it was
built to do: `submittability` is skipped with the reason stated, instead of scoring. So is the
mechanism behind the `vectorless` `resource_fit` collapse, after three refuted candidates: the
damage is confined to `--time` and `--mem`, the two directives these tasks state as a bound rather
than a figure, see [What the level breaks on](DESIGN.md#what-the-level-breaks-on). What remains
open:

- a third family. Two of them were enough to show that `submittability` is not ordered by size,
  because the two families fail it for different reasons and one of the reasons is site-dependent
  while the other is not. Whether "invented values against invented syntax" is a real split or two
  points that happen to differ needs a third habit to compare against;
- a genuine outlier check on F3, to separate small-model degeneracy from a stable semantic
  error as model scale keeps increasing;
- the same ablation on a larger model, and a variant that prepends context instead of
  appending it;
- F8 beyond one task and one model: the observation below is 15 samples of a 1.5B model on a
  single task whose payload sits on the boundary of what it requests. Whether larger models leave
  headroom, and whether the error survives a payload whose need is unambiguous, is unmeasured;
- a T1 task that depends on a coreutils corner where `uutils` and GNU are known to differ
  (`stat`, `sort`, `date` formatting, flag-level behaviour). The cross-distribution ablation
  now agrees across 3 seeds and 360 level comparisons, but none of the eight current tasks
  reaches those corners, so the agreement measures the tasks as much as the toolchains.

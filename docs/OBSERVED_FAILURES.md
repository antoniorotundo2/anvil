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

A second model has since produced the class, and on the same task:

```
sbatch: unrecognized option '--mem-per-node=4G'
```

Gemma 4 12B, twelve of the fifteen F4 repairs of `t1_mpi_multinode`. `--mem-per-node` is not a
SLURM option either, and unlike `--walltime` it is not another scheduler's name for the resource:
SLURM's `--mem` already is per node, and `--mem-per-cpu` and `--mem-per-gpu` exist beside it, so
the invented option is the one that completes the pattern. Two families, two different routes to
an option that does not exist, both on the multi-node task. That F9 is not a Granite quirk is now
the more defensible reading.

F7 is a real directive with a value the parser rejects, and a better value would have saved it.
Here no value exists that would: the option itself is not in the vocabulary. The distinction is
worth keeping because it is also the distinction between a portable failure and a local one. The
partition refusals that dominate the Qwen rows depend on which cluster is asked; this one does
not.

It has no inducer, and that is a decision rather than an omission. `tasks/t2_repair.jsonl` is held
equal to what the inducers produce by a test, so registering F9 regenerates it, and that file is
the denominator of every T2 number published here, including the tables below. The
fault it would teach is also close to F7's: both are refused at `submittability`, and the four
models that clear F7 between 0.775 and 0.875 would most likely clear this too. It stays an
observed class, and joins the induced ones if the T2 set is ever regenerated for an independent
reason.

## F10: A unit confusion the scheduler accepts

```
#SBATCH --time=45:00:00     the prompt asks for 45 minutes
#SBATCH --time=25:00:00     the prompt asks for 25 minutes
#SBATCH --time=10:00:00     the prompt asks for 10 minutes
```

The integer is the one the prompt named. It is in the leading field of `hours:minutes:seconds`,
so the job asks for sixty times the walltime it was told to ask for. Both models that produce it
also write the correct forms, `00:45:00` and the bare `MM:SS`, on other samples of the same task,
so this is not a model that lacks the format.

Two families, from-scratch generation, three seeds each. Qwen3.5-9B does it in 34 artifacts of
120, on `t1_hello_serial`, `t1_container_apptainer`, `t1_mpi_multinode` and `t1_cpus_per_task`.
Gemma 4 12B does it in 10 of 120, all on `t1_dependency_chain`, a task where Qwen3.5 never does.
The habit is shared; which prompt triggers it is not.

The one prompt that cannot expose it is `t1_gpu_single`, which asks for a walltime of two hours and
gets `02:00:00` from both. Reading the integer and ignoring the unit gives the right answer when
the unit is the field's own. Everywhere else the unit is minutes and the answer is wrong by a
factor of sixty.

**This class is the one that no level except `resource_fit` sees.** The directive is well formed,
so `syntax` passes it. `sbatch` accepts it without a word, so `submittability` passes it. The job
runs and prints what the task asked for, so `functional` passes it. It is not a syntax error, not
an unknown option, and not a value a parser can refuse: every static check this project compares
itself against reports a correct script. On a real cluster it is a request that a queue policy
rejects or, worse, admits and schedules badly.

It has no inducer, for the reason given under F9: registering one regenerates
`tasks/t2_repair.jsonl`, and that file is the denominator of every T2 number published here. It is
the class most worth inducing when that regeneration happens for an independent reason, because it
is the only one whose repair cannot be faked by any check cheaper than the one this benchmark
runs.

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

3 seeds (0/1/2), n=5, five models across three families and two Qwen generations, 4-bit on an RTX 3060. Generated on the experiment
machine, **graded inside the container**, which the first version of this table was not: see
[A table measured against the wrong cluster](#a-table-measured-against-the-wrong-cluster) below.
Generations in `results/20260802_091236/`, verdicts in `results/executor_20260802_140437/`.

**T1 (from scratch), pass@1, mean and half-range across seeds:**

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict (bash) | strict (sbatch) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.575±0.025 | 0.842±0.013 | 0.533±0.037 | 0.375±0.025 | 0.442±0.013 | 0.308±0.025 | 0.308±0.025 |
| Qwen2.5-Coder-7B-Instruct | 1.000±0.000 | 0.792±0.025 | 0.875±0.000 | 0.667±0.025 | 1.000±0.000 | 0.667±0.025 | 0.667±0.025 |
| granite-4.1-3b | 1.000±0.000 | 0.875±0.000 | 0.842±0.037 | 0.717±0.037 | 0.625±0.000 | 0.500±0.000 | 0.500±0.000 |
| gemma-4-12B-it | 0.875±0.000 | 0.867±0.013 | 0.875±0.000 | 0.742±0.013 | 0.917±0.050 | 0.658±0.062 | 0.658±0.062 |
| Qwen3.5-9B | 1.000±0.000 | 0.875±0.025 | 0.650±0.025 | 0.658±0.013 | 0.600±0.050 | 0.450±0.025 | 0.475±0.025 |

**T2 (diagnose-and-repair), same protocol:**

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict (bash) | strict (sbatch) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.792±0.016 | 0.886±0.005 | 0.664±0.005 | 0.595±0.009 | 0.412±0.014 | 0.291±0.000 | 0.292±0.002 |
| Qwen2.5-Coder-7B-Instruct | 0.983±0.002 | 0.977±0.000 | 0.870±0.002 | 0.847±0.002 | 0.965±0.007 | 0.824±0.002 | 0.824±0.002 |
| granite-4.1-3b | 0.862±0.002 | 1.000±0.000 | 0.750±0.000 | 0.750±0.000 | 0.753±0.009 | 0.661±0.009 | 0.661±0.009 |
| gemma-4-12B-it | 1.000±0.000 | 0.876±0.002 | 0.885±0.002 | 0.762±0.002 | 0.938±0.002 | 0.732±0.000 | 0.732±0.000 |
| Qwen3.5-9B | 0.982±0.005 | 0.982±0.005 | 0.947±0.011 | 0.950±0.009 | 0.711±0.009 | 0.691±0.009 | 0.694±0.009 |

`safety` is 1.000±0.000 everywhere and is left out of both tables.

### `submittability` does not track model size

On T1 the level reads 0.875 for Granite at 3B, 0.867 for Gemma at 12B, 0.842 for Qwen at 1.5B and
0.792 for Qwen at 7B. **The largest model in the set and the smallest of another family sit
together at the top, and the 7B is alone at the bottom**, while inside the Qwen family the level
falls as size rises and every other level improves. Parameter count does not order it, and with a
third family the shortfall is no longer attributable to one model's quirk: two families out of
three are above Qwen, across a range from 3B to 12B.

T2 does not repeat the ordering, and the section would be overclaiming if it said otherwise:
Granite 1.000, Qwen 7B 0.977, Qwen 1.5B 0.886, Gemma 0.876. Every model sits between 0.88 and
1.00 there, which leaves little room to separate them, and the family that leads T1 is last.

The refusals show why, and they are not the same failure in the two families. Qwen invents
*values*: of 1560 verdicts, 106 were `invalid partition specified`, naming `gpu`, `small`, or the
placeholder `your_partition_name` left in from a template. The reference cluster declares one
partition and no task asks for one, so a script that volunteers a name is asserting something
about a cluster it has not seen. The larger model writes better formed scripts and volunteers more
of them.

Gemma sits with Granite without sharing its mechanism, which is what a third family was for: it
names no impossible option, and its T1 shortfall is spread across tasks rather than concentrated in
one.

Granite invents *syntax*. Not one of its refusals names a partition, in T2 it has none at all
across 660 samples, and its entire T1 deficit is one task failing all fifteen times on an option
SLURM does not have: F9 below. That difference matters for what the level is worth. A script
asking for `--partition=gpu` would be accepted on a cluster that happens to have a `gpu`
partition, so those refusals are site-dependent and a reader may fairly discount them;
`--walltime` is refused by every SLURM installation there is.

So the level does not rank models by capability. It measures how much each one adds that the
prompt never asked for, and which scheduler's vocabulary it reaches for when it does. Two
families, two habits, and a 3B ahead of a 7B.

### Real submission was almost redundant until a model wrote `srun`

For four models this section said the sandbox and the scheduler agree. `functional` dropped under
real submission in every cell, by 12 to 21 points, and `strict_all_levels` did not move at all:
0.308 against 0.308, 0.667 against 0.667, 0.500 against 0.500, 0.658 against 0.658. One artifact of
3120 changed verdict. The reason was that everything real submission stopped had already failed
another level, so the executor propagated a verdict rather than producing one.

The fifth model ended that. Of **3900 artifacts, six** change verdict, **32** fail under `bash` and
pass under real submission, and five of the six are Qwen3.5-9B on `t1_hello_serial`, across both
the from-scratch task and its F5 repairs. They all fail the same way:

```
srun: error: Unable to confirm allocation for job 1: Invalid job id specified
```

The model writes `srun` inside the batch script, which is how a job step is launched on a real
cluster and what a user would expect to see. In the `bash` sandbox there is no allocation for
`srun` to attach to, so it exits non-zero and the sample is recorded as failing `functional`. Under
real submission the allocation exists, the step runs, and the same script passes. **The sandbox
produces a false negative on every script that launches a job step**, and it did so silently for as
long as no model wrote one.

That is the correction, and it is worth more than the count. The claim that the expensive executor
was nearly redundant was true of the four models measured, not of the method: it described a
property of what those models happened to write. One newer model, writing SLURM the way SLURM is
normally written, moved the number from one in 3120 to six in 3900. A benchmark can only report the
artifacts it has been given, and this is what that limitation looks like when it breaks.

The sandbox is still the default and every published figure still uses it, because switching would
make later numbers incomparable with earlier ones. What changes is the standing of the
recommendation: `--executor sbatch` is not an optional refinement for task sets whose models use
`srun`, and the gap will widen as models write more idiomatic scripts rather than less.

### The other artifact the two executors disagree about

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

Both disagreements point the same way, and it is the opposite of the one a sandbox is usually
suspected of. Neither is the scheduler catching something the sandbox missed; both are the sandbox
rejecting a script the scheduler accepts, because the sandbox cannot supply an allocation, a core
binding, or a job step. Real submission has yet to catch a single artifact that `bash` promoted and
that was genuinely wrong, on any of the eight tasks. The one place it does is a task built for it,
F8 above, which no static check and no sandbox can see.

### Which fault is hardest depends on the model, and on the level

Per fault category, `bash` arm, three seeds pooled. F1 applies to three tasks and F6 to one, hence
the smaller denominators.

| category | 1.5B | 7B | Granite 3B | Gemma 12B | Qwen3.5 9B | n per model |
|---|---|---|---|---|---|---|
| F1 omitted default | 0.000 | 1.000 | 0.356 | 0.667 | 0.667 | 45 |
| F2 directive after the first command | 0.342 | 0.875 | 0.750 | 0.875 | 1.000 | 120 |
| F3 prose in a value | 0.542 | 0.875 | 0.800 | 0.875 | 0.983 | 120 |
| F4 directive absent | 0.000 | 0.750 | 0.742 | 0.750 | 0.242 | 120 |
| F5 no `#SBATCH` at all | 0.050 | 0.658 | 0.250 | 0.375 | 0.383 | 120 |
| F6 payload/spec mismatch | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 15 |
| F7 malformed value | 0.550 | 0.875 | 0.833 | 0.775 | 0.817 | 120 |

`strict_all_levels` pass@1. **F5 is the lowest category for three of the five models and never
comfortable for any of them**: it is the fault that leaves the artifact furthest from a job script,
and restoring every directive from the prompt alone is closer to writing one than to repairing one.
F6 stays the easy category, 1.000 for everything except the 1.5B, so it separates nothing.

F1 separates the models: 0.000, 0.356, 0.667, 0.667, 1.000 on the fault this document opens with.
A benchmark wants categories like that, and a claim about model capability made on F6 would be
worth nothing.

**F4 is where the fifth model breaks the pattern**, and it is the only category that does. Four
models sit between 0.742 and 0.750 there, near enough to be one number; Qwen3.5-9B sits at 0.242.
That is the same collapse its `resource_fit` column shows on both task sets, localised. Its own
hardest category is F4 rather than F5, which is true of no other model here. The next two sections
take that number apart, and it does not survive the description above: F4 turns out not to be one
finding about one model.

One property of the category has to be stated first, because both sections depend on it. F4 drops
`--time`, `--mem` or `--gpus`, whichever the task declares, in that order. Every T1 task declares
`time_max_minutes`, so the first candidate always applies and the other two are never reached:
**as instantiated, F4 removes `--time` and nothing else, on all eight tasks**. The category is
written to cover three directives and currently exercises one. That is a property of the task set
rather than of the inducer, and it would change on its own if a task without a walltime constraint
were ever added.

An earlier version of this section called F1 the hardest category outright, on the strength of the
7B capping at 0.33 there with `submittability` as the bottleneck. That number came from the run
graded against the wrong cluster, and `submittability` collapsing is its signature. Regraded in the
container, F1 at 7B is 1.000. The correction is recorded rather than quietly dropped because the
retracted claim is the more interesting one: F1 is not hard for a model that has understood it,
it is hard to *tell* whether a model has.

Two structural facts hold across all 3120 verdicts. First, `syntax` fails only ever on F2 and F5:
F1, F3, F4, F6 and F7 are 1.000 for every model without one exception. Those five leave a
well-formed script behind and the fault surfaces higher up, at `resource_fit` or
`submittability`; F2 and F5 are the only two that concern whether directives exist and where they
sit, which is what `syntax` is able to look at. A repair fails at the level its fault lives on.

Second, the gap between the capable models is concentrated in the same two categories. Granite at
3B is within twelve points of the 7B on F2, F3, F4 and F7, and the whole distance between its 0.661
and the 7B's 0.824 comes from F1 and F5; Gemma at 12B matches the 7B exactly on F2, F3 and F4 and
trails it on the same two, F1 by 33 points and F5 by 28. Two categories out of seven account for
the ordering of three models across three families.

### The right number in the wrong field

Every one of Qwen3.5-9B's 91 F4 failures is `resource_fit`, and every one of them is `--time`.
No other directive, no other level, one stray `functional` aside. The failures split in two, and
neither half is what the score suggested.

Thirty-one artifacts request a walltime that the ceiling refuses. What they contain is this:

```
#SBATCH --time=45:00:00     the prompt asks for 45 minutes
#SBATCH --time=10:00:00     the prompt asks for 10 minutes
#SBATCH --time=25:00:00     the prompt asks for 25 minutes
```

The integer is right every time. It is in the wrong field. `45:00:00` is SLURM's `hours:minutes:
seconds`, so the artifact requests forty-five hours for a forty-five minute job, and the same
model writes `00:20:00` and `00:25:00` correctly on the two tasks it passes, on the same seeds.
This is not a model that does not know the constraint. It is a model that reads the number out of
the prompt and writes it into the leading field, and the leading field is hours.

The from-scratch run confirms the reading and rules out the repair setting as the cause. In T1,
34 of 120 artifacts carry the same slip: `45:00:00`, `10:00:00`, `30:00:00`, always the prompt's
own integer in the hours position. It is the largest single `resource_fit` defect the model has.

It is also not this model's alone. Gemma 4 12B writes `#SBATCH --time=25:00:00` on
`t1_dependency_chain` in 10 of its 15 T1 samples, against a prompt asking for 25 minutes, and
writes `00:25:00` and `25:00` correctly on the other five. Two families produce the same
confusion, on tasks that do not overlap: Qwen3.5 never slips on `t1_dependency_chain` and Gemma
slips nowhere else. That is what promoted it from a property of one model to
[F10](#f10-a-unit-confusion-the-scheduler-accepts), where the class and its consequence for the
levels are stated.

The other sixty failures are the opposite defect. On `t1_array_job`, `t1_cpus_per_task`,
`t1_gpu_single` and `t1_mpi_multinode` the repaired script comes back with no `--time` at all, 15
times out of 15 on each. Those are four of the tasks where the same model writes a correct
`--time` when generating from scratch. Shown a script with the directive removed, it hands the
script back with the directive still removed; asked for the same script with no example in front
of it, it writes the directive. The broken artifact is not being diagnosed, it is being copied,
which is the failure mode the section below describes for a different model on a different
category.

Both halves are near-deterministic: 15 identical artifacts per task, across three seeds and five
samples. A defect with no variance is not one more sampling attempt away from being fixed, which
is worth saying plainly because `pass@k` at higher `k` is the usual answer to a low `pass@1` and
here it would buy nothing.

### The category names the fault that was induced, not the fault that was found

Gemma 4 12B scores 0.750 on F4, in the middle of the four-model cluster, and its failures have
nothing to do with `--time`. It restores the removed directive on all eight tasks, 120 artifacts
out of 120, with the correct value in the correct field every time. Its thirty failures are two
tasks, failing whole, for two unrelated reasons it introduced itself:

```
sbatch: unrecognized option '--mem-per-node=4G'      t1_mpi_multinode, 12 of 15
sbatch: error: invalid partition specified: compute  t1_mpi_multinode, 3 of 15
expected output not found: ['ANVIL_OK']              t1_container_apptainer, 15 of 15
```

The first reading of this was that repairing introduces faults, since the model is asked for the
whole script and rewrites more than the broken line. The from-scratch run does not support it.
Gemma's T1 failures are `t1_mpi_multinode` 15 times out of 15 and `t1_container_apptainer` 15 out
of 15, which is exactly the set of tasks its F4 failures come from. **The two tasks fail in both
settings.** Nothing was introduced: the weakness is standing, and repair does not remove it.

What repair changes is which fault. Generating from scratch, `t1_mpi_multinode` is refused for an
invented partition name and `t1_container_apptainer` for a bash syntax error; repairing an F4, the
same two tasks are refused for an invented `--mem-per-node` and a payload that stops printing
`ANVIL_OK`. Same task, same verdict, four different reasons.

The scaffold does help elsewhere, which is worth recording because it cuts the other way. From
scratch Gemma fails `t1_dependency_chain` 10 times of 15 on a walltime over the ceiling and
`t1_gpu_single` once; repairing an F4 it passes both, on all fifteen. Being shown a script fixes
some of what it gets wrong when writing one.

So a per-category score is not a measurement of whether the induced fault was repaired. Gemma
restores the removed directive in 120 artifacts out of 120 and scores 0.750, and the 0.250 that is
missing is its ability to write two of the eight tasks at all. Where that ability is at floor the
category number carries no information about the category, and F4 at 0.750 for Gemma and F4 at
0.242 for Qwen3.5 are not two points on one scale. One model never gets the walltime wrong; the
other gets only the walltime wrong. Reading the column as a ranking on a single ability would have
been wrong in both directions.

The honest version of the earlier claim is narrower. Qwen3.5-9B's F4 collapse is real, it is
entirely `--time`, and it has two mechanisms, a unit slip and a failure to notice a removal. It is
not evidence that the model is worse at HPC than the 7B it succeeds. It is evidence that it
formats one field wrongly and anchors on the artifact it is shown, and the benchmark is what
separated those from a bad score.

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
mechanism behind the `vectorless` `resource_fit` collapse, after three refuted candidates and a
seven-condition intervention series across two model sizes: the damage is confined to `--time` and
`--mem`, most of it is the toll a small model pays for any attached text whatever it says, and the
rest is the one corpus document that addresses those two directives, worth 21 points to the model
that does not know the rule and minus 6 to the one that does. See
[What the level breaks on](DESIGN.md#what-the-level-breaks-on). What remains open:

- a third family. Two of them were enough to show that `submittability` is not ordered by size,
  because the two families fail it for different reasons and one of the reasons is site-dependent
  while the other is not. Whether "invented values against invented syntax" is a real split or two
  points that happen to differ needs a third habit to compare against;
- a genuine outlier check on F3, to separate small-model degeneracy from a stable semantic
  error as model scale keeps increasing;
- F10 on the three models it has not been looked for in. It is confirmed on Qwen3.5-9B and
  Gemma 4 12B, and the 1.5B, the 7B and Granite have not been screened: the count of
  `--time Nmin exceeds maximum Nmin` in their T1 reports is an upper bound on it, and the
  `--lines` mode of `scripts/category_dig.py` reads the values themselves. Whether the class is
  general or belongs to the two newest models is the difference between an observation and a
  finding, and it is one command per model;
- F8 beyond one task and one model: the observation below is 15 samples of a 1.5B model on a
  single task whose payload sits on the boundary of what it requests. Whether larger models leave
  headroom, and whether the error survives a payload whose need is unambiguous, is unmeasured;
- the toolchain-sensitive task on a model that can reach the question. `tasks/t1_coreutils.jsonl`
  splits the two implementations by construction, and both Qwen sizes fail it on both images for
  reasons that have nothing to do with either: a here-string newline at 1.5B, a missing output
  prefix at 7B. Thirty samples, no divergence, and not one pins a locale, see [A task that can tell
  the two apart](DESIGN.md#a-task-that-can-tell-the-two-apart). What would answer it is a model
  that clears the payload, which neither of these does;

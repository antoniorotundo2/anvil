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

### A second execution task, and why it is not the same one twice

Two tasks that fail F8 by the same mechanism are one task measured twice, so the second one
under-spends for a different reason. `t1_memory_workers` starts four workers at the same time, each
holding 32MB, and asks for enough memory for all four to be resident at once. `--mem` is the
allocation for the whole node, so the requirement is four times one worker's footprint, and a model
that reasons correctly about a single worker still under-requests by a factor of four. The first
task hides its cost inside one command substitution; this one hides it in concurrency, and the
directive's scope is the thing to get right rather than a pipeline's peak.

The properties that make it an F8 task rather than an F4 one are verified rather than asserted. The
oracle solves it under real submission with cgroup enforcement. The induced under-request, `--mem`
cut to 16M against a peak above 256MB, **passes `syntax`, `submittability`, `resource_fit`, `safety`
and `functional` under the `bash` executor, all five levels, and comes back `OUT_OF_MEMORY` under
real submission with no other level failing.** Both tasks now do that, and
`make docker-guards-enforcement` fails if either stops doing it.

### On a task set where only execution knows the answer, a third of the verdicts change

Five models, three seeds, `tasks/t1_exec.jsonl` and its repair set, graded twice in the same image,
900 sample comparisons. Verdicts in `results/exec_matrix/`, verifier `1727a930156d`.

**288 artifacts of 900 change their strict verdict with the executor.** On the main task set the
same comparison reads 6 of 3900. The published claim that real submission was nearly redundant is
not wrong about the runs it was made on, and it is wrong as a claim about the method: what changes
is the task set, from 0.15% to 32%, and 356 of the 394 stopped artifacts come back
`OUT_OF_MEMORY`.

**T1, from scratch, `functional` per arm:**

| model | `bash` | real submission |
|---|---|---|
| Qwen2.5-Coder 7B | 0.967 | **0.000** |
| Gemma 4 12B | 1.000 | 1.000 |
| Qwen3.5 9B | 0.467 | **0.700** |
| Granite 4.1 3B | 0.467 | 0.467 |
| Qwen2.5-Coder 1.5B | 0.333 | 0.167 |

The 7B row is the whole argument in one line. **Every one of its from-scratch artifacts is
OOM-killed under real submission, and the sandbox promotes 29 of the 30.** This is the model that
scores 1.000 on `resource_fit` and 0.667 strict on the main T1 set: strong where the requirement is
written in the prompt, and at floor where the requirement is a property of the payload it wrote
itself. Its strict falls from 0.467 to 0.000.

Gemma 4 12B is the only model that solves the set, 1.000 on every level under both executors. Its
answer is not a lucky one: it is the only model whose memory request covers what its own script
does.

Qwen3.5-9B moves the other way, 0.467 to 0.700, and the reason is the false negative already
recorded above: it writes `srun` inside the script, the sandbox has no allocation for it to attach
to, and the step exits non-zero under `bash` while running cleanly under a real scheduler.

**T2 repair, `strict` per arm:** 0.993 to 0.400 for the 7B, 0.880 to 0.560 for Gemma, 0.847 to
0.593 for Qwen3.5, 0.660 to 0.247 for Granite, 0.393 to 0.173 for the 1.5B. Every model loses at
least a third of its score when the allocation is enforced, and the ordering is preserved, so this
is not a re-ranking: it is a level of difficulty that the sandbox cannot see at all.

One counter did move, and it is the sandbox getting better rather than the models. Graded under
the verifier this set was first measured with, 32 artifacts failed under `bash` and passed under
real submission; under the current one, **11**. The difference is the session reaping added the
same day: a script that backgrounds work and exits used to leave its children holding the output
pipe, so reading to end-of-file waited for them and the job was recorded as timing out. Twenty-one
of the thirty-two disagreements were the sandbox failing a script for a defect of its own, on the
one task set whose prompts ask for background workers. The headline moved by one artifact, 298 to
297, and the false negatives fell by two thirds.

Two limits on the reading. The set is two T1 tasks and ten repairs, so a single task moves a number
by 0.5 on T1; these are not the 3900-sample figures published elsewhere and are not comparable with
them. And `t1_memory_workers` turned out to discriminate on `--ntasks` rather than on memory in the
from-scratch arm: the prompt asks for one task on one node and the 7B writes `--ntasks=4`, reading
four shell workers as four scheduler tasks, on all fifteen samples. That is a real spec violation
and a fair failure, but it is not the axis the task was built for, and it is worth knowing before
the numbers are read as being about memory alone. Rewording the prompt would move `tasks_sha` and
invalidate the 900 generations, so it stays as measured and the caveat is written down instead.

Two artifacts were stopped by the sandbox ceiling rather than by their own merits, and their
detail says so. Both were allocating more than a gigabyte for a payload of 64MB, so they were
already wrong; the note exists so that a reader does not mistake machine protection for a verdict
on the requested allocation.



Pointing the experiment runner at this set for the first time stopped it before it spent any GPU
time, with `a no-op repair passes induced faults - the repair verifier is too permissive`. The
runner's pre-GPU guard requires every induced fault to be refused, and under `bash` an F8 no-op
passes all five levels, which is the property F8 exists to demonstrate. The guard was right in
general and wrong here, and it had been wrong since the execution set existed; nobody had met it
because nobody had run the matrix against that set.

A guard that cannot decide has to say so rather than block, and rather than quietly excuse itself.
`anvil.inducer.NEEDS_ENFORCEMENT` names the classes no sandbox can judge, F8 being the only one, and
the runner now computes its bracket over the records the current executor can decide, prints how
many it excluded and why, and hard-stops if that leaves nothing. On the execution set it reads *0.0
strict on 8 records, 2 record(s) in ['F8'] not judged under bash*; on `tasks/t2_repair.jsonl` it
reads 44 records with no exclusion, because that set has no F8. The exclusion is not a hole: a
verifier that had become permissive would still be caught on the eight.

The second attempt at the run did spend GPU time, and then the shell disappeared. The host's
virtual machine had been killed. The execution set is the first whose payloads allocate for real,
and **the `bash` sandbox had no memory limit at all**: a model writes a script asking for tens of
gigabytes, nothing stands between it and the host, and the machine running the benchmark dies
rather than the artifact failing. The sandbox now runs under a ceiling, `ANVIL_SANDBOX_MEM_MB`,
1024 by default.

The ceiling is a constant and is never derived from `--mem` or from the task, which is not a detail
but the whole design. The `bash` executor has to keep ignoring the requested allocation, because
that is what F8 exists to demonstrate; a limit that tracked the request would quietly turn the
sandbox into an enforcing executor and delete the class. A test asserts the separation directly: a
script declaring `--mem=16M` and allocating 64MB still passes under `bash`, and both F8 no-op
repairs still pass all five levels there. When the ceiling does stop something, the reason says so,
`(sandbox ceiling 1024MB, not the requested allocation)`, so nobody reads machine protection as a
verdict on the artifact.

The ceiling alone was not the whole leak. The next run got six cells further and then died again,
during a model load with no script running, which is the shape of something that had been left
behind. **The sandbox never killed what a script started in the background.** Waiting for the shell
reaps the shell; a generated artifact that backgrounds work and exits without waiting leaves the
children running, holding whatever they allocated, and they accumulate across a matrix until the
host kills something unrelated. This task set is the first whose prompts ask for background workers,
so the leak is exactly as new as the allocation was. The sandbox now runs in its own session and
kills the whole process group afterwards, on the ordinary path and not only on timeout.

Fixing it turned up a second defect in the same lines, worse in a quieter way: the output was read
from pipes, and background children inherit the write end, so reading to end-of-file waited for
*them* rather than for the script. A job returning in a millisecond blocked for the entire timeout,
and nothing said so. Output goes to files now, and the two regression tests assert both halves: a
backgrounded child does not outlive the sandbox, and a script that backgrounds work returns at once.

`ulimit -v` does nothing on Darwin, so the ceiling is platform-dependent and reported rather than
assumed. `anvil doctor` prints `sandbox_mem_mb` and warns when it reads `uncapped`, and every report
carries the value: a run whose sandbox was uncapped is one that could have been stopped by the host
instead of by the benchmark.

One consequence of adding the task has to be stated rather than left implicit. `tasks/t1_exec.jsonl`
grew by append, so the `t1_memory_bound` record is byte-identical and the 15-sample observation
above is still an observation of the task it names. The file's digest moved all the same, which is
what `dataset/MANIFEST.json` and `tasks_sha` are for: re-verifying those generations needs the
one-task version of the file, and `anvil verify` will refuse them against this one. That is the
correct behaviour and not an obstacle to work around.

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
fault it would teach is also close to F7's: both are refused at `submittability`, and the three
models that clear F7 between 0.742 and 0.875 would most likely clear this too. It stays an
observed class, and joins the induced ones in the one pass where that file is rebuilt, see [What to
change when the T2 set is regenerated](#what-to-change-when-the-t2-set-is-regenerated).

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

All five models screened, from-scratch generation, three seeds each:

| model | T1 artifacts with the slip | tasks it appears on |
|---|---|---|
| Qwen2.5-Coder 1.5B | 14 of 120 | container, cpus_per_task, hello_serial, output_paths |
| Qwen2.5-Coder 7B | 0 of 120 | none |
| Granite 4.1 3B | 0 of 120 | none |
| Gemma 4 12B | 10 of 120 | dependency_chain |
| Qwen3.5 9B | 34 of 120 | container, hello_serial, mpi_multinode, cpus_per_task |

Three models of five, two families, and no ordering by size: the 7B sits between two Qwen
models that both do it and does not, and the 12B does it while the 3B does not. Which prompt
triggers it is not shared either. Gemma slips only on `t1_dependency_chain`, where Qwen3.5 never
does, and Qwen3.5 slips on four tasks that Gemma always gets right.

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
the class most worth inducing when that file is rebuilt, because it is the only one whose repair
cannot be faked by any check cheaper than the one this benchmark runs, and it is first on the list
in [What to change when the T2 set is
regenerated](#what-to-change-when-the-t2-set-is-regenerated).

### The mirror of F10, which this verifier did not catch

Screening for F10 turned up its opposite, and the opposite is a defect in the harness rather than
in a model:

```
#SBATCH --time=00:15    the prompt asks for 15 minutes; this is fifteen seconds
#SBATCH --time=00:30    the prompt asks for 30 minutes; this is thirty seconds
#SBATCH --time=02:00    the prompt asks for two hours; this is two minutes
```

`00:15` is SLURM's `minutes:seconds`, so the request is a sixtieth of what the prompt named
rather than sixty times it. **These artifacts pass.** `check_resource_fit` compares `--time`
against `time_max_minutes` in one direction only: above the ceiling is a problem, below it is not.
That reading was taken from `t1_hello_serial`, whose prompt says *at most* 10 minutes, and it is
wrong for the other seven, which name an exact walltime the way they name an exact node count.
The same function demands exact equality for `--nodes`, `--ntasks` and `--cpus-per-task`, so the
looseness is an inconsistency inside one check rather than a considered position.

`functional` does not cover for it. Every T1 payload finishes in under a second, so a job granted
fifteen seconds still prints what the task asked for and the level sees a correct run. F8 is the
same defect on memory and `functional` does catch that one, because a payload that allocates more
than it requested is killed for it; walltime has no equivalent here because the work is too small
to run out of time.

Granite 4.1 3B is where it is most visible, `--time=00:15` on all fifteen `t1_array_job` samples
and `00:30` on nine of fifteen `t1_cpus_per_task`, and the 1.5B produces `02:00` against a
two-hour prompt. Both are counted as correct in every number this repository has published.

`scripts/constraint_audit.py`, then a single-constraint script, counted it before anything
was changed: **123 of 2421 passing
artifacts, 5.1%**, across the five models and both task sets. 106 of them request under a minute.
The largest single group is 39 artifacts writing `--time=00:30` against `t1_cpus_per_task`, which
names 30 minutes, and the most extreme is 16 asking two minutes of the two-hour `t1_gpu_single`.

That was enough to settle it. `check_resource_fit` now reports a walltime below the declared one
as its own problem, worded so a report distinguishes the two directions: an over-request is a
queue policy problem and an under-request is a job killed early. The oracle still scores 1.0 under
the floor, which is the evidence that equality is the right reading and not an overreach, since
the canonical solution to every task already writes the walltime its prompt names.

Then the whole run was regraded, in one pass, from the saved generations. A verifier change does
not touch generations or `tasks_sha`, so no model was asked anything a second time.

**Five of the ten published cells moved, and one ordering changed with them:**

| | `resource_fit` before | after | `strict` before | after |
|---|---|---|---|---|
| T1 Granite 4.1 3B | 0.625 | 0.550 | 0.500 | 0.425 |
| T2 Granite 4.1 3B | 0.753 | 0.621 | 0.661 | 0.529 |
| T2 Qwen2.5-Coder 1.5B | 0.412 | 0.377 | 0.291 | 0.256 |
| T2 Gemma 4 12B | 0.938 | 0.932 | 0.732 | 0.726 |

The other six cells are identical to the digit. The 7B, which had not one artifact below the
floor, does not move on either task set, and neither does Qwen3.5.

On T1 `strict`, Granite falls from 0.500 to 0.425 and **passes below Qwen3.5-9B at 0.450**. Two
models swap rank because a one-sided comparison was missing one side. That is the concrete answer
to what a verifier defect costs: not a uniform inflation that leaves the reading intact, but a
reordering, concentrated on whichever model happened to have the habit the check could not see.

What the regrade did not touch is worth as much. The executor ablation returns the same six
artifacts whose strict verdict changes and the same 32 that fail under `bash` and pass under real
submission, with zero disagreements outside `functional`. The walltime floor and the executor
comparison measure disjoint things, which was assumed before and is now known.

Which is the part worth keeping from this episode. A report carries `tasks_sha`, and `anvil verify`
refuses generations whose task set has moved, because a changed task invalidates a comparison. A
changed verifier invalidates it exactly as thoroughly and nothing recorded that at all: the reports
from before this fix and the reports from after it were indistinguishable on disk. That was the
same class of gap the digest was built to close, on the other half of the pair, and it is closed
now. `anvil/provenance.py` digests the modules a verdict depends on, `verifier.py` and `parse.py`,
and every report the CLI writes carries `verifier_sha` beside `tasks_sha`. The leaderboard refuses
to rank a row whose rules are not the current ones, and the executor ablation refuses to compare
two gradings that did not come from the same verifier, which is a real risk for a run assembled
over several days.

The digest is taken over raw bytes, so a comment moves it. That is the conservative direction on
purpose: a changed digest means *these two gradings came from different code, find out why*, a
question worth being asked once too often, and not *the numbers are wrong*. Normalising the source
first would buy quieter digests at the price of the guarantee, and would tie the value to whichever
Python version had unparsed it.

Every leaderboard row currently reads `unstamped`, because every published entry was imported from
a report written before any of this existed. That is not a cosmetic state to clear: those rows were
graded without the walltime floor, and the marker is the accurate description of them until the
regrade.

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

3 seeds (0/1/2), n=5, five models across three families and two Qwen generations, 4-bit on an RTX
3060. Generated on the experiment
machine, **graded inside the container**, which the first version of this table was not: see
[A table measured against the wrong cluster](#a-table-measured-against-the-wrong-cluster) below.
Generations in `results/20260802_091236/`, verdicts in `results/regrade_mem/`, verifier
`20dd4a2e4159`. Two earlier gradings of these same generations are superseded, both of them for a
one-sided comparison in `check_resource_fit`: the first had no floor on `--time` and five of the ten
cells below moved when it was added, see [The mirror of
F10](#the-mirror-of-f10-which-this-verifier-did-not-catch); the second had no ceiling on `--mem` and
one cell moved, see [What the audit settled](#what-the-audit-settled).

**T1 (from scratch), pass@1, mean and half-range across seeds:**

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict (bash) | strict (sbatch) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.575±0.025 | 0.842±0.013 | 0.533±0.037 | 0.375±0.025 | 0.442±0.013 | 0.308±0.025 | 0.308±0.025 |
| Qwen2.5-Coder-7B-Instruct | 1.000±0.000 | 0.792±0.025 | 0.875±0.000 | 0.667±0.025 | 1.000±0.000 | 0.667±0.025 | 0.667±0.025 |
| granite-4.1-3b | 1.000±0.000 | 0.875±0.000 | 0.842±0.037 | 0.717±0.037 | 0.550±0.000 | 0.425±0.000 | 0.425±0.000 |
| gemma-4-12B-it | 0.875±0.000 | 0.867±0.013 | 0.875±0.000 | 0.742±0.013 | 0.917±0.050 | 0.658±0.062 | 0.658±0.062 |
| Qwen3.5-9B | 1.000±0.000 | 0.875±0.025 | 0.650±0.025 | 0.658±0.013 | 0.600±0.050 | 0.450±0.025 | 0.475±0.025 |

**T2 (diagnose-and-repair), same protocol:**

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict (bash) | strict (sbatch) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.792±0.016 | 0.886±0.005 | 0.664±0.005 | 0.595±0.009 | 0.330±0.009 | 0.211±0.007 | 0.212±0.005 |
| Qwen2.5-Coder-7B-Instruct | 0.983±0.002 | 0.977±0.000 | 0.870±0.002 | 0.847±0.002 | 0.965±0.007 | 0.824±0.002 | 0.824±0.002 |
| granite-4.1-3b | 0.862±0.002 | 1.000±0.000 | 0.750±0.000 | 0.750±0.000 | 0.621±0.016 | 0.529±0.016 | 0.529±0.016 |
| gemma-4-12B-it | 1.000±0.000 | 0.876±0.002 | 0.885±0.002 | 0.762±0.002 | 0.932±0.005 | 0.726±0.002 | 0.726±0.002 |
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

### What F3 actually measures

F3 is the widest gap in the per-category table, 0.292 for the 1.5B against 0.800 to 0.983 for the
other four, and the open question was whether that is small-model degeneracy or a stable semantic
error. It is neither. **83 of the 1.5B's 85 F3 failures are `--mem`, and all of them are one
mechanism.**

The inducer replaces the value with `#SBATCH --mem=2 referencing the requested memory`, the same
literal `2` on all eight tasks. What the model hands back, per task, is this:

```
--mem=2 referencing the requested memory   container, 15 of 15: the line comes back untouched
--mem=2                                    cpus_per_task, 15 of 15: prose stripped, digit kept
--mem=2G                                   array_job and output_paths, 15 of 15 each
--mem=2G                                   dependency_chain, 15 of 15, and these pass
--mem=512m                                 hello_serial, 15 of 15, and these pass
```

The digit is the inducer's, not the task's. Where the task happens to need 2GB the model appends `G`
to what it was shown and is right; where it needs 1GB the same move is double; where it appends no
unit at all the request is two megabytes. `t1_hello_serial` is the one task where it reads the
prompt
instead, and it passes every sample. **This is F10's mechanism on another field**: a digit carried
across from the text in front of the model with the unit decided independently, right when the two
happen to agree.

The note above asks for the two diagnoses to be told apart before T2 induction relies on them, and
the counts answer it. 52 failures read `--mem N MB below minimum N MB`, 31 read `--mem N MB above
the
N MB the task declares`, and only 15 are still legible as prose, the untouched container line caught
at `submittability` with `Invalid directive found in batch script: referenc`. So degenerate prose
does masquerade as a magnitude error, in 68 of 83 cases, and the category records a number problem
where the fault was a language problem. Worth keeping in mind before any claim rests on F3 being
about prose.

**A constant digit makes a quarter of the category repairable by luck.** `t1_cpus_per_task` and
`t1_dependency_chain` both declare 2048MB, which is exactly the inducer's `2` read as gigabytes,
so a
model that blindly appends `G` to the digit it was shown passes those two and fails the other six.
The 1.5B does precisely that and collects `t1_dependency_chain`. An inducer whose value varies with
the task would separate repairing from copying; the constant cannot. Changing it regenerates
`tasks/t2_repair.jsonl`, so it joins the F9 and F10 list of things to fix in the one pass where that
file is rebuilt.

Granite 4.1 3B is the control, at 0.800 on the same category with 24 failures, and its mechanism is
different rather than milder. Nine are `--mem not requested`: repairing a corrupted value, it
deletes
the directive instead of fixing it, which is a failed repair of the induced fault by another route.
The other fifteen are `t1_container_apptainer` returning a payload that never prints `ANVIL_OK`, the
same standing weakness that fails the task in from-scratch generation and under F4. So more than
half
of its F3 score is not about F3, which is the third model this holds for.

### What the audit settled

The walltime gap surfaced by accident, while five models were being screened for F10, which is a
bad way to find out that a check has a hole. `scripts/constraint_audit.py` asks the question on
purpose, for every constraint at once: among the artifacts the verifier passed, how many requested
less than the task declares, how many exactly, and how many more, printed beside the direction the
check actually refuses. Over the 2298 passes of the regraded run:

| constraint | below | exact | above | refused before the audit |
|---|---|---|---|---|
| `--time` | 0 | 2298 | 0 | either side |
| `--mem` | 0 | 2268 | 30 | below only |
| `--gpus` | 0 | 343 | 0 | below only |

`--time` reading 2298 exact is the control: the floor holds and introduced no regression in the
other direction. **`--gpus` was closed by
measurement first and in the check afterwards.** Its loose side was a theoretical hole, one task
declares `gpus_min`, and no model in any run over-requested, so the measurement said there was
nothing to correct. It was tightened anyway, once the other two were done: a hole nobody has
fallen into is still a hole, and this one costs a re-verification rather than a regeneration.
All six constraints carrying a value now demand equality.

`--mem` had 30 passes above the declared value, 1.3%, and the shape of them decided it. All 30 are
one model, Qwen2.5-Coder 1.5B, writing `--mem=2G` against the 1024MB that `t1_array_job` and
`t1_output_paths` name, and all 30 are **F3 repairs**. So `--mem` now demands equality, like
`--nodes`, `--ntasks`, `--cpus-per-task` and `--time` before it.

The reason given here at first was that the model had rewritten a directive the induced fault never
touched, the way Gemma invents `--mem-per-node` while repairing an F4. That was wrong, and
[the F3 dig](#what-f3-actually-measures) has the correct account: F3 replaces the `--mem` value with
`2 referencing the requested memory`, so the directive is the broken one, and `2G` is the fault's
own
digit with a unit appended. The tightening is better justified than the reason first given for it,
not worse. An artifact that keeps the digit it was shown is a failed repair, and under a floor-only
check it passed whenever the guess landed above the requirement.

The case was closer than the walltime one and the argument against is worth recording. `mem_min_mb`
is named a minimum, F8 is a whole fault class about requesting *less* memory than the payload uses,
and over-requesting wastes an allocation rather than killing a job. What settles it is that
`resource_fit` is a conformance level and not a liveness one: a script asking two nodes where one
was specified also runs, and is also refused. `functional` is the level that asks whether the job
works.

The measured cost is one cell, and the estimate made before the regrade is worth recording beside
what it produced. Qwen2.5-Coder 1.5B was expected to lose 30 T2 passes, taking its F3 from 0.542 to
about 0.29 and its T2 `strict` from 0.256 to about 0.21. Reverified, F3 reads **0.292** and `strict`
**0.211**, `resource_fit` **0.330** from 0.377, and every other cell on both task sets is identical
to the digit. No ordering moves; the 1.5B is last on T2 by thirty points either way. An audit that
can predict the effect of a fix to three decimals before applying it is the difference between
tightening a check and guessing at one.

The constraint keeps the name `mem_min_mb`, misleading as that now is, because renaming it moves
`tasks_sha` and would invalidate every generation ever measured against it, which is a far larger
price than a stale name.

One thing happened by itself and is worth pointing at. Editing `verifier.py` moved `verifier_sha`
from `74e00ebdcced` to `20dd4a2e4159`, and the next test run failed because the leaderboard page no
longer matched its entries: all ten rows had become *stale rules*. Nothing was remembered, nobody
had to notice. That is the mechanism built four commits earlier doing the exact job it was built
for, on the first change after it landed.

### Which fault is hardest depends on the model, and on the level

Per fault category, `bash` arm, three seeds pooled. F1 applies to three tasks and F6 to one, hence
the smaller denominators.

| category | Qwen2.5-Coder 1.5B | Qwen3.5 9B | Granite 4.1 3B | Gemma 4 12B | Qwen2.5-Coder 7B | n per model |
|---|---|---|---|---|---|---|
| F1 omitted default | 0.000 | 0.667 | 0.356 | 0.667 | 1.000 | 45 |
| F2 directive after the first command | 0.342 | 1.000 | 0.642 | 0.875 | 0.875 | 120 |
| F3 prose in a value | 0.292 | 0.983 | 0.800 | 0.875 | 0.875 | 120 |
| F4 directive absent | 0.000 | 0.242 | 0.383 | 0.750 | 0.750 | 120 |
| F5 no `#SBATCH` at all | 0.050 | 0.383 | 0.250 | 0.375 | 0.658 | 120 |
| F6 payload/spec mismatch | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 15 |
| F7 malformed value | 0.358 | 0.817 | 0.575 | 0.742 | 0.875 | 120 |

Emitted by `scripts/category_table.py` from the reports rather than assembled by hand, which is
how the previous version of it went to press twice with a column missing.

`strict_all_levels` pass@1. **F5 is the lowest category for three of the five models and never
comfortable for any of them**: it is the fault that leaves the artifact furthest from a job script,
and restoring every directive from the prompt alone is closer to writing one than to repairing one.
F6 stays the easy category, 1.000 for everything except the 1.5B, so it separates nothing.

F1 separates the models: 0.000, 0.356, 0.667, 0.667, 1.000 on the fault this document opens with.
A benchmark wants categories like that, and a claim about model capability made on F6 would be
worth nothing.

**F4 splits the five models in two.** It reads 0.750 for the 7B and for Gemma, 0.383 for Granite,
0.242 for Qwen3.5 and 0.000 for the 1.5B: two models at three quarters, three well below half, and
nothing in between. It is the widest spread of any category except F1. For Qwen3.5 it is also its
own hardest category rather than F5, which is true of no other model here. The next two sections
take that number apart, and it does not survive as one finding about one model.

An earlier version of this paragraph read F4 as four models clustered between 0.742 and 0.750 with
Qwen3.5 alone at 0.242, and built a claim about the newest model being the outlier on it. That
cluster was an artifact. Granite's 0.742 came from a grading with no floor on `--time`, and under
the floor it is 0.383, which puts it in the low group rather than the high one. The retraction is
kept here because the pattern it describes, one model isolated against a tight group, is exactly
the shape a missing check produces, and it read as a finding for a day.

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

Second, the gap to the 7B is concentrated in two categories for one model and spread across five
for another, and the difference between those two cases is worth more than the pattern it replaces.
Gemma at 12B matches the 7B exactly on F2, F3 and F4, trails it by 13 points on F7, and the rest of
the distance between 0.726 and 0.824 is F1 by 33 points and F5 by 28. Granite at 3B trails on
everything except F6: F1 by 64, F5 by 41, F4 by 37, F7 by 30, F2 by 23. Its 0.529 is not two
categories of weakness against an otherwise matched profile, it is a weaker profile.

That distinction is the one the regrade produced. Before the walltime floor, Granite read 0.742 on
F4, 0.833 on F7 and 0.750 on F2, which put it within twelve points of the 7B on four categories and
made it look like the same two-category story as Gemma. Three of those four numbers were inflated
by artifacts requesting seconds where the prompt named minutes. A shared pattern across two
families was the more interesting claim and it was the false one.

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

Gemma 4 12B scores 0.750 on F4, the joint highest of the five, and its failures have nothing to
do with `--time`. It restores the removed directive on all eight tasks, 120 artifacts
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

### A task set that depended on how fast the scheduler started

CI failed with four tests down and one of them explaining the rest: `tasks/t2_repair.jsonl` read
stale, with `t1_hello_serial__F1` and `t1_container_apptainer__F1` as *extra* items. Regenerating
the file on that machine produced faults that the committed file does not have.

`induce_t2_tasks` keeps an induced variant only when the verifier refuses it, which is the right
rule: an inducer that produces an accidentally valid script has induced nothing. But
`submittability`
is one of the levels doing the refusing, and on that runner `sbatch --test-only` was answering
`Requested node configuration is not available` for the canonical `t1_hello_serial`. Dropping
`--ntasks=1` from that task leaves the effective count at 1 and `resource_fit` passes it, so the
variant is normally discarded; with the scheduler refusing everything it was kept instead. **A
controller that was still starting changed which faults the benchmark contains.**

The container's entrypoint waited for the scheduler with two fixed `sleep 3`. That is an assumption
about how fast a machine is, and the assumption is what failed. It now waits for `scontrol ping` to
answer and for a node to leave the down state, up to sixty seconds, and says so loudly if it never
does.

A first attempt at that fix probed readiness with `sbatch --test-only`, which broke every
dependency task at once: a test-only submission still takes a job id, and the entrypoint relies on
`FirstJobId=12345` plus a placeholder landing on exactly 12345 so that `--dependency=afterok:12345`
resolves. The probe consumed the id, the placeholder moved, and `t1_dependency_chain` began failing
with `Job dependency problem`.

Waiting on the cheap signals alone was not enough either: the next run came back with two failures
instead of four, intermittently inside a single container, which says the controller answers
`scontrol ping` and reports its nodes up while still refusing the first job it is asked to place.
The probe that settles it has to be the call every level makes, so it is now made **after** the
placeholder has taken 12345, where the ids it consumes are 12346 upward and nothing depends on
them. Two readiness checks that cost nothing and one that costs a job id nobody needs.

The deeper fix is not the timing. `anvil induce` now **refuses to run** where `submittability`
cannot be judged, naming the reason, and `make induce-t2` builds the set inside the container the
way `make induce-exec` already did. Without that, running it on a laptop with no scheduler keeps
every variant, since a skipped level is never a passed one: the file would be larger, silently, and
would still carry a digest and look authoritative. A task set is the definition of the benchmark,
so this is a refusal rather than a warning.

A footnote worth keeping, because it is the only time today a defence paid for itself somewhere
other than where it was built. The readiness wait added here is what stopped a Podman run from
grading: under rootless Podman `slurmd` cannot create its step scope, no job is ever placeable, and
the container reported `a placeable job did not become ready in 60s` and refused to continue rather
than producing a table with `submittability` quietly unreliable. The check was written for a CI
race on a loaded runner and caught an unrelated limitation of a different container runtime, which
is what a check on a precondition does and an assumption about timing does not.

None of that fixed it. The third run came back with three failures again, on a third disjoint set
of tests, all of them the same message. A defect that moves to different tests each time is not a
startup race being narrowly missed, it is a condition recurring throughout the run: on a four-core
runner hosting four `slurmd` daemons and a test suite, a node drops out of service for a moment and
`sbatch --test-only` answers an ordinary job with `Requested node configuration is not available`.

So the fix belongs in the verifier rather than in the entrypoint. The first attempt there was to
retry a refusal that describes the cluster's state rather than the script, three attempts a second
apart. It made things worse in an instructive way: the next run took **324 seconds instead of 16**
and still failed four tests. Hundreds of calls were being refused and recovering, which is not the
occasional flap the retry was written for, and the number of attempts was the wrong lever.

What decides is not how many times the same question is asked but whether the scheduler can answer
any question at all, and the project already has that rule at a coarser grain: `slurm_healthy`
refuses to score a scheduler that cannot judge, because those numbers would be the harness and not
the model. So a cluster-state refusal that survives one retry now **re-runs the preflight**. The
canary is a script this scheduler must accept; if it is refused too, nothing is being judged and the
level is **skipped with the reason**, exactly as when no scheduler is reachable. If the canary
passes, the refusal really is about this script and it fails.

Skipped is never passed, so nothing is masked: a level that cannot decide contributes nothing to
`strict_all_levels` and says so in the report. And a refusal that names the artifact, an invalid
partition, an unsatisfiable memory specification, an option SLURM does not have, is never retried at
all, so the common path costs one call as before.

What none of this does is prove the CI failure fixed. It was never reproduced locally, on either
machine available here, and a resource-starvation flap on someone else's runner rarely is. What can
be said is that the fixed sleeps were an assumption and are now checks, that the last probe tests
the exact call that was failing rather than a proxy for it, and that a level now reports on the
artifact or reports nothing, which is the property it was supposed to have all along.

### Quantization moves three levels and leaves the fourth alone

Every figure this project publishes was measured with the model loaded in 4 bit, and nothing said
whether that choice carried any of the result. Qwen2.5-Coder 1.5B run again at fp16, three seeds,
n=5, same tasks, graded in the same container under the `bash` executor:

| level | 4-bit | fp16 |
|---|---|---|
| `syntax` | 0.575±0.025 | 0.750±0.000 |
| `submittability` | 0.842±0.013 | 1.000±0.000 |
| `resource_fit` | 0.442±0.013 | 0.617±0.013 |
| `functional` | 0.533±0.037 | 0.533±0.025 |
| `strict_all_levels` | 0.308±0.025 | 0.408±0.025 |

Both arms are graded by `d4af1eaf9809`. They were not at first: the 4-bit figures came from
`20dd4a2e4159`, and between the two verifiers `functional` gained the sandbox ceiling and the
session reaping while `submittability` gained the retry and the skip. Putting them side by side
without saying so was a mistake, and it was the leaderboard that caught it, marking 34 rows *stale
rules* next to one fresh one. Rather than argue the four changes were inert, the whole run was
reverified: **every cell of all five models on both task sets came back identical to the digit**,
and so did the executor comparison, 6 changed verdicts of 3900 and the same 245 and 32. Four
changes to the verifier, zero verdicts moved, measured rather than assumed.

Three levels move by fifteen to eighteen points. **`functional` does not move at all**, the same
0.533 on both, with overlapping ranges. Quantizing this model costs it the form of the artifact,
valid shell, submittable directives, values that match the request, and costs it nothing measurable
in whether the payload does what the task asked.

The reading needs one qualification, because `functional` is not independent of `syntax`: a script
that fails to parse is recorded as failing `functional` without being run. So fp16 sends more
scripts to the executor, 0.750 of them against 0.575, and the pass rate stays where it was, which
means the extra scripts that now parse fail at execution more often than the ones that already did.
The two levels are not measuring the same thing, and the invariance is a real observation rather
than an artifact of the ordering.

`submittability` at 1.000 is worth its own line. That level [does not track model
size](#submittability-does-not-track-model-size), which is one of this project's stranger findings,
and here it tracks how the model was loaded, sharply: at fp16 the 1.5B writes a submittable script
every time out of 120.

One model. The 7B does not fit in fp16 on a 12GB card, so this is not a statement about the table,
and the table stays 4-bit throughout with the condition recorded in every entry. What it does
establish is that the condition is load-bearing for the smallest model, by ten points of strict, and
that a leaderboard row without its quantization would be a number about how a model was loaded as
much as about the model. Both arms are published, which is why the entry key now carries it.

### Two spellings of one request, one of them refused

`--nodes=2 --ntasks=4` and `--nodes=2 --ntasks-per-node=2` ask SLURM for the same four tasks.
`check_resource_fit` read only the first: `--ntasks-per-node` was absent from the effective-request
computation, so the second was judged as two tasks and refused. A correct artifact failed for
writing the request in the other spelling, which is the false negative the level was built to avoid
and the one the section it lives under names as the reason for computing effective values at all.

It surfaced from the other direction. Counting the directives that passing artifacts carry and no
task demands, over the same 11774 that settled the unchecked values, put `--ntasks-per-node` at 157
occurrences across three models. Every one of the 157 also wrote `--ntasks`, which wins in SLURM and
wins here, so no published verdict rested on the wrong count: the defect was reachable and never
reached. What made it visible was asking what artifacts carry, rather than what they omit.

The fix is in the verifier and not in the tasks, like the walltime and memory bounds before it, so
it costs a re-verification rather than a regeneration. It does move `verifier_sha`, and every entry
measured under the previous digest reads *stale rules* until the matrix is verified again.

### The same artifact, verified twice, two answers

Verifying one cell twice with the same generations, the same verifier and the same image
returned `functional` 0.85 and then 0.875. One sample in forty had changed its mind. The
level that moved is the one that runs the payload, and it moved without any of the four
static levels moving, which is why it left `strict_all_levels` untouched and went unnoticed
across earlier regrades: a per-level number drifted while the headline did not.

Diffing the two reports level by level named the artifact:

```bash
exec > >(tee "logs/out_$$.txt")
exec 2> >(tee "logs/err_$$.txt")
echo ANVIL_OK
```

Process substitution puts a `tee` between the payload and the harness, and no `&` appears
anywhere in the script. The sandbox kills the session as soon as the shell exits, then reads
the captured files, so whatever `tee` had not yet written was gone. Under a real scheduler
the job completes and its output arrives, so failing it was a false negative, intermittent
because it is a race.

The reaping itself is not the mistake: without it a matrix fills the host with orphans, and
that is a failure this project has already had. What was missing is the distinction between
a process that is *finishing* and one that is *working*. The sandbox now waits up to half a
second for the session to empty and then kills it regardless, which lets an exiting `tee`
flush and still stops a background worker with its work unfinished. The test asserting that
a child sleeping two seconds never completes is what fixes that duration: a grace long
enough to let it through would have traded one defect for the other.

Two things this changes about reading any number here. A single grading is not sufficient
evidence for a `functional` figure, since the level is the only one that executes and the
only one that can disagree with itself; and `verifier_sha` bounds which rules produced a
verdict without bounding the verdict, so two gradings under one digest can still differ. The
practical consequence is that a cell worth publishing is worth verifying twice, and
`./scripts/regrade_diff.py results/first results/second` is what reads the second answer:
per level, not per strict verdict, which is the distinction that kept this hidden.

Verifying twice then found a second one, in the other executor, and each further grading
changed what it looked like. Across four gradings of the execution matrix under real
submission the OUT_OF_MEMORY count went 358, 357, 356, 357, 356. Three points read as a
drift and this section said so, correcting an earlier reading that had called it jitter; the
fourth and fifth say it oscillates by one, and the second reading was as wrong as the first.
Three points are not enough to name a shape, which is the part of this worth keeping. The
grace above cannot account for any of it, being reached only from the bash path.

`regrade_diff.py` names the artifact behind the last step: `t1_memory_workers__F8`, sample
45 of the 9B at seed 2, job 12440, OUT_OF_MEMORY in one grading and COMPLETED with the
expected output in the next. The mechanism is the one the set exists to exercise, arrived at
from the least convenient direction: F8 is the class that only an enforced allocation can
refuse, `t1_memory_workers` is the task whose cost is four concurrent workers rather than
one buffer, and a repair of that pair sits against the cgroup limit by construction. Whether
the same artifact moved at each step or three different ones did is not established; one
pair was compared, and the answer for the other two is a comparison away.

The fourth grading also moved a cell in the main matrix, which had reproduced exactly three
times running: one artifact under real submission came back `COMPLETED but wrote nothing`.
That one is a defect and is fixed. `scontrol` reports COMPLETED when the job is done, not
when `slurmstepd` has closed its output file, so reading at once can find no file or a
partial one, and the executor now waits up to two seconds for what the task expects before
concluding the job wrote nothing. It is the sandbox race again, on the other executor, found
because a regrade for an unrelated change happened to run the matrix a fourth time.

The regrade after that fix is the check on it. One level moved, `t1_output_paths` on granite
at seed 2 under real submission, from `COMPLETED but wrote nothing to slurm-%j.out` to the
expected output present, and every other figure in the main matrix came back to the value it
had held for three gradings before the false negative appeared. A fix whose effect is one
artifact and no collateral is what the per-level comparison was built to be able to say.

The OUT_OF_MEMORY oscillation is not that, and may not be a defect at all: a real scheduler
would answer the same way, and an artifact that close to its allocation is the thing being
measured. What it does mean is
that this cell is worth reading as plus or minus one artifact, and that six thousandths
there is not a difference between models.

## What to change when the T2 set is regenerated

`tasks/t2_repair.jsonl` is the denominator of every T2 number published here, so nothing is
regenerated for one improvement at a time: a new digest means the whole matrix has to be generated
again with all five models, which is days of GPU rather than a re-verification. Five separate
findings are waiting on that one pass, four of them recorded in four different sections above and
the fifth in `DESIGN.md`, where the choice it follows from is argued. This is the list, in one
place, so that the pass does not have to be repeated because one of them was missed.

1. **Register F9 as an inducer.** An option this scheduler does not have, `--walltime` from Granite
   and `--mem-per-node` from Gemma, both on the multi-node task. Two families reach it by different
   routes, so it is not one model's quirk. See [F9](#f9-an-option-this-scheduler-does-not-have).
2. **Register F10 as an inducer**, and first among the four. A unit confusion the scheduler accepts
   is the only observed class that no level except `resource_fit` can see: well formed, submitted
   without complaint, runs and prints what was asked. Its repair cannot be faked by any check
   cheaper than the one this benchmark runs, which is exactly what a repair set should be made of.
   See [F10](#f10-a-unit-confusion-the-scheduler-accepts).
3. **Give F3 a per-task value.** The inducer writes the literal `2` on all eight tasks, and 2GB is
   the correct answer on two of them, so a model that blindly appends `G` to the digit it was shown
   passes a quarter of the category. A value that varies with the task separates repairing from
   copying; a constant cannot. See [What F3 actually
   measures](#what-f3-actually-measures).
4. **Decide what F4 should exercise.** It is written to drop `--time`, `--mem` or `--gpus`,
   whichever the task declares, in that order, and every T1 task declares a walltime, so the first
   candidate always applies and the other two are never reached. As instantiated the category is
   about one directive. That may be the right scope, but it should be a decision rather than a
   consequence of the ordering.
5. **Decide what verifies the values a prompt names.** `required_directives` asks whether a
   directive is written, never what it says, which is the deliberate refusal of surface-form
   matching argued in [`DESIGN.md`](DESIGN.md#what-that-choice-leaves-unchecked); the measured
   consequence is that four T1 tasks accept a script that does not do what their prompt asked.
   The sharpest is `t1_array_job`: it declares `"array": true`, a boolean, so `--array=1-1`, one
   task where the prompt asks for five, passes every level strictly, and so does `--array=0-99`.
   Neither executor closes it, the sandbox simulating a single `SLURM_ARRAY_TASK_ID` and real
   submission accepting the array and completing it, both checked. Closing any of the four means
   execution rather than matching, and an edit to `tasks/t1_slurm.jsonl`, which is what puts this
   here. The shape it should take is already in the repository: T3, written later, compares
   `Bootstrap` and `From` by value in `resource_fit` and leaves each section's substance to what
   the container prints, so `%environment` is judged by `GREETING=hello` appearing and not by the
   section being present. How often the opening was taken is now measured, by
   `./scripts/unchecked_values.py 'results/*/*__bash.json'` over the full matrix: on 11774
   passing artifacts, all ten values were written exactly as their prompt names them, on every
   task and every model. The only apparent exception was 229 artifacts writing `--array=1-5%5`
   or `1-5%1`, all from one 7B model, and the `%N` there caps how many array tasks run at once
   and leaves the five indices alone, so the audit was wrong and not the artifacts. The opening
   is therefore real and unexploited, which lowers its priority against the four items above
   without closing it: five models writing the obvious thing is not a property of the check.

Two things that are *not* on this list, deliberately. The execution-sensitive set has its own file
and its own digest, so `tasks/t1_exec.jsonl` and `tasks/t2_exec_repair.jsonl` can grow without
touching any of this. And the walltime and memory bounds were fixed in the verifier rather than in
the tasks, as was the effective count above, which is why those corrections cost a re-verification
and not a re-generation: a constraint whose name is now misleading, `mem_min_mb` demanding equality,
is a smaller price than moving `tasks_sha`.

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
- the F3 inducer's constant value. The category is measured and understood, see [What F3 actually
  measures](#what-f3-actually-measures), and the finding is that its literal `2` is the right answer
  in gigabytes on two of the eight tasks, so appending a unit blindly passes a quarter of the
  category. A per-task value would separate repairing from copying. It regenerates
  `tasks/t2_repair.jsonl`, so it waits for the same pass as the F9 and F10 inducers;
- F8 beyond one model. The task set is now two tasks that under-spend for different reasons,
  a pipeline's peak and a directive's scope, both verified to be invisible to every static check
  and to the `bash` executor, see [A second execution
  task](#a-second-execution-task-and-why-it-is-not-the-same-one-twice). What is missing is the
  models: the only observation is still 15 samples of the 1.5B, so whether larger models leave
  headroom, and whether the concurrency case is harder than the pipeline one, is unmeasured;
- the toolchain-sensitive task on a model that can reach the question. `tasks/t1_coreutils.jsonl`
  splits the two implementations by construction, and both Qwen sizes fail it on both images for
  reasons that have nothing to do with either: a here-string newline at 1.5B, a missing output
  prefix at 7B. Thirty samples, no divergence, and not one pins a locale, see [A task that can tell
  the two apart](DESIGN.md#a-task-that-can-tell-the-two-apart). What would answer it is a model
  that clears the payload, which neither of these does;

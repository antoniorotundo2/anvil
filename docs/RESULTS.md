# Results

Every measured number this project stands behind, on one page, so it can be read and cited without
running anything. The reasoning behind each is in [DESIGN.md](DESIGN.md) and
[OBSERVED_FAILURES.md](OBSERVED_FAILURES.md); this page is the numbers and their provenance.

Everything here was graded by verifier `81106f3da7a1`, which every report and every leaderboard
entry records. This checkout carries `89a70295ce84`, so every row is marked *stale rules* until the
generations are graded again. What moved is `safety`, which refused `rm -rf /` and let `rm -rf /*`
through, along with four other spellings of hazards already on its list. No artifact in this
repository trips the widened patterns and none of the figures below was measured on one that does,
so they are expected to survive; the regrade is what settles it.

The last change to reach the verifier closed the last one-sided comparison: `--gpus` now demands
the count the task declares, where it accepted any number at or above it. Its loose side had been
measured before it was closed, 343 passes and every one exact, and the regrade bore that out: every
figure on this page came back identical across 3900 sample comparisons, as it had for the two
changes before it. Three times the prediction held, which is a reason to keep checking rather than
to start assuming.

The execution set is the exception, and not because of this change. One artifact there,
`t1_memory_workers__F8` on the 9B, has come back OUT_OF_MEMORY in one grading and COMPLETED in the
next across three consecutive regrades, moving that cell by one artifact each time. It is a repair
of a memory under-request sitting against the allocation it asked for, which is what the set exists
to exercise, so the number to read there is plus or minus one artifact. The detail is in
[`OBSERVED_FAILURES.md`](OBSERVED_FAILURES.md#the-same-artifact-verified-twice-two-answers).

Three changes to the verifier have landed since the figures were first measured, and all three
have now been answered by regrading the same generations rather than by expectation. A ceiling on
the `bash` sandbox, machine protection rather than a rule about artifacts, added after a generated
script with an unbounded allocation took a host down. `--ntasks-per-node` counted toward the
effective request, which it had not been, so `--nodes=2 --ntasks-per-node=2` was read as two tasks
where SLURM makes four. And a bounded wait before the sandbox reaps what a script left running,
after an artifact writing through `tee` was failed for output the harness had discarded.

The regrade moved nothing: every figure on this page came back identical across 3900 sample
comparisons. That is the useful outcome and not a foregone one, since the second and third changes
can both move a verdict; what it says is that nothing published here rested on the defects they
fixed.

One qualification the regrade produced, which applies to every `functional` figure below.
Verifying one cell twice, under one verifier and one image, returned two different numbers: the
level that executes is the only one that can disagree with itself, and `verifier_sha` bounds which
rules produced a verdict without bounding the verdict. The instance behind it is fixed, and the
property is not a claim that it was the only one. Figures here come from a single grading unless
stated; `./scripts/regrade_diff.py` compares a second, and
[the section on it](OBSERVED_FAILURES.md#the-same-artifact-verified-twice-two-answers) has the
detail.

Two earlier gradings of the same generations are superseded, both of them for a one-sided comparison
in `check_resource_fit`. The first had no floor on `--time`: adding one moved five of the ten cells
and swapped two models on T1, see [The mirror of
F10](OBSERVED_FAILURES.md#the-mirror-of-f10-which-this-verifier-did-not-catch). The second had no
ceiling on `--mem`, found by auditing the remaining checks on purpose rather than by accident, and
moved one cell, see [What the audit
settled](OBSERVED_FAILURES.md#what-the-audit-settled).

## How to read them

`pass@1` with the unbiased estimator (Chen et al., 2021), per level, plus `strict_all_levels`,
which requires every level either to pass or to be out of the machine's reach. A skipped level is
never a passed one.

The `±` is **half the range across three seeds, not a confidence interval**. Three draws say
whether an effect survives reseeding; they do not support a significance claim, and none is made
here. Where two arms differ by less than their ranges, this page says they do not separate.

## Environment

All figures below were graded inside the container: Ubuntu 24.04, GNU coreutils 9.4, bash 5.2, a
live `slurmctld` on the declared reference topology. Generation ran on the experiment machine, an
RTX 3060 12 GB, every model in 4-bit NF4. Generating where the accelerator is and grading where the
scheduler is is a deliberate split, not a convenience: see
[HARDWARE.md](HARDWARE.md).

Numbers produced outside the container are not on this page. One earlier table was, and it was
wrong; that correction is recorded in
[A table measured against the wrong
cluster](OBSERVED_FAILURES.md#a-table-measured-against-the-wrong-cluster).

## T1: writing a job script from scratch

Eight tasks, 3 seeds (0/1/2), n=5, five models across three families and two Qwen generations.

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.575±0.025 | 0.842±0.013 | 0.533±0.037 | 0.375±0.025 | 0.442±0.013 | 0.308±0.025 |
| Qwen2.5-Coder-7B-Instruct | 1.000±0.000 | 0.792±0.025 | 0.875±0.000 | 0.667±0.025 | 1.000±0.000 | 0.667±0.025 |
| granite-4.1-3b | 1.000±0.000 | 0.875±0.000 | 0.842±0.037 | 0.717±0.037 | 0.550±0.000 | 0.425±0.000 |
| gemma-4-12B-it | 0.875±0.000 | 0.867±0.013 | 0.875±0.000 | 0.742±0.013 | 0.917±0.050 | 0.658±0.062 |
| Qwen3.5-9B | 1.000±0.000 | 0.875±0.025 | 0.650±0.025 | 0.658±0.013 | 0.600±0.050 | 0.450±0.025 |

`safety` is 1.000±0.000 everywhere and is left out. Qwen3.5-9B is the only model whose `strict`
differs between the two executors, 0.450 against 0.475, and the reason is the subject of the second
finding below. It is also the only one measured with `--disable-thinking`: it reasons by default,
and under this benchmark's token budget it would be cut off mid-thought and never reach the code
block, so every sample would fail `syntax` for a reason belonging to the harness.

## T2: diagnosing and repairing a broken one

Induced faults from the same eight tasks, same protocol, 220 repairs per seed.

| model | syntax | submittability | functional (bash) | functional (sbatch) | resource_fit | strict |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.792±0.016 | 0.886±0.005 | 0.664±0.005 | 0.595±0.009 | 0.330±0.009 | 0.211±0.007 |
| Qwen2.5-Coder-7B-Instruct | 0.983±0.002 | 0.977±0.000 | 0.870±0.002 | 0.847±0.002 | 0.965±0.007 | 0.824±0.002 |
| granite-4.1-3b | 0.862±0.002 | 1.000±0.000 | 0.750±0.000 | 0.750±0.000 | 0.621±0.016 | 0.529±0.016 |
| gemma-4-12B-it | 1.000±0.000 | 0.876±0.002 | 0.885±0.002 | 0.762±0.002 | 0.932±0.005 | 0.726±0.002 |
| Qwen3.5-9B | 0.982±0.005 | 0.982±0.005 | 0.947±0.011 | 0.950±0.009 | 0.711±0.009 | 0.691±0.009 |

Per fault category, `strict_all_levels`, three seeds pooled. F1 applies to three tasks and F6 to
one, hence the smaller denominators.

| category | Qwen2.5-Coder 1.5B | Qwen3.5 9B | Granite 4.1 3B | Gemma 4 12B | Qwen2.5-Coder 7B | n per model |
|---|---|---|---|---|---|---|
| F1 omitted default | 0.000 | 0.667 | 0.356 | 0.667 | 1.000 | 45 |
| F2 directive after the first command | 0.342 | 1.000 | 0.642 | 0.875 | 0.875 | 120 |
| F3 prose in a value | 0.292 | 0.983 | 0.800 | 0.875 | 0.875 | 120 |
| F4 directive absent | 0.000 | 0.242 | 0.383 | 0.750 | 0.750 | 120 |
| F5 no `#SBATCH` at all | 0.050 | 0.383 | 0.250 | 0.375 | 0.658 | 120 |
| F6 payload/spec mismatch | 0.933 | 1.000 | 1.000 | 1.000 | 1.000 | 15 |
| F7 malformed value | 0.358 | 0.817 | 0.575 | 0.742 | 0.875 | 120 |

## Eight findings

**`submittability` is not ordered by model size.** T1 reads 0.875 for Granite at 3B, 0.867 for
Gemma at 12B, 0.842 for Qwen at 1.5B and 0.792 for Qwen at 7B: the largest model in the set and the
smallest of another family sit together at the top, the 7B is alone at the bottom, and inside the
Qwen family the level falls as size rises while every other level improves. Two families of three
are above Qwen, from 3B to 12B, so the shortfall is not one model's quirk. T2 does not repeat the
ordering, where every model sits between 0.876 and 1.000 and Gemma is last.

The families also fail the level differently. Qwen invents partition names, which a different
cluster might genuinely have; Granite invents an option SLURM does not have, which no cluster has.
See [`submittability` does not track model
size](OBSERVED_FAILURES.md#submittability-does-not-track-model-size).

**The sandbox was almost redundant until a model wrote `srun`.** `scripts/executor_ablation.sh`
grades the same generations twice in the same image, once through the `bash` sandbox and once
through real `sbatch`. With four models the count was one artifact of 3120. With a fifth it is
**six of 3900, and 32 samples that fail under `bash` and pass under real submission**. Five of the
six are the same mechanism: the model writes `srun` inside the script, the sandbox has no
allocation for it to attach to, and the step exits non-zero. **The sandbox produces a false
negative on every script that launches a job step**, and did so silently while no model wrote one.

Both artifacts the executors disagree about point that way: neither is the scheduler catching
something the sandbox missed, both are the sandbox rejecting a script the scheduler accepts. The
earlier claim that real submission was nearly redundant was true of the models measured, not of the
method. See [Real submission was almost redundant until a model wrote
`srun`](OBSERVED_FAILURES.md#real-submission-was-almost-redundant-until-a-model-wrote-srun).

**The executor's value is a property of the task set, and it ranges over two orders of magnitude.**
The 0.15% above is measured on tasks whose resource requirements are written in their prompts.
`tasks/t1_exec.jsonl` states no memory minimum, so what a script needs is a property of the payload
it wrote itself and only execution knows it. Graded the same way, five models and three seeds, the
same comparison reads **297 artifacts of 900, 33%**, with 356 of the 394 stopped artifacts coming
back `OUT_OF_MEMORY`. Qwen2.5-Coder 7B goes from 0.967 to **0.000** on `functional`: the sandbox
promotes 29 of its 30 from-scratch artifacts and the scheduler kills every one. Gemma 4 12B is the
only model that solves the set, 1.000 under both arms. See [On a task set where only execution knows
the
answer](OBSERVED_FAILURES.md#on-a-task-set-where-only-execution-knows-the-answer-a-third-of-the-verdicts-change).
Two T1 tasks and ten repairs: a small set, reported as such, and not comparable with the
3900-sample figures above.

**Retrieval costs this model family, and the sign depends on what the model already knows.** Seven
conditions across two sizes. Any attached text costs the 1.5B about 28 points of `resource_fit`,
whether it is SLURM documentation or a passage about coastal tides; the 7B pays no such toll and
two off-topic controls leave it on its zero-shot values to the digit. The one document that
addresses `--time` and `--mem` is worth +21 points to the model that does not know the rule and
-6 to the one that does. See [The intervention
series](DESIGN.md#the-intervention-series).

**A model asked for 45 minutes requested 45 hours, and only one level noticed.** Qwen3.5-9B's
worst per-category score, F4 at 0.242, is entirely `--time`, and thirty-one of its ninety-one
failures read `#SBATCH --time=45:00:00` where the prompt asked for forty-five minutes. The integer
is right and the field is wrong: SLURM reads `hours:minutes:seconds`, so the artifact requests
sixty times the walltime it needs. The same slip appears in 34 of the model's 120 from-scratch
artifacts, and in 10 of Gemma 4 12B's, so it is a habit shared by two families and not a
repair-time accident. It is invisible to four of the five levels: the script is well formed,
`sbatch` accepts it silently, the job runs and prints what was asked. `resource_fit` is the only
thing between it and a queue, and no static check the project compares itself against would report
it. The remaining sixty failures are the opposite defect, a repaired script that comes back with
`--time` still missing, on four tasks where the same model writes it correctly from scratch. See
[F10](OBSERVED_FAILURES.md#f10-a-unit-confusion-the-scheduler-accepts).

**A per-category score does not measure whether the induced fault was repaired.** Gemma 4 12B
scores 0.750 on the same category and never once gets the walltime wrong: it restores the removed
directive correctly in all 120 artifacts. Its thirty failures are two tasks it also fails 15 times
out of 15 when writing them from scratch, so the missing 0.250 measures its ability to produce
those two artifacts at all and not anything about F4. Where that ability is at floor the category
number carries no information about the category, and the 0.242 and 0.750 cells are not two points
on one scale. See [The right number in the wrong
field](OBSERVED_FAILURES.md#the-right-number-in-the-wrong-field).

**A missing floor in the verifier reordered two models, and only the regrade showed it.**
`check_resource_fit` compared `--time` against the walltime a task declares from above only, so
`#SBATCH --time=00:15` against a task naming 15 minutes was fifteen seconds and passed. 123 of 2421
passing artifacts were requests like it. Regrading the saved generations under the floor moved five
of the ten cells above, left the other five identical to the digit, and dropped Granite 4.1 3B on
T1 `strict` from 0.500 to 0.425, **below Qwen3.5-9B at 0.450**. A one-sided comparison does not
inflate a table evenly: it inflates whichever model has the habit the check cannot see, which is a
reordering rather than an offset. Every report and entry now records a `verifier_sha` beside
`tasks_sha`, so two gradings of one set of generations can no longer read as one series. See [The
mirror of F10](OBSERVED_FAILURES.md#the-mirror-of-f10-which-this-verifier-did-not-catch).

**Loading the model in 4 bit costs three levels and leaves the fourth untouched.** Every figure
here was measured at 4 bit, and nothing said whether that carried any of the result. Qwen2.5-Coder
1.5B run again at fp16, same tasks, same seeds, same container, gains 17 points of `syntax`, 16 of
`submittability` and 17 of `resource_fit`, and 10 of `strict`. **`functional` does not move**, 0.533
on both with overlapping ranges. Quantizing this model costs it the form of the artifact and costs
it nothing measurable in whether the payload does what was asked. One model: only the 1.5B fits in
fp16 on a 12GB card, so this qualifies the table rather than restating it, and both arms are
published because the loading is part of what a row measures. See [Quantization moves three
levels](OBSERVED_FAILURES.md#quantization-moves-three-levels-and-leaves-the-fourth-alone).

**Two coreutils implementations are not interchangeable, and no model has reached the difference.**
101 invocations run in Ubuntu 24.04 (GNU 9.4) and 26.04 (`uutils` 0.8.0): 91 agree exactly. Of the
rest, three are behavioural and need no misconfigured locale, `wc -m` and `expand` on a non-ASCII
payload and `numfmt --to=si`. `tasks/t1_coreutils.jsonl` sits in that corner deliberately and
`make docker-guards-coreutils` proves the same script gets two verdicts. Asked to solve it, three
models produced 45 samples, **none pinning a locale and none diverging**: each fails on both
implementations for a reason that precedes the difference.

## Reproducing

Generation needs an accelerator, grading needs the container, and the two are separate commands on
purpose.

```
MODELS="Qwen/Qwen2.5-Coder-7B-Instruct" SEEDS="0 1 2" N=5 ./scripts/run_experiments.sh
./scripts/executor_ablation.sh results/<the directory it printed>
```

The brackets that must hold before any of the above means anything:

```
make guards && make guards-t2 && make docker-guards-enforcement && make docker-guards-coreutils
```

## What these numbers are not

Three seeds and 24 to 220 verifications per cell. Three model families at four sizes, from 1.5B to
12B, quantized to 4 bit except where a row says otherwise. Eight T1 tasks, which is a small
denominator and makes each task worth 0.125 of every T1 figure on this page. One reference
topology, declared rather than borrowed
from a real centre. Nothing here has been replicated by anybody else.

Every figure moved at least once while being measured, and the corrections are recorded next to the
findings rather than folded away. That is the reason to trust the current values, not a reason to
treat them as settled.

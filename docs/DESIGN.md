# Design

Why this benchmark exists, and why it is built the way it is.

## The question

*When an LLM writes the SLURM job script a supercomputer user actually needs, is it correct?*

Not "does it resemble the reference answer": **does it parse, does the scheduler accept it, does
it run, and does it request the right resources?**

Assistants that help HPC users are appearing, but they are evaluated with semantic similarity
metrics because no validated HPC benchmark exists, as their own authors say. Meanwhile,
execution-based benchmarks are the norm everywhere else: parallel code, PDE solvers, quantum SDKs,
and cloud infrastructure-as-code. **Operational HPC artifacts are the empty slot.**

Cloud IaC benchmarks fall back on graph-edit-distance and LLM judges because a Terraform plan
cannot be executed cheaply or repaired iteratively. HPC operational artifacts are the opposite:
`sbatch --test-only` costs milliseconds, the job really runs, and repair is iterable. Objective,
execution-based ground truth is *possible* here where it is not possible there.

A wrong `--mem` does not look wrong. It looks plausible. Then the job dies at hour six.

## The verifier

Five independent levels, weakest to strongest:

| Level | Question | How |
|---|---|---|
| `syntax` | Is it a valid script? | shebang, `bash -n`, misplaced `#SBATCH` |
| `submittability` | Would SLURM accept it? | `sbatch --test-only` |
| `functional` | Does it run and exit 0? | `bash` sandbox, or real `sbatch` submission |
| `resource_fit` | Does it request what was asked? | effective request vs. task constraints |
| `safety` | Is it dangerous? | destructive-pattern probes |

Two design choices carry the scientific weight.

**`skipped` is never `passed`.** No scheduler on your laptop? `submittability` is skipped and
scored as *not passed*. The metrics stay honest on any machine.

**Dangerous scripts are never executed.** `safety` gates `functional`.

### The misplaced-directive check

SLURM stops reading `#SBATCH` lines at the first real command. Directives after it are **silently
ignored**: `sbatch` accepts the job and the request is wrong. Anvil catches this;
`sbatch --test-only` cannot.

### Effective requests, not string presence

`resource_fit` compares the **effective** resource request against the spec, applying SLURM's
documented defaults: `--nodes` → 1, `--ntasks` → one task per node, `--cpus-per-task` → 1. A serial
script that omits `--nodes` still requests one node, and is correct.

Directives with no universal default (`--time`, `--mem`, `--gpus`) depend on partition
configuration. Omitting them means the resource was never requested: a genuine failure against a
spec that asks for it. Tasks can still demand explicitness through `required_directives`.

The distinction is the point. Checking whether a string appears is surface-form matching,
precisely what this benchmark exists to replace. An early version did exactly that, and failed
scripts that `sbatch` accepted.

## Oracle and broken model

Every benchmark should ship both. Few do.

- **Oracle**: canonical solutions. Proves the tasks are solvable and the verifier is not too
  strict. CI fails if it drops below 1.0.
- **Broken**: faulty artifacts (missing shebang, misplaced directive, walltime overrun,
  `rm -rf /`, non-zero exit). Proves the verifier is not too permissive. CI fails if it scores
  above 0.0 strict, **or if the safety guard is never exercised**.

Together they bracket the verifier from both sides. Neither test is decorative: the oracle caught
a real bug during development, where the harness injected `SLURM_CPUS_PER_TASK=1` into a task that
requested 4 cores: the harness was contradicting the spec it was checking.

An earlier broken model sampled the same three flavours for every task, so the destructive one was
never drawn and `check_safety` was never tested. A guard that never fires is decoration.

## The preflight

Before scoring anything, the verifier submits a **canary** to the scheduler: a minimal,
certainly-valid script. If the canary fails, `submittability` is marked *skipped*, not *failed*.

Without this, a misconfigured cluster produces eight zeros that are **indistinguishable from a
terrible model**. It happened during development. A benchmark that executes code must be able to
tell a failure of its subject from a failure of itself.

## Metrics

`pass@k` with the unbiased estimator (Chen et al., 2021), computed per level, plus
`strict_all_levels` (every non-skipped level passed).

## Diagnose-and-repair (T2)

T1 measures whether a model can write a correct artifact from scratch. T2 measures whether it can
recognise and fix a broken one. That is a distinct situation and, for an assistant embedded in a
support workflow, arguably a more common one: a user already has a script, and it is already wrong.

**Repair is graded by the same verifier, not a softer one.** A repaired script must clear every
level that a from-scratch T1 solution would have to clear against the same task. There is no
partial credit for "closer to correct": that would reintroduce the similarity-based scoring this
benchmark exists to replace.

**The faults are induced, not hand-written.** `anvil/inducer.py` mechanically derives seven fault
classes from the T1 canonical solutions, anchored to [failures observed on a real
model](OBSERVED_FAILURES.md) (F1–F7): a silently under-requested resource, a misplaced directive, a
prose-corrupted value, a missing no-default directive, a script with no `#SBATCH` at all, a
payload that no longer matches its own spec, and a directive value the scheduler rejects outright.
Hand-authoring this many broken variants across every task does not scale; mechanical induction
from a known-good starting point does.

**Broken must mean broken.** Building `tasks/t2_repair.jsonl` (`anvil induce`) runs each induced
variant through the real verifier and discards any that still verifies clean. Not every fault class
applies to every task (F1 needs a directive with a SLURM default to hide behind, F6 needs a
derived-value payload), so applicability is decided empirically, not declared in advance. The same
bracket as T1's oracle/broken guard applies here: the oracle repair (the T1 canonical solution,
returned regardless of the diagnosis) must score 1.0, and a no-op repair (the broken script,
unchanged) must score 0.0. `make guards-t2` checks both.

## Apptainer recipes (T3)

A third artifact type: a model writes an Apptainer definition file (`.def`), not a SLURM script.
The verifier keeps T1's shape but with a different vocabulary: `syntax` (a well-formed header and
at least a `%runscript`), `buildable` (does `apptainer build` succeed, playing the role
`submittability` plays for SLURM), `functional` (does the built container run and produce the
expected output), `resource_fit` (does the header and section set match the spec) and `safety`
(the same dangerous-pattern probe used for shell scripts, since `%post`/`%runscript` are shell
too).

**A harder dependency than T1's.** `submittability` needs SLURM; `functional` for SLURM scripts
only needs `bash`, which is essentially always present. For recipes, both `buildable` and
`functional` need a real `apptainer` binary, which is far less commonly installed than `bash`.
Without it, both are `skipped`, not failed, same discipline as `submittability` without a
scheduler, but a strictly larger fraction of the bracket depends on the missing tool. `make
guards-t3` therefore only asserts what `syntax`/`resource_fit`/`safety` can prove; the full
oracle-1.0/broken-0.0 bracket is `make docker-guards-t3`, which needs the opt-in
`docker-build-apptainer` image.

**Unprivileged build and run, not `--privileged`.** No capability is granted; what the
container needs is exemptions from Docker's confinement, and how many depends on the host.
On Docker Desktop for Windows (WSL2, no AppArmor) two suffice: `--security-opt
seccomp=unconfined --device /dev/fuse`. On a native Ubuntu 24.04 host, established one CI
run at a time on GitHub's runners, the full set is:

* `--security-opt seccomp=unconfined`: the user namespace the unprivileged build lives in;
* `--security-opt apparmor=unconfined`: `docker-default` denies the `mount` syscall
  outright (`failed to mount ...: permission denied`);
* `kernel.apparmor_restrict_unprivileged_userns=0` on the host: otherwise an unconfined
  process that creates a user namespace is moved to a stripped profile with no
  capabilities inside it (`mount namespace requires privileges`);
* `--security-opt systempaths=unconfined`: a fresh procfs cannot be mounted in a user
  namespace while Docker's masked `/proc` entries cover the original (`failed to mount
  proc filesystem`);
* `--device /dev/fuse` plus, in the image, a subuid/subgid range for root and
  `APPTAINER_UNPRIVILEGED=1`, since the PPA package has no setuid starter and every build
  goes through `--fakeroot`.

`--privileged` also works and collapses the list, but grants every capability besides.

**One configuration, two hosts, identical numbers.** The set above is not "what GitHub
needs" versus "what WSL2 needs": with it, `make docker-guards-t3` passes on both, and the
broken model's per-level scores agree to the digit (`syntax` 0.6, `buildable` 0.4 with 9
skipped, `functional` 0.0667, `resource_fit` 0.2, `safety` 0.8). The exemptions that only
matter under AppArmor are accepted and inert where AppArmor is not applied, so nothing has
to be selected per environment. `APPTAINER_UNPRIVILEGED` therefore defaults to on.

On Docker Desktop for Mac, `build` succeeded but `run` failed with `exec ... failed:
invalid argument`; this was once attributed to the nested `linuxkit` VM, an explanation the
AppArmor findings above weaken, and it has not been retested since.

## Cross-distribution ablation

Generation and verification are decoupled by design (`--save-generations`, then `anvil verify`
elsewhere): the same generated scripts can be verified again inside a different base image
without spending accelerator time twice. This is what makes the ablation possible at all.

`docker/Dockerfile` accepts `BASE_IMAGE` as a build argument for exactly this purpose, and
`scripts/crossdist_ablation.sh` drives the whole comparison from the generations a matrix run
already saved:

```
./scripts/crossdist_ablation.sh results/<run>
BASES="ubuntu:24.04 ubuntu:26.04 rockylinux:9" ./scripts/crossdist_ablation.sh results/<run>
```

It compares **per sample and per level**, not summary against summary: two environments can
reach the same pass@k while disagreeing about which samples pass, so only the per-sample
comparison supports the claim being made. Alignment is well defined because `verify` walks the
generations file in order, and the script checks the `task_id` at each index rather than
trusting it. It also prints the toolchain each image reported and refuses to let agreement pass
for evidence when every image reports the same one: comparing an image against itself would
otherwise "confirm portability" while testing nothing, the same trap the scheduler canary guards
against on `submittability`.

### Result

Qwen2.5-Coder-1.5B-Instruct, 8 T1 tasks, n=3, seeds 0/1/2, so 72 generated scripts, each
verified in both images on the experiment machine with a working scheduler, so
`submittability` took part in the comparison rather than being skipped on both sides. The
two environments differ in two ways at once, which the script reports before comparing
anything:

| image | coreutils | `gnu_faithful` | bash |
|---|---|---|---|
| `ubuntu:24.04` | GNU coreutils 9.4 | true | 5.2 |
| `ubuntu:26.04` | `uutils` (Rust), not GNU | false | 5.3 |

**72 sample comparisons, 360 level comparisons, zero divergence.** Same pass, same fail,
same skip decisions, everywhere. The earlier single-seed run reached the same conclusion
from 24 samples; three seeds of real-model output triple the material and, unlike that run,
put `submittability` inside the comparison.

This is a real result, not a shortcut past the fidelity concern that motivates pinning
`ubuntu:24.04` as the default. The T1 task suite's shell payloads are dominated by bash builtins
and `mkdir -p`, and the ablation did not find a difference because the current tasks are not
shaped to surface one. Tripling the samples does not change that argument: 72 scripts drawn from
the same eight prompts probe the same operations more times, not more operations.

### Where the two implementations do differ

That argument was worth settling directly rather than assuming, so
`scripts/coreutils_divergence.sh` asks the question with no tasks and no model in the way: 101
invocations run in both images and diffed. The first 73 were chosen blind, for being what a job
script does with its results and what a careful one does around them. They turned up nothing
behavioural, so a third family of 28 was chosen to hunt rather than to survey, aimed at
byte-versus-character width, number formatting, and the newer digest and slicing flags, which is
where two independent implementations have room to disagree without either being wrong.

**Ninety-one of the 101 agree**, exactly, output for output: sorting, counting, cutting, hashing,
`stat -c`, `du`, `df`, `seq`, `numfmt --to=iec`, `date` in five formats, `split`, `join`, `comm`,
`od`, `base64`, `basenc`, `cksum -a`, `b2sum`, `ls -v`. So does every exit code, including the ones
a careful script checks: a missing file, a bad flag, an expired `timeout`, `timeout
--preserve-status`.

Two of the ten that differ are there to be read rather than counted. `ls --version` says which
toolchain answered. `printf` overflow belongs to bash and not to coreutils, and sits in the probe
because the images differ in the shell as well, 5.2 against 5.3, so nothing found here is
attributable to coreutils until the shell is ruled out. The remaining eight fall into three kinds,
and only two of those kinds were known before the third family was added.

**Error text, not behaviour.** `mkdir`, `stat` and `ls` word their failures differently, and
`uutils` drops the `Try 'ls --help'` line and adds `(os error 2)`. Exit codes match, so only a
script that greps stderr would notice. That is not unheard of in job scripts, but it is a thin
target for a task.

**Locale fallback, which is behaviour.** Both images generate only `C`, `C.utf8` and `POSIX`.
Asked for `en_US.UTF-8`, GNU falls back to C and `uutils` applies its own Unicode collation
anyway:

| | GNU 9.4 | uutils 0.8.0 |
|---|---|---|
| `LC_ALL=C sort` | `10 9 A B a b` | `10 9 A B a b` |
| `LC_ALL=en_US.UTF-8 sort` | `10 9 A B a b` | `10 9 a A b B` |
| `LC_ALL=en_US.UTF-8 numfmt --grouping 1234567` | `1234567` | `1,234,567` |

A cluster whose login profile sets a locale that the compute nodes do not generate is not a
contrived situation, and there the same script produces a different ordering depending on which
coreutils the node runs.

**Character width, which needs no locale at all.** `café naïve` is ten characters and twelve
bytes, and the two implementations answer differently about it in the environment these containers
already run in:

| `LC_ALL` | GNU 9.4 `wc -m` | uutils 0.8.0 `wc -m` |
|---|---|---|
| `C` | 13 | 11 |
| `C.utf8` | 11 | 11 |

GNU answers the question the locale asks, and under `C` a character is a byte. `uutils` counts
codepoints whichever locale it is handed. `expand` divides the same way, padding to the byte column
under GNU and to the character column under `uutils`, so a table built around a non-ASCII field
comes out misaligned on one of them. And `numfmt --to=si 1536` is `1.6K` from GNU and `1.6k` from
`uutils`: one character, in the value a script prints or a downstream tool parses.

This is a more comfortable corner for a distribution-sensitive task than the locale one, because it
needs no misconfigured environment at all, only a payload that is not pure ASCII or a number
formatted in SI. It is also where the usual portability advice backfires. Pinning `LC_ALL=C` is
what makes a sort order reproducible across machines, and it is exactly what makes `wc -m` disagree
across these two toolchains.

So `uutils` 0.8.0 is a faithful stand-in for GNU coreutils 9.4 for what the eight current tasks do,
and that is a narrower statement than it looked before this family was added. What it is not is a
drop-in replacement in general.

### A task that can tell the two apart

`tasks/t1_coreutils.jsonl` sits in that corner deliberately. It asks for the character count of
`café naïve`, ten characters and twelve bytes, and states that the count must be characters rather
than bytes and must come out the same whichever implementation the compute node ships. It lives in
its own file for the same reason `tasks/t1_exec.jsonl` does: the eight in `tasks/t1_slurm.jsonl`
are the denominator of every published T1 number.

The point is not that a script can get it wrong. It is that the same script gets two verdicts:

| | `LC_ALL=C.utf8` (the reference) | `LC_ALL=C` |
|---|---|---|
| GNU coreutils 9.4 | `CHARS=10`, passes | `CHARS=12`, fails |
| `uutils` 0.8.0 | `CHARS=10`, passes | `CHARS=10`, passes |

`make docker-guards-coreutils` asserts both rows. The reference has to pass in both images, since a
task whose own solution is environment-dependent is defective rather than sensitive, and the
`LC_ALL=C` variant has to be judged differently in the two, since that is the only reason the task
exists. The variant is derived from the reference by substituting one line rather than written out,
so the two cannot drift into a comparison of two unrelated scripts.

It does not run under `make guards`. The ground truth is defined in the declared environment, and
on the BSD userland of a developer machine the reference counts bytes and pads its output, neither
of which is a defect in it.

**Three models measured, and none reaches the question.** Qwen2.5-Coder at 1.5B and 7B and
Granite 4.1 3B, 3 seeds each, n=5, verified in both images: 45 samples, 225 level comparisons,
**no divergence anywhere**. `functional` fails on all forty five in both images. **Not one of the
forty five pins a locale.**

That last number is the measurement the task was built for, and it is the same at every size and
in both families.

Why the divergence never surfaces is worth more than the agreement, and the two models fail
differently. The 1.5B passes `syntax` 14 of 15 and `resource_fit` 12, and its dominant shape is

```bash
echo "CHARS=$(wc -m <<< "café naïve")"
```

where a here-string appends a newline, so the count is 13 under GNU and 11 under `uutils`. Neither
is 10. Without that newline the same script would read 12 against 10, which is exactly the split
the task exists to produce: a second error, unrelated to the first, suppressed it.

The 7B passes `syntax` and `resource_fit` 15 of 15, writes better formed scripts than the smaller
model on every level the scheduler can see, and fails the payload for reasons that are not about
counting at all. One shape prints the count with no `CHARS=` prefix; the other pipes the prefix
itself into the counter, `echo -n "CHARS=" | wc -m`, and counts that. Both are wrong identically
on GNU and on `uutils`.

Granite 4.1 3B also passes `syntax` and `resource_fit` 15 of 15, and removes the divergence
outright:

```bash
CHARS=$(echo "caf\u00e9 na\u00efve" | wc -m)
```

It emits the escape sequences as literal text rather than the characters they stand for, so the
payload it counts is twenty ASCII characters and a newline. The answer is 21 and it is 21 on both
implementations, because a string with nothing above U+007F in it cannot make two coreutils
disagree. Asked to count non-ASCII characters, the model wrote a script with no non-ASCII in it.

So the task splits the two implementations by construction, `make docker-guards-coreutils` proves
it, and no model has yet written an artifact good enough for the split to decide anything. The
cross-distribution agreement these runs report is a statement about the models, not the
environments. That has been the honest reading of every agreement this ablation has produced, and
here it is demonstrated rather than conceded: on a task built to diverge, the divergence is still
out of reach.

## Retrieval ablation

Does giving a model reference material about SLURM semantics change how correctly it writes a
script? Three conditions, compared on the same model, seeds and tasks:

* **zero-shot**: the task prompt alone. This is what T1/T2/T3 have always done; introducing the
  other two arms changes nothing about the default behaviour.
* **vector**: TF-IDF cosine similarity between the task prompt and a small corpus of reference
  documents (`tasks/retrieval_corpus.jsonl`), implemented in pure Python (stdlib only): the corpus
  is small enough that a neural embedding model would be a dependency this ablation does not need.
* **vectorless**: exact tag overlap between the task and a document, no similarity scoring.
  Structure-based, not similarity-based: a document is retrieved because it is declared to be
  *about* the task's topic, not because its text happens to resemble the prompt.

The corpus is anchored to the same F1–F7 taxonomy as T2: the two most-referenced documents state
SLURM's silent resource defaults (F1) and the directive-placement rule (F2) directly, so the
hypothesis is not "does retrieval help in general" but "does surfacing the exact fact a model
tends to get wrong change whether it gets it wrong."

**`--retrieval` never changes the oracle/broken baseline.** `OracleModel` matches on
`prompt.startswith(task.prompt)`, not exact equality, because `build_prompt_with_context` always
appends retrieved material after the original prompt, never before it. `make guards` therefore
stays valid regardless of which retrieval arm is active.

### Result

Qwen2.5-Coder-1.5B-Instruct, n=3, seeds 0/1/2, 8 T1 tasks, on the experiment machine with a
real scheduler and GNU coreutils. Mean pass@1 across seeds, plus half the range:

| strategy | `syntax` | `submittability` | `functional` | `resource_fit` | `safety` | strict |
|---|---|---|---|---|---|---|
| zero-shot | 0.58±0.04 | 0.82±0.02 | 0.54±0.04 | **0.49±0.02** | 1.00±0.00 | **0.31±0.02** |
| vector | 0.54±0.04 | 0.83±0.04 | 0.49±0.06 | 0.42±0.08 | 1.00±0.00 | 0.21±0.04 |
| vectorless | 0.53±0.02 | 0.79±0.00 | 0.44±0.06 | **0.19±0.04** | 1.00±0.00 | **0.11±0.02** |

**Two of these columns were measured twice.** `scripts/retrieval_ablation.sh` grades with the
project venv against whatever scheduler the machine happens to run, and the machine it ran on has
one node, no GPUs and no job 12345, the environment described in [A table measured against the
wrong cluster](OBSERVED_FAILURES.md#a-table-measured-against-the-wrong-cluster) where the oracle
itself scores `submittability` 0.625. It reported 0.65 / 0.71 / 0.65 there, and strict
0.18 / 0.21 / 0.11. The saved generations were verified again inside `anvil:sched`
(`results/retrieval_regraded/`) and those are the numbers above.

The other three levels came back identical to the digit in all nine cells. That is the third
independent confirmation that `syntax`, `functional` and `resource_fit` do not depend on the
scheduler, and the first from an experiment outside the campaign the claim was first made on. The
mistake also cannot recur silently now: `_topology_healthy` skips `submittability` with its reason
stated when the scheduler in front of it is not the declared one, so this same script on that same
machine reports a skip instead of a number.

**Retrieval does not help this model, it costs it, and tag-based retrieval costs it most.** The
intervention series below takes that apart: most of the cost is paid for having anything attached
at all, whatever it says, and the arms differ by whether what they attach happens to teach the rule
the model is missing. Two
levels separate and they separate together. `resource_fit` falls 0.49, 0.42, 0.19 across the three
arms and `strict_all_levels` falls with it, 0.31, 0.21, 0.11, both monotone. Strict is the cleaner
of the two: no two of its ranges touch, so all three arms are separated. In `resource_fit` the
`vector` range still overlaps zero-shot's and only `vectorless` stands clear of both, which is the
same shape the first version of this table reported and the reason its headline named that arm.
`syntax` drifts down inside its ranges, `submittability` and `safety` do
not move at all: retrieved context does not change whether the scheduler accepts these scripts,
only whether they ask for the right things.

The correction changed the conclusion, not only the numbers. On the wrong cluster, strict read
0.18 / 0.21 / 0.11 and put `vector` nominally ahead of zero-shot, so the ordering the single-seed
pilot had reported (0.38 / 0.29 / 0.21) looked like it had failed to reproduce. Graded against the
declared topology the pilot's ordering is exactly what comes back, with the arms further apart
than it saw. The three-seed run was right that the pilot's magnitudes were one draw of a spread;
it was the grading environment, not the seeds, that inverted the ranking.

### Three explanations, all tested, all refuted

`scripts/retrieval_copying.py` reads the saved generations and measures each candidate
mechanism against the zero-shot arm as a control, since a value the model would have written
anyway is evidence of nothing.

**Copying is refuted.** The corpus states concrete values, so the model might have reproduced
them instead of deriving the ones the task asks for. It does not. `--array=1-5` appears 9
times in all three arms, unchanged by whether it was retrieved. `--nodes=2` falls from 3 to 0
as retrieval strengthens, and `--output=logs/out_%j` from 9 to 3 under vectorless. Not one
sample used a retrieved value where that value was wrong for its task. The arm whose
`resource_fit` collapses is the arm that reproduces corpus values *least*.

**Omission is refuted too, as a mechanism.** Retrieval does suppress directives, monotonically
and in the same order as the damage: 4.43 written per script zero-shot, 4.24 vector, 3.49
vectorless. Since `check_resource_fit` passes only on an empty problem list, one missing
directive sinks a whole sample, which would let a 21% drop in directives produce a 61% drop in
the level. But the prediction that follows, that failures shift toward omissions, does not
hold: omissions are 81% of problems zero-shot, 78% vector, 75% vectorless. Both kinds grow in
absolute terms and wrong values grow faster (times 2.4 against times 1.7).

**Shell expansion in the directive block is refuted as well.** The regrade surfaced
`sbatch: error: Invalid numeric value "${SLURM_NTASKS:-4}" for --ntasks`, which is the idiom the
corpus teaches for the payload, appearing where it cannot work: `sbatch` reads the `#SBATCH` lines
before any shell expands anything, so the directive receives the literal text. A model migrating
that idiom upward under retrieved context would fail `resource_fit` on a value that looks derived
and is not. The direction fits and the size does not: zero scripts do it zero-shot, one under
vector, two under vectorless, out of 72 per arm. `resource_fit` loses about 21 samples of 72
between the outer arms, and three scripts across two arms cannot carry that.

### What the level breaks on

The three refutations above all ask *how* the model writes. The regraded results answer a
different question, *where* it fails, counted per sample rather than per problem:

| | zero-shot | vector | vectorless |
|---|---|---|---|
| `resource_fit` passes | 35 | 30 | 14 |
| fails on `--time` or `--mem` alone | 5 | 23 | 30 |
| fails on something else as well | 32 | 19 | 28 |

72 samples per arm. The bottom row is flat, 32 against 28 between the outer arms. The passes fall
by 21 and the samples that fail on nothing but `--time` and `--mem` rise by 25, which is the whole
of it. Per directive the picture is the same: `--cpus-per-task` accounts for nine failing samples
in every arm and `--nodes` and `--ntasks` for six to nine, unchanged, while the counts for `--time`
and `--mem` triple and sextuple across the three arms.

Those two are the only directives SLURM has no default for, and they are also the two these tasks
state as a bound rather than a figure: a maximum walltime, a minimum memory. Everything the prompt
gives as an exact number survives retrieval untouched. What degrades is the part the model has to
decide.

So the earlier reading of this table, that resource-fitting competence degrades broadly rather
than in one identifiable way, is withdrawn. It was what a per-problem count looks like when a
per-sample count was the right one: the damage is confined to the decisions and leaves the
transcriptions intact. The three mechanisms above stay refuted, and none of them predicted this.

It also puts the copying refutation in its place rather than overturning it. That test measured
`--array=1-5`, `--nodes=2` and `--output=logs/out_%j`, which are three of the directives that turn
out not to move at all. It could not have caught this, and it could not have caught it in the
other direction either: the corpus states no walltime and no memory figure anywhere, so on these
two directives there is nothing available to copy.

### The intervention series

`--time` and `--mem` are the subject of exactly one corpus document, `doc_time_mem`, which states
that they have no SLURM-wide default and must be declared. `vectorless` fills its remaining slots
from the documents tagged `general` in corpus order, so one of them is attached to all eight tasks
and the rest are never seen, and the one that wins that slot by default is
`doc_directive_placement`. The arm whose `resource_fit` collapses is therefore the arm that never
meets the document about the two directives it fails on.

That is a claim the harness can test rather than a story, and `CORPUS=` and `RETRIEVAL_K=` on
`scripts/retrieval_ablation.sh` exist to test it. Seven conditions were run, same tasks, same three
seeds, same `n`, all graded in `anvil:sched`. The off-topic controls are passages about coastal
tides and about sourdough, with no term in common with the domain, each cut to exactly the 309
characters of `doc_time_mem` and placed first among the general documents so they take the same
slot. Against the real documents the variable is relevance, give or take the eight characters by
which a control's id outruns `doc_time_mem`'s in the `[id]` line that precedes every document in
the prompt.

`scripts/corpus_variants.py` builds all three variant corpora from the default one and prints, for
each, the document `vectorless` ends up attaching to each of the eight tasks. That listing is the
experimental precondition, and `tests/test_retrieval.py` asserts it: a document added to the corpus
above the general block would quietly send a different one to every task, and the table below would
then describe a run nobody performed.

Qwen2.5-Coder-1.5B-Instruct:

| general slot | `syntax` | `submittability` | `functional` | `resource_fit` | strict |
|---|---|---|---|---|---|
| nothing (zero-shot) | 0.58±0.04 | 0.82±0.02 | 0.54±0.04 | 0.49±0.02 | 0.31±0.02 |
| `doc_directive_placement` | 0.53±0.02 | 0.79±0.00 | 0.44±0.06 | 0.19±0.04 | 0.11±0.02 |
| `doc_time_mem` | 0.49±0.02 | 0.72±0.04 | 0.43±0.06 | **0.42±0.08** | 0.18±0.06 |
| both, k=3 | 0.50±0.06 | 0.72±0.08 | 0.35±0.06 | **0.40±0.08** | 0.13±0.04 |
| off-topic, tides | 0.50±0.00 | 0.83±0.04 | 0.43±0.02 | 0.21±0.06 | 0.03±0.04 |

Qwen2.5-Coder-7B-Instruct:

| general slot | `syntax` | `submittability` | `functional` | `resource_fit` | strict |
|---|---|---|---|---|---|
| nothing (zero-shot) | 1.00±0.00 | 0.79±0.00 | 0.88±0.00 | 1.00±0.00 | 0.67±0.00 |
| `doc_directive_placement` | 0.82±0.02 | 0.88±0.00 | 0.69±0.02 | 1.00±0.00 | 0.57±0.02 |
| `doc_time_mem` | 0.83±0.00 | 0.78±0.02 | 0.71±0.00 | **0.94±0.02** | 0.58±0.00 |
| off-topic, tides | 1.00±0.00 | 0.93±0.06 | 0.88±0.00 | 1.00±0.00 | 0.81±0.06 |
| off-topic, sourdough | 1.00±0.00 | 0.93±0.06 | 0.88±0.00 | 1.00±0.00 | 0.81±0.06 |

### Two factors account for all of it

**Whether a model can ignore what does not concern it.** The 1.5B cannot. A paragraph about tides
takes its `resource_fit` from 0.49 to 0.21, which is the 0.19 of the real document that does not
address that level: an irrelevant passage costs it as much as an unhelpful relevant one. The 7B
can. Two unrelated off-topic passages leave its `syntax`, `functional` and `resource_fit` on their
zero-shot values to the digit, while the two real documents cost it 18 points of `syntax`. So the
loss is not the extra text in the prompt for one model and is exactly that for the other, and what
separates them is capacity, not the corpus.

**What `doc_time_mem` is worth, which depends on what the model already knows.** At 1.5B it is
worth about 21 points of `resource_fit`: the two conditions carrying it read 0.42 and 0.40, the two
without it 0.21 and 0.19. At 7B the same document costs 5.6 points, 1.00 falling to 0.94, per seed
0.958, 0.917 and 0.958 against three exact 1.000s in every other condition. It teaches the model
that does not know these two directives have no default, and gets in the way of the one that does.

The two compose, and together they read the 1.5B column off: 0.49 with no context, about 0.20 once
anything is attached, back to 0.42 when what is attached is the document that teaches the missing
rule. The corpus ordering matters because it decides whether that recovery happens, not because
ordering is what damages the level.

`functional` is conditional on passing `syntax`, which is why 9 to 12 of its 24 samples are skipped
per cell and why its column tracks the syntax column rather than moving on its own.

### Observed, replicated, unexplained

At 7B an off-topic passage raises `submittability` from 0.79 to 0.93 and `strict_all_levels` from
0.67 to 0.81, the best of the five conditions. Two unrelated control texts give identical
aggregates, so it is not the text, and at 1.5B the same passage leaves `submittability` flat, 0.83
against 0.82, so it is not the design either. It fits what that level was shown to measure
elsewhere in this project, how much a model volunteers that nothing asked for, and an irrelevant
passage may hold it closer to the prompt. No mechanism is established and none is proposed here.

### What the series retired

Three readings written while it ran, each refuted by the run made to test it. That
`doc_directive_placement` costs `syntax` because it is the document about directive placement:
displacing it at 7B leaves `syntax` at 0.83 against 0.82. That the loss on `syntax` and
`functional` is independent of content: at 7B the off-topic controls cost nothing at all. That it
is therefore a content effect at both sizes: at 1.5B the off-topic control is as damaging as the
real documents. Each was plausible, each was cheap to test, and none survived.

Caveats that belong next to the numbers: three seeds and 24 verifications per cell, so the
half-ranges are spread, not confidence intervals; one model family at two sizes; the 7B
`resource_fit` dip is about four samples of 72, small and present on every seed; the off-topic
controls are two texts, not a distribution of texts.

### Where the context sits

Every condition above puts the documents after the task. Position is a variable of its own, since
the same paragraph reads as background when it comes first and as an afterthought when it comes
last. `--retrieval-position prepend` moves them, on the same corpus, model, seeds and `n`, so the
only difference from the second 7B row is the order:

| 7B, `doc_directive_placement` | `syntax` | `submittability` | `functional` | `resource_fit` | strict |
|---|---|---|---|---|---|
| appended | 0.82±0.02 | 0.88±0.00 | 0.69±0.02 | 1.00±0.00 | 0.57±0.02 |
| prepended | **0.75±0.00** | **0.99±0.02** | 0.62±0.00 | 0.92±0.08 | 0.57±0.06 |

Both directions at once. `syntax` falls a further 7 points with no overlap between the ranges, so
the 18 points the document costs when appended are not the whole price: some of the damage is the
content being there and some is it arriving first. `submittability` goes the other way and further,
to one refusal in 72 samples against twenty four under the same document appended, the highest this
level reaches on any 7B condition measured here. `strict_all_levels` is unchanged, 0.57 either way,
because the two cancel.

`resource_fit` reads 0.92 and should not be read: the seeds are 0.833, 0.917 and 1.000, so the
range touches the ceiling it sits at in every other condition with this document.

What makes it awkward is the upper bound rather than the model. The oracle indexes tasks by prompt
and used to match with `startswith`, which appending happens to preserve, so a prepend arm would
have scored a benchmark whose own reference solutions had quietly gone missing. It matches the task
prompt anywhere in what it receives instead, and `tests/test_retrieval.py` checks that the oracle
returns the same script under both positions for all eight tasks.

## Limitations

`functional` runs the script under `bash` in a sandbox by default, and every number published so
far was measured that way: `functional_executor: "bash"` in the result file says so. Real
submission is available as a second executor (see [Real submission](#real-submission-the-sbatch-executor)),
and with cgroup enforcement behind it a job is held to the memory it requested. What no task
exercises yet is binding: a job is confined to its cores, but nothing in the set asks what it was
given, and the GPUs are device files with nothing behind them. See
[`REFERENCE_CLUSTER.md`](REFERENCE_CLUSTER.md).

T2 failures will be partly synthetic, induced to obtain ground truth. The taxonomy is anchored to
[failures observed on real models](OBSERVED_FAILURES.md) and to published HPC-centre FAQs, and we
say so plainly.

## Roadmap

- [x] **Phase 1**: verifier (5 levels), 8 T1 tasks, oracle + broken, `pass@k`, reference cluster,
      preflight, generate/verify decoupling
- [x] **Phase 2**
  - [x] T2 diagnose-and-repair: mechanical fault induction (F1–F7), `tasks/t2_repair.jsonl`,
        `anvil repair` / `anvil verify-repair`, oracle-repair/no-op-repair guards
  - [x] failure-category breakdown: `aggregate_by_category`, per-category tables in
        `anvil repair` / `anvil verify-repair` output
  - [x] cross-distribution ablation: `BASE_IMAGE` build arg plus
        `scripts/crossdist_ablation.sh`, comparing per sample and per level. 24.04 against
        26.04 across 3 seeds: 360 level comparisons, zero divergence, see [Cross-distribution
        ablation](#cross-distribution-ablation)
  - [x] Apptainer recipes: `RecipeTask`, `RecipeLevel`, `anvil recipe` / `anvil verify-recipe`,
        `tasks/t3_apptainer.jsonl`, see [Apptainer recipes (T3)](#apptainer-recipes-t3). Both
        guards confirmed: `make guards-t3` (lenient) and `make docker-guards-t3` (strict,
        oracle 1.0 / broken 0.0, `apptainer` active) on Docker Desktop for Windows; ruled out on
        Docker Desktop for Mac (`apptainer run` fails there, see the section above)
  - [x] retrieval ablation: `anvil/retrieval.py` (TF-IDF vector / tag-based vectorless),
        `--retrieval` on `anvil run`, `scripts/retrieval_ablation.sh`, see [Retrieval
        ablation](#retrieval-ablation). Generated on the experiment machine across 3 seeds,
        all 9 cells, graded in the container: retrieval does not help this model, `vectorless`
        costs 30 points of `resource_fit` and 20 of `strict_all_levels`, and 22 of those 30 come
        back by attaching one different document, without the verdict moving with them
- [ ] **Phase 3**
  - [x] real submission: `--executor sbatch`, opt-in beside the `bash` default, with its own
        preflight and its own guard (`make guards-sbatch`), see [Real submission](#real-submission-the-sbatch-executor)
  - [x] cgroup enforcement: `task/cgroup` with RAM, swap and cores constrained, plus the task set
        that exercises it (`tasks/t1_exec.jsonl`, fault F8) and `make docker-guards-enforcement`
  - [ ] binding: a task that reads the affinity and the GPU it was actually given, which needs
        real devices rather than the placeholder files the declared topology stands on
  - [ ] Podman as a second verification runtime, rootless by default, so the confinement Docker
        had to be exempted from may not apply in the first place: the T3 unprivileged path needs
        `seccomp=unconfined`, `apparmor=unconfined`, `systempaths=unconfined`, `/dev/fuse` and a
        host `kernel.apparmor_restrict_unprivileged_userns=0` (see [Apptainer recipes
        (T3)](#apptainer-recipes-t3)), and its cgroup v2 delegation is precisely what the
        container's `slurmd` lacks. Two things to establish rather than assume: that nested user
        namespaces still allow `apptainer --fakeroot`, and that both strict brackets return the
        same per-level scores as Docker. A runtime that changes the numbers is not a fix, it is a
        second environment to declare
  - [ ] more families, on borrowed hardware: the second family answered the question this item was
        written for, and sharpened it. Granite 4.1 3B scores `submittability` above both Qwen sizes
        and fails it for a different reason, an option SLURM does not have rather than a partition
        name this cluster does not have, so the level is not flat across families and is not
        ordered by size either. What a third family would settle is whether that split between
        invented syntax and invented values is a pattern or a coincidence of two points, and it
        costs GPU time this project does not have. The split it already relies on covers it, since only generation
        needs an accelerator: generate wherever a free quota is, then verify at home in the
        container. Kaggle is the first choice for its documented quota, around 30 GPU-hours a week
        at roughly 16 GB, which fits a 7B in fp16 or a 30B in 4-bit; Colab's free tier has the same
        hardware but Google states the GPU is not guaranteed, and a sweep interrupted halfway
        resumes from the missing cells. Paperspace Gradient gives a free GPU with a 5 GB storage
        cap, which a 7B in fp16 does not fit, and public projects only. Spaces is out, its free
        hardware being CPU alone, and so is AI Studio, which is an API onto Gemini rather than a
        machine to compute on. A three-seed T1 matrix travels as 200 KB of JSONL, and `--no-exec` keeps the
        borrowed machine from grading anything: it has no scheduler and whatever coreutils it
        happens to ship. An arm behind an inference API was considered and set aside, free
        endpoints included: they route to whichever provider is cheapest at that moment, so the
        revision is not pinned and the seed is not honoured, which is the comparability every
        table here rests on
  - [ ] QLoRA reference model; state-space arm; hybrid classical-quantum artifacts
- [ ] **Phase 4: from a benchmark into something people run.** Everything above measures models,
      which is an audience of the few people studying those models. The verifier inside is useful
      to a much larger one, anybody generating HPC artifacts with an LLM who needs to know whether
      they will hold up before they are submitted. That audience needs no tasks, no oracle and no
      pass@k, only a verdict on the artifact in front of it, and almost all of the code it needs is
      already written.
  - [x] `anvil check`: a verdict on one artifact, with no task file, no model and no generations
        wrapper. Exit codes usable from a shell, so it composes into a hook or a CI step. This is
        the piece that turns the verifier from something the benchmark calls into something a user
        calls
  - [x] an installable distribution, so using it does not start with cloning this repository. The
        task files are mapped into the package at build time and `anvil/resources.py` prefers a
        checkout and falls back to them, so a clone reads `tasks/` exactly as every published
        number was measured against
  - [x] a submit-time policy check: `anvil check --policy`, `anvil/policy.py`. Ceilings on what a
        job may request, an allow-list of partitions, directives the site requires or forbids,
        compared against the *effective* request rather than the written one. The comparison runs
        the opposite way from `resource_fit`, which is the whole distinction between a task spec
        and a site rule: one fails a script that asks for too little, the other one that asks for
        too much
  - [x] published results, as a page rather than a table buried in these documents:
        [RESULTS.md](RESULTS.md), which carries every figure with its provenance, the four findings
        they support, and a section on what they are not
  - [x] dataset release: [DATASET.md](DATASET.md) and `dataset/MANIFEST.json`, which ties
        every published figure to the digest of the files it was measured against. The
        upload to a hosting platform needs the author's account and has not been done
  - [x] leaderboard: [LEADERBOARD.md](LEADERBOARD.md), rendered from
        `leaderboard/entries/` by `scripts/leaderboard.py` and held to them by a test, so
        the page cannot drift from the runs it claims to summarise. An entry graded
        against an older task digest is shown as stale rather than ranked
  - [x] preprint: `paper/anvil.tex`, compiled to `paper/anvil.pdf`. Its tables and the
        data behind its figures are emitted from `leaderboard/entries/` by
        `scripts/paper_data.py`, so the manuscript cannot quote a run that has since been
        re-imported, and `tests/test_paper.py` checks that plus the structure a compiler
        would catch. Related work is written from `scripts/litsweep.py`, 761 records over
        fourteen queries and four open APIs, with the sweep committed under `sweep/` so the
        selection can be audited; every entry carries a DOI, which a test enforces
  - [ ] one external user. It is the only item here that cannot be built, and the only one that
        decides whether any of the others mattered

### Real submission (the sbatch executor)

`submittability` runs `sbatch --test-only`, which decides whether a script would be *accepted*,
not whether it runs. `functional` closed part of that gap with `bash`, which ignores every
`#SBATCH` line and simulates three variables from the task constraints. `--executor sbatch`
submits the script for real instead, waits for the job, and reads its fate from `scontrol`.

It is opt-in, and stays opt-in. Every T1/T2 number in
[`OBSERVED_FAILURES.md`](OBSERVED_FAILURES.md) was measured under `bash`; making real submission
the default would have made everything measured afterwards incomparable with them, while as a
second arm both remain valid. The environment report carries `functional_executor`, so no number
is ambiguous about which produced it.

`anvil/cli.py` and `anvil/verifier.py` both used to call this Phase 2 work. It was never a listed
Phase 2 deliverable and it did not ship with the five that were, which is why it is recorded here
instead of inside a closed phase.

What the switch buys, concretely: the walltime the script asked for is enforced, so a job that
overruns comes back `TIMEOUT` instead of finishing; the payload runs with every variable the
scheduler injects, not the three simulated ones; and the output has to arrive through the files
the script's own `--output`/`--error` name, which `bash` never opens. What it does not buy: OOM
kills and binding, which need cgroup enforcement, and cgroups collide with the declared-topology
principle. A job asking for 64 GB on the reference cluster cannot be held to a real machine's
memory without turning the score back into a property of the host.

Two things had to be settled before it could grade anything.

**A scheduler that accepts jobs without running them.** `slurm_healthy` proves `sbatch
--test-only` works, which is a configuration check needing no `slurmd` at all: the verification
image ran for months with no `slurmd` and still reported `idle` nodes, because `SlurmdTimeout=0`
keeps slurmctld from marking them DOWN. Under this executor that would have failed every artifact
and looked like a terrible model, so there is a second preflight: a canary submitted for real,
which must reach `COMPLETED` and leave its output where the harness can read it. When it does not,
`functional` is **skipped**, never failed.

**Whose fault a job that never finishes is.** Four outcomes are the scheduler's and are skipped: a
pending job whose `Reason` can never clear (`t1_dependency_chain` asks for
`--dependency=afterok:12345`, which the reference cluster satisfies with a *held* placeholder job,
so it never starts), a job still queued when the timeout expires, a record the scheduler discarded
before it could be read, and an unhealthy canary. One outcome is the script's and fails: a job
still *executing* at the timeout. The distinction is the same one the canary was built for, one
level up.

Where it can run at all turned out to be its own story. The verification image had never executed
a job, because `sbatch --test-only` needs no `slurmd` and so no `slurmd` ever had to work; four
faults sat there undisturbed, the last of them a property of Ubuntu 24.04's SLURM package rather
than of this configuration. All four are recorded in
[`REFERENCE_CLUSTER.md`](REFERENCE_CLUSTER.md#making-the-container-execute-and-where-it-stops),
together with the opt-in accounting image that closes them without giving up GNU coreutils. The
bracket there: eight tasks, seven executed for real, one skipped for a dependency that can never
clear, `strict_all_levels` 1.0, broken model 0.0.

Measuring the executor means comparing it with the one it stands beside, not reporting its pass@k
alone: `scripts/executor_ablation.sh` verifies the same generations twice inside the same image and
counts the samples on which the two arms part company.

The answer on the current task set is deflationary, and worth stating that way. Across 2340
artifacts from three models, real submission costs `functional` up to 21 points depending on the
cell and nothing at all in one of them, and moves `strict_all_levels` by nothing anywhere: the
scripts it stops were already failing another level, mostly `submittability`, so the executor
propagates a verdict rather than producing one.
Exactly one artifact of the 2340 changes verdict, and it changes in favour of real submission,
which accepts a script the sandbox wrongly rejects. The nine runs of the retrieval intervention
series were graded the same way and add 864 comparisons without a single further disagreement,
which puts the count at one artifact of 3204. The numbers are in
[`OBSERVED_FAILURES.md`](OBSERVED_FAILURES.md#real-submission-moves-functional-and-barely-touches-the-verdict).

That is a statement about these eight tasks, not about the executor. A task built to need real
execution does need it: F8 below is invisible to every static level and to bash, and there the
verdict does change.

### The fault only execution can see

Enforcement is worth nothing unless something exercises it, and none of the eight T1 tasks
allocates enough memory to notice a limit. `tasks/t1_exec.jsonl` adds one that does: it holds 64MB
in a shell variable and asks for enough memory to fit, without the spec stating a number. That
omission is the whole design. A task that pins a minimum has the fault covered statically already,
since cutting the value below it fails `resource_fit`; leaving it open makes the payload's real
need the only ground truth, and no check that reads the text can reach it.

From it the inducer builds **F8**, a `--mem` cut to 16M. The value is well formed, within spec, and
accepted by the scheduler; the script runs to completion under `bash` and dies `OUT_OF_MEMORY`
under real submission with enforcement. That is the first fault class in the taxonomy that only
execution can catch, and it is why the set lives in its own file: `anvil induce` keeps only
variants that actually fail, so inducing it needs `--executor sbatch`, and the shared
`tasks/t2_repair.jsonl` stays byte-identical to the one every published T2 number was measured on.

The guard that protects it is `make docker-guards-enforcement`, and its last assertion is the one
that matters: the no-op repair of the F8 sample must *fail* `functional`. It passes wherever the
allocation is not enforced, so a guard that only checked the bracket would happily certify an
environment enforcing nothing.

An induced fault only shows that the harness can catch something. Whether a model commits it is a
separate question, and on this task the answer is yes: asked for enough memory to hold 64MB, the
1.5B model writes `--mem=64M` in fourteen of fifteen samples, the size of the data with nothing
left for the process holding it, and one of those samples passes every other level and is then
OOM-killed. The measurement is in
[`OBSERVED_FAILURES.md`](OBSERVED_FAILURES.md#f8-memory-request-below-what-the-payload-uses).

One property of real submission had to be handled before a correct script could pass at all.
slurmstepd opens the file named by `--output` before the script's first command runs, so the
`mkdir -p logs` inside the `t1_output_paths` reference solution is dead code here: the job fails to
open `logs/out_%j.txt` and never starts. `bash` executes that line in time, a real scheduler does
not. Preparing the working directory is the submitter's job on a real cluster too, so the harness
creates those directories before submitting, and the level goes back to measuring the script.

Evaluated models will include a **state-space** model alongside transformers, to test whether
architecture matters for operational artifacts.

Evaluated models will include a **state-space** model alongside transformers, to test whether
architecture matters for operational artifacts.

# Design

Why this benchmark exists, and why it is built the way it is.

## The question

*When an LLM writes the SLURM job script a supercomputer user actually needs, is it correct?*

Not "does it resemble the reference answer" — **does it parse, does the scheduler accept it, does
it run, and does it request the right resources?**

Assistants that help HPC users are appearing, but they are evaluated with semantic similarity
metrics because no validated HPC benchmark exists — their own authors say so. Meanwhile,
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
| `functional` | Does it run and exit 0? | sandboxed execution, expected output |
| `resource_fit` | Does it request what was asked? | effective request vs. task constraints |
| `safety` | Is it dangerous? | destructive-pattern probes |

Two design choices carry the scientific weight.

**`skipped` is never `passed`.** No scheduler on your laptop? `submittability` is skipped and
scored as *not passed*. The metrics stay honest on any machine.

**Dangerous scripts are never executed.** `safety` gates `functional`.

### The misplaced-directive check

SLURM stops reading `#SBATCH` lines at the first real command. Directives after it are **silently
ignored** — `sbatch` accepts the job and the request is wrong. Anvil catches this;
`sbatch --test-only` cannot.

### Effective requests, not string presence

`resource_fit` compares the **effective** resource request against the spec, applying SLURM's
documented defaults: `--nodes` → 1, `--ntasks` → one task per node, `--cpus-per-task` → 1. A serial
script that omits `--nodes` still requests one node, and is correct.

Directives with no universal default — `--time`, `--mem`, `--gpus` — depend on partition
configuration. Omitting them means the resource was never requested: a genuine failure against a
spec that asks for it. Tasks can still demand explicitness through `required_directives`.

The distinction is the point. Checking whether a string appears is surface-form matching —
precisely what this benchmark exists to replace. An early version did exactly that, and failed
scripts that `sbatch` accepted.

## Oracle and broken model

Every benchmark should ship both. Few do.

- **Oracle** — canonical solutions. Proves the tasks are solvable and the verifier is not too
  strict. CI fails if it drops below 1.0.
- **Broken** — faulty artifacts (missing shebang, misplaced directive, walltime overrun,
  `rm -rf /`, non-zero exit). Proves the verifier is not too permissive. CI fails if it scores
  above 0.0 strict, **or if the safety guard is never exercised**.

Together they bracket the verifier from both sides. Neither test is decorative: the oracle caught
a real bug during development, where the harness injected `SLURM_CPUS_PER_TASK=1` into a task that
requested 4 cores — the harness was contradicting the spec it was checking.

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
recognise and fix a broken one — a distinct and, for an assistant embedded in a support workflow,
arguably more common situation: a user already has a script, and it is already wrong.

**Repair is graded by the same verifier, not a softer one.** A repaired script must clear every
level that a from-scratch T1 solution would have to clear against the same task. There is no
partial credit for "closer to correct" — that would reintroduce the similarity-based scoring this
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
applies to every task — F1 needs a directive with a SLURM default to hide behind, F6 needs a
derived-value payload — so applicability is decided empirically, not declared in advance. The same
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

**Unprivileged build and run, not `--privileged`.** Apptainer's unprivileged build needs a user
namespace, which Docker's default seccomp profile blocks; running the built `.sif` needs
`/dev/fuse`. `--security-opt seccomp=unconfined --device /dev/fuse` grants exactly these two and
nothing else. `--privileged` also works but grants far more. Observed: both flags together work
fully on Docker Desktop for Windows; on Docker Desktop for Mac (a nested `linuxkit` VM), `build`
succeeds but `run` fails with `exec ... failed: invalid argument`, a limit of that specific nested
virtualization stack, not of Linux or Docker in general.

## Cross-distribution ablation

Generation and verification are decoupled by design (`--save-generations`, then `anvil verify`
elsewhere): the same generated scripts can be verified again inside a different base image
without spending accelerator time twice. This is what makes the ablation possible at all.

`docker/Dockerfile` accepts `BASE_IMAGE` as a build argument for exactly this purpose:

```
docker build -t anvil:2604 --build-arg BASE_IMAGE=ubuntu:26.04 docker/
```

A first run (Qwen2.5-Coder-1.5B-Instruct, 8 T1 tasks, n=3, no seed variation) verified the same
24 generations inside `ubuntu:24.04` (GNU coreutils 9.4) and `ubuntu:26.04` (`uutils`, the Rust
reimplementation, `gnu_faithful: false`). Every level, on every sample, agreed exactly: same
pass/fail, same skip decisions, zero divergence.

This is a real result, not a shortcut past the fidelity concern that motivates pinning
`ubuntu:24.04` as the default. The T1 task suite's shell payloads are dominated by bash builtins
and `mkdir -p`, operations where `uutils`'s GNU compatibility is presumably solid; they do not
exercise the coreutils corners (`stat`, `sort`, `date` formatting, flag-level differences) where
`uutils` and GNU coreutils are known to diverge. The ablation did not find a difference here
because the current tasks are not shaped to surface one, not because the difference does not
exist. A meaningful negative result would need a task that specifically depends on one of those
corners; none of the eight T1 tasks currently do.

## Retrieval ablation

Does giving a model reference material about SLURM semantics change how correctly it writes a
script? Three conditions, compared on the same model, seeds and tasks:

* **zero-shot** — the task prompt alone. This is what T1/T2/T3 have always done; introducing the
  other two arms changes nothing about the default behaviour.
* **vector** — TF-IDF cosine similarity between the task prompt and a small corpus of reference
  documents (`tasks/retrieval_corpus.jsonl`), implemented in pure Python (stdlib only): the corpus
  is small enough that a neural embedding model would be a dependency this ablation does not need.
* **vectorless** — exact tag overlap between the task and a document, no similarity scoring.
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

A first pilot (Qwen2.5-Coder-1.5B-Instruct, n=3, single seed) found `strict_all_levels` at
0.38 (zero-shot), 0.29 (vector) and 0.21 (vectorless): retrieval made this small model *worse*,
not better. Single seed, single small model: an observation, not a result. Plausible reading, not
yet tested: appending reference text after the prompt pushes the actual instructions further from
the point where a small, weakly instruction-following model pays the most attention, independent
of whether the appended text is relevant. Multiple seeds, a larger model, and a variant that
prepends context instead of appending it are the next measurements, not a conclusion to draw from
one run.

## Limitations

`functional` runs the script under `bash` in a sandbox; it does not submit it to `sbatch`. This is
recorded as `functional_executor: "bash"` in every result file. OOM kills, walltime overruns and
CPU/GPU binding are therefore not observed — precisely the failure modes most interesting for the
planned repair task. See [`REFERENCE_CLUSTER.md`](REFERENCE_CLUSTER.md).

T2 failures will be partly synthetic, induced to obtain ground truth. The taxonomy is anchored to
[failures observed on real models](OBSERVED_FAILURES.md) and to published HPC-centre FAQs, and we
say so plainly.

## Roadmap

- [x] **Phase 1** — verifier (5 levels), 8 T1 tasks, oracle + broken, `pass@k`, reference cluster,
      preflight, generate/verify decoupling
- [x] **Phase 2**
  - [x] T2 diagnose-and-repair — mechanical fault induction (F1–F7), `tasks/t2_repair.jsonl`,
        `anvil repair` / `anvil verify-repair`, oracle-repair/no-op-repair guards
  - [x] failure-category breakdown — `aggregate_by_category`, per-category tables in
        `anvil repair` / `anvil verify-repair` output
  - [x] cross-distribution ablation — `BASE_IMAGE` build arg, first run (24.04 vs 26.04) found
        no divergence on the current T1 task suite, at one seed, see [Cross-distribution
        ablation](#cross-distribution-ablation)
  - [x] Apptainer recipes — `RecipeTask`, `RecipeLevel`, `anvil recipe` / `anvil verify-recipe`,
        `tasks/t3_apptainer.jsonl`, see [Apptainer recipes (T3)](#apptainer-recipes-t3). Both
        guards confirmed: `make guards-t3` (lenient) and `make docker-guards-t3` (strict,
        oracle 1.0 / broken 0.0, `apptainer` active) on Docker Desktop for Windows; ruled out on
        Docker Desktop for Mac (`apptainer run` fails there, see the section above)
  - [x] retrieval ablation — `anvil/retrieval.py` (TF-IDF vector / tag-based vectorless),
        `--retrieval` on `anvil run`, `scripts/retrieval_ablation.sh`, see [Retrieval
        ablation](#retrieval-ablation). The tooling is what this box covers. The measurement is
        still a pilot: one seed of three, so 3 of the 9 cells. It belongs on the experiment
        machine, since a development machine skips `submittability` and is not `gnu_faithful`.
- [ ] **Phase 3** — real submission via `sbatch` (see below); QLoRA reference model; state-space
      arm; hybrid classical-quantum artifacts
- [ ] **Phase 4** — dataset release, leaderboard, preprint

### Real submission is not in Phase 2

`functional` executes the payload with `bash` in a sandbox, never through `sbatch`, so no level
observes what only a scheduler produces: OOM kills, walltime overruns, the allocation a job
actually receives. `submittability` runs `sbatch --test-only`, which decides whether a script
would be accepted, not whether it runs.

`anvil/cli.py` and `anvil/verifier.py` both used to call real submission Phase 2 work. It was
never a listed Phase 2 deliverable and it did not ship with the five that were, so it is recorded
here as Phase 3 rather than left as a promise inside a closed phase. Until it lands, every result
carries `functional_executor: "bash"` in its environment report, and the gap between submittable
and runnable stays open.

Evaluated models will include a **state-space** model alongside transformers, to test whether
architecture matters for operational artifacts.

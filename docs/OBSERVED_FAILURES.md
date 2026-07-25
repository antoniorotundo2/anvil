# Observed failure classes

Not invented: **observed**, on the first run of a real model against the T1 tasks.
This is the empirical seed of the T2 (diagnose-and-repair) taxonomy, and the answer to the
reviewer question *"are your induced failures representative?"*

Generated with `Qwen/Qwen2.5-Coder-1.5B-Instruct`, fp16 on MPS, n=1, seed=0, 8 tasks.
Verified inside the container (Ubuntu 24.04, bash 5.2, GNU coreutils, live `slurmctld`).
`strict_all_levels` pass@1 = 0.25; `submittability` = 0.625.
**Single sample, single seed: an observation, not a result.** Numbers predate the parser fix below.

---

## F1 — Silent under-request through an omitted default
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

## F2 — Directives after the first command
On the I/O task the model emitted `mkdir -p logs` and *then* four `#SBATCH` lines. SLURM stops
reading directives at the first command: job name, walltime and output paths are **silently
ignored**. `sbatch` accepts the job.

Caught by `check_syntax`, not by `sbatch --test-only`. This is the check that justifies going
beyond dry-run validation.

## F3 — Prose leaking into directive values
```
#SBATCH --mem=2 referencing GB
#SBATCH --time=20:00: referencing the time in hours, minutes, and seconds
```
The model narrates inside the directive. Degenerate small-model behaviour, but it produces an
artifact that parses as a script and fails only at the semantic level.

### F3 note: the diagnosis is parser-dependent
Before the multi-option parser fix, `#SBATCH --mem=2 referencing GB` was reported as
`--mem unparsable`. After it, `shlex` splits the line, the value becomes `2`, which parses
cleanly as 2 MB — so the same artifact is now reported as `--mem 2MB below minimum 2048MB`.

The error is still caught, but **its category changed**: degenerate prose now masquerades as an
under-request. For a taxonomy built on these categories this matters, and the two must be told
apart before T2 induction relies on them.

## F4 — Missing directive with no universal default
`--time` and `--mem` omitted where the spec demanded them. Unlike F1 there is no defensible
default: the resource is simply never requested.

## F5 — No `#SBATCH` at all
On the container task the model returned a plain shell script. The artifact is not a job script.

## F6 — Payload/spec mismatch
`THREADS=4` expected, not printed: the script exports `OMP_NUM_THREADS` from a variable that its
own (missing) `--cpus-per-task` never set. The failure cascades from F4.

## F7 — Malformed directive values rejected by the scheduler
```
sbatch: error: Invalid directive found in batch script: referencing
sbatch: error: Invalid --time specification
```
F3's prose leakage surfaces at `submittability`: the scheduler itself refuses the artifact.
Three of the eight tasks failed here. This is the class where dry-run validation *does* work.

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
several options may share one line — the parser read only the first and swallowed the rest.

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
   presence and reported `found None` — which would have hidden F1 entirely behind a generic
   "missing directive". The bug fix *produced* the finding.

## Multi-seed validation (T1 and T2)

The single-seed pilot above named three gaps before its numbers could be quoted as a rate.
This run closes all three: 3 seeds (0/1/2), n=5, two model sizes, on a real machine with GNU
coreutils and a live `slurmctld` (WSL2, Ubuntu 24.04, RTX 3060, 4-bit quantization). Full
per-category results are in `results/20260724_124032/` on the experiment machine.

**T1 (from scratch), pass@1, mean ± half-range across seeds:**

| model | syntax | submittability | functional | resource_fit | strict_all_levels |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.58±0.02 | 0.68±0.05 | 0.53±0.04 | 0.44±0.01 | 0.18±0.03 |
| Qwen2.5-Coder-7B-Instruct   | 1.00±0.00 | 0.62±0.00 | 0.88±0.00 | 1.00±0.00 | 0.50±0.00 |

**T2 (diagnose-and-repair), same protocol:**

| model | syntax | submittability | functional | resource_fit | strict_all_levels |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 0.79±0.02 | 0.57±0.00 | 0.66±0.00 | 0.41±0.01 | 0.20±0.00 |
| Qwen2.5-Coder-7B-Instruct   | 0.98±0.00 | 0.61±0.00 | 0.87±0.00 | 0.97±0.01 | 0.46±0.00 |

Two findings hold across both models and all three seeds.

**`submittability` does not scale with model size.** Every other level improves sharply from
1.5B to 7B. `submittability` stays roughly flat (0.68 to 0.62 in T1, 0.57 to 0.61 in T2). These
failures look like scheduler-facing syntactic edge cases rather than semantic misunderstanding,
and a bigger model does not fix them on its own.

**F1 is the hardest repair category, now with numbers behind it.** At 1.5B, `resource_fit` for
F1 repairs is 0.0 on all three seeds: the small model never restores the missing directive. At
7B, `resource_fit` reaches 1.0, but `strict_all_levels` still caps at 0.33, because the
bottleneck moves to `submittability` (also 0.33): the model now understands what was missing,
but the resulting script still often fails `sbatch --test-only`. F6 (payload/spec mismatch), by
contrast, reaches `strict_all_levels` = 1.0 for both models on almost every seed: it is the easy
category of the taxonomy.

## Next measurements needed

The three gaps named above (multiple seeds, a scheduler-faithful environment, a larger model)
are resolved; see [Multi-seed validation](#multi-seed-validation-t1-and-t2). What remains open:

- more than two model sizes and families, to see where the `submittability` plateau breaks, if
  it breaks;
- a genuine outlier check on F3, to separate small-model degeneracy from a stable semantic
  error as model scale keeps increasing;
- the retrieval ablation, whose tooling is in place but whose measurement is still one seed
  of three, so 3 of the 9 cells;
- a second seed for the cross-distribution ablation, which so far agreed level by level on a
  single one.

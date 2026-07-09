# Observed failure classes

Not invented: **observed**, on the first run of a real model against the T1 tasks.
This is the empirical seed of the T2 (diagnose-and-repair) taxonomy, and the answer to the
reviewer question *"are your induced failures representative?"*

Run: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, fp16 on MPS, n=1, seed=0, 8 tasks.
`strict_all_levels` pass@1 = 0.25. **Single sample, single seed: an observation, not a result.**

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

## F4 — Missing directive with no universal default
`--time` and `--mem` omitted where the spec demanded them. Unlike F1 there is no defensible
default: the resource is simply never requested.

## F5 — No `#SBATCH` at all
On the container task the model returned a plain shell script. The artifact is not a job script.

## F6 — Payload/spec mismatch
`THREADS=4` expected, not printed: the script exports `OMP_NUM_THREADS` from a variable that its
own (missing) `--cpus-per-task` never set. The failure cascades from F4.

---

## What this changes

1. **T2 induction can be grounded.** F1–F6 are real, and each maps to an inducer.
2. **F1 is the class to build the paper around.** It is invisible to every proxy metric, invisible
   to the scheduler, and expensive in practice.
3. **`resource_fit` must reason about effective requests.** An earlier version compared string
   presence and reported `found None` — which would have hidden F1 entirely behind a generic
   "missing directive". The bug fix *produced* the finding.

## Next measurements needed

- multiple seeds and n > 1 before any of this is quoted as a rate;
- inside the container, so `submittability` is active and `functional` runs on GNU coreutils;
- a larger model, to separate "small-model degeneracy" (F3) from "genuine semantic error" (F1).

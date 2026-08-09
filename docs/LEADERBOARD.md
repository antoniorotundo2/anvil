# Leaderboard

Generated from `leaderboard/entries/`, never edited by hand: run
`./scripts/leaderboard.py render`. Every figure is `pass@1`, the mean across seeds with
half the range beside it. Half the range is not a confidence interval, and with three
seeds no significance is claimed anywhere on this page.

`strict_all_levels` is the ranking column: it requires every level either to pass or to
be out of the machine's reach, and a skipped level is never a passed one.

An import refuses to replace an entry measured under other conditions, so a cell
taken at fp16 cannot quietly overwrite one taken at 4-bit. Quantization, base image,
samples per task and seeds are recorded but not in the key: publishing both means
widening the key, and the refusal is what makes that a decision.

A model can appear twice under one task file, once per executor, and on
`tasks/t1_exec.jsonl` it should: that set states no memory minimum, so what a script
needs is a property of the payload the model wrote and only real submission can
decide it. Reading the `bash` row there as the result is the mistake the set exists
to expose.

## `tasks/t1_exec.jsonl`

| model | `syntax` | `submittability` | `functional` | `resource_fit` | `safety` | `strict_all_levels` | conditions |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-1.5B-Instruct (stale rules) | 1.000±0.000 | 0.500±0.200 | 0.333±0.150 | 0.667±0.050 | 1.000±0.000 | 0.133±0.050 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen2.5-Coder-1.5B-Instruct (stale rules) | 1.000±0.000 | 0.500±0.200 | 0.167±0.100 | 0.667±0.050 | 1.000±0.000 | 0.133±0.050 | 3 seeds, n=5, sbatch, 4-bit |
| Qwen/Qwen2.5-Coder-7B-Instruct (stale rules) | 1.000±0.000 | 0.500±0.000 | 0.967±0.050 | 0.500±0.000 | 1.000±0.000 | 0.467±0.050 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen2.5-Coder-7B-Instruct (stale rules) | 1.000±0.000 | 0.500±0.000 | 0.000±0.000 | 0.500±0.000 | 1.000±0.000 | 0.000±0.000 | 3 seeds, n=5, sbatch, 4-bit |
| Qwen/Qwen3.5-9B (stale rules) | 0.833±0.100 | 0.833±0.100 | 0.467±0.050 | 1.000±0.000 | 1.000±0.000 | 0.467±0.050 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen3.5-9B (stale rules) | 0.833±0.100 | 0.833±0.100 | 0.700±0.100 | 1.000±0.000 | 1.000±0.000 | 0.700±0.100 | 3 seeds, n=5, sbatch, 4-bit |
| google/gemma-4-12B-it (stale rules) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 3 seeds, n=5, bash, 4-bit |
| google/gemma-4-12B-it (stale rules) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 3 seeds, n=5, sbatch, 4-bit |
| ibm-granite/granite-4.1-3b (stale rules) | 1.000±0.000 | 1.000±0.000 | 0.467±0.200 | 0.133±0.050 | 1.000±0.000 | 0.067±0.100 | 3 seeds, n=5, bash, 4-bit |
| ibm-granite/granite-4.1-3b (stale rules) | 1.000±0.000 | 1.000±0.000 | 0.467±0.200 | 0.133±0.050 | 1.000±0.000 | 0.067±0.100 | 3 seeds, n=5, sbatch, 4-bit |

An entry marked *stale tasks* was measured against a different version of this task file; one marked *stale rules* was graded by a different verifier, and *unstamped* predates the digest being recorded at all. Any of the three means the row is not comparable with the rest of the column. They are shown rather than deleted.

## `tasks/t1_slurm.jsonl`

| model | `syntax` | `submittability` | `functional` | `resource_fit` | `safety` | `strict_all_levels` | conditions |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-1.5B-Instruct (stale rules) | 0.575±0.025 | 0.842±0.013 | 0.533±0.037 | 0.442±0.013 | 1.000±0.000 | 0.308±0.025 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen2.5-Coder-7B-Instruct (stale rules) | 1.000±0.000 | 0.792±0.025 | 0.875±0.000 | 1.000±0.000 | 1.000±0.000 | 0.667±0.025 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen3.5-9B (stale rules) | 1.000±0.000 | 0.875±0.025 | 0.650±0.025 | 0.600±0.050 | 1.000±0.000 | 0.450±0.025 | 3 seeds, n=5, bash, 4-bit |
| google/gemma-4-12B-it (stale rules) | 0.875±0.000 | 0.867±0.013 | 0.875±0.000 | 0.917±0.050 | 1.000±0.000 | 0.658±0.062 | 3 seeds, n=5, bash, 4-bit |
| ibm-granite/granite-4.1-3b (stale rules) | 1.000±0.000 | 0.875±0.000 | 0.842±0.037 | 0.550±0.000 | 1.000±0.000 | 0.425±0.000 | 3 seeds, n=5, bash, 4-bit |

An entry marked *stale tasks* was measured against a different version of this task file; one marked *stale rules* was graded by a different verifier, and *unstamped* predates the digest being recorded at all. Any of the three means the row is not comparable with the rest of the column. They are shown rather than deleted.

## `tasks/t2_exec_repair.jsonl`

| model | `syntax` | `submittability` | `functional` | `resource_fit` | `safety` | `strict_all_levels` | conditions |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-1.5B-Instruct (stale rules) | 0.880±0.020 | 0.993±0.010 | 0.847±0.010 | 0.513±0.010 | 1.000±0.000 | 0.393±0.030 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen2.5-Coder-1.5B-Instruct (stale rules) | 0.880±0.020 | 0.993±0.010 | 0.267±0.040 | 0.513±0.010 | 1.000±0.000 | 0.173±0.030 | 3 seeds, n=5, sbatch, 4-bit |
| Qwen/Qwen2.5-Coder-7B-Instruct (stale rules) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.993±0.010 | 1.000±0.000 | 0.993±0.010 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen2.5-Coder-7B-Instruct (stale rules) | 1.000±0.000 | 1.000±0.000 | 0.400±0.000 | 0.993±0.010 | 1.000±0.000 | 0.400±0.000 | 3 seeds, n=5, sbatch, 4-bit |
| Qwen/Qwen3.5-9B (stale rules) | 0.993±0.010 | 0.993±0.010 | 0.993±0.010 | 0.847±0.010 | 1.000±0.000 | 0.847±0.010 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen3.5-9B (stale rules) | 0.993±0.010 | 0.993±0.010 | 0.733±0.020 | 0.847±0.010 | 1.000±0.000 | 0.587±0.010 | 3 seeds, n=5, sbatch, 4-bit |
| google/gemma-4-12B-it (stale rules) | 0.893±0.020 | 0.947±0.030 | 0.880±0.020 | 1.000±0.000 | 1.000±0.000 | 0.880±0.020 | 3 seeds, n=5, bash, 4-bit |
| google/gemma-4-12B-it (stale rules) | 0.893±0.020 | 0.947±0.030 | 0.560±0.050 | 1.000±0.000 | 1.000±0.000 | 0.560±0.050 | 3 seeds, n=5, sbatch, 4-bit |
| ibm-granite/granite-4.1-3b (stale rules) | 0.900±0.000 | 0.887±0.020 | 0.900±0.000 | 0.660±0.040 | 1.000±0.000 | 0.660±0.040 | 3 seeds, n=5, bash, 4-bit |
| ibm-granite/granite-4.1-3b (stale rules) | 0.900±0.000 | 0.887±0.020 | 0.280±0.020 | 0.660±0.040 | 1.000±0.000 | 0.247±0.010 | 3 seeds, n=5, sbatch, 4-bit |

An entry marked *stale tasks* was measured against a different version of this task file; one marked *stale rules* was graded by a different verifier, and *unstamped* predates the digest being recorded at all. Any of the three means the row is not comparable with the rest of the column. They are shown rather than deleted.

## `tasks/t2_repair.jsonl`

| model | `syntax` | `submittability` | `functional` | `resource_fit` | `safety` | `strict_all_levels` | conditions |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-1.5B-Instruct (stale rules) | 0.792±0.016 | 0.886±0.005 | 0.664±0.005 | 0.330±0.009 | 1.000±0.000 | 0.211±0.007 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen2.5-Coder-7B-Instruct (stale rules) | 0.983±0.002 | 0.977±0.000 | 0.870±0.002 | 0.965±0.007 | 1.000±0.000 | 0.824±0.002 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen3.5-9B (stale rules) | 0.982±0.005 | 0.982±0.005 | 0.947±0.011 | 0.711±0.009 | 1.000±0.000 | 0.691±0.009 | 3 seeds, n=5, bash, 4-bit |
| google/gemma-4-12B-it (stale rules) | 1.000±0.000 | 0.876±0.002 | 0.885±0.002 | 0.932±0.005 | 1.000±0.000 | 0.726±0.002 | 3 seeds, n=5, bash, 4-bit |
| ibm-granite/granite-4.1-3b (stale rules) | 0.862±0.002 | 1.000±0.000 | 0.750±0.000 | 0.621±0.016 | 1.000±0.000 | 0.529±0.016 | 3 seeds, n=5, bash, 4-bit |

An entry marked *stale tasks* was measured against a different version of this task file; one marked *stale rules* was graded by a different verifier, and *unstamped* predates the digest being recorded at all. Any of the three means the row is not comparable with the rest of the column. They are shown rather than deleted.

## Getting on it

Generate where the accelerator is, grade where the scheduler is, then import the cells:

```
anvil run --model <model id> --tasks tasks/t1_slurm.jsonl -n 5 --seed 0 \
  --save-generations gen_seed0.jsonl
./scripts/executor_ablation.sh <the directory holding the generations>
./scripts/leaderboard.py import <the per-cell result files> --seeds 0,1,2 --n 5
./scripts/leaderboard.py render
```

Grading outside the container is not accepted: the levels that depend on the scheduler
are the ones a wrong environment silently changes, which is recorded in
[OBSERVED_FAILURES.md](OBSERVED_FAILURES.md#a-table-measured-against-the-wrong-cluster).

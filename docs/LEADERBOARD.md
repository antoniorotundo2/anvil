# Leaderboard

Generated from `leaderboard/entries/`, never edited by hand: run
`./scripts/leaderboard.py render`. Every figure is `pass@1`, the mean across seeds with
half the range beside it. Half the range is not a confidence interval, and with three
seeds no significance is claimed anywhere on this page.

`strict_all_levels` is the ranking column: it requires every level either to pass or to
be out of the machine's reach, and a skipped level is never a passed one.

## `tasks/t1_slurm.jsonl`

| model | `syntax` | `submittability` | `functional` | `resource_fit` | `safety` | `strict_all_levels` | conditions |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-7B-Instruct | 1.000±0.000 | 0.792±0.025 | 0.875±0.000 | 1.000±0.000 | 1.000±0.000 | 0.667±0.025 | 3 seeds, n=5, bash, 4-bit |
| google/gemma-4-12B-it | 0.875±0.000 | 0.867±0.013 | 0.875±0.000 | 0.917±0.050 | 1.000±0.000 | 0.658±0.062 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen3.5-9B | 1.000±0.000 | 0.875±0.025 | 0.650±0.025 | 0.600±0.050 | 1.000±0.000 | 0.450±0.025 | 3 seeds, n=5, bash, 4-bit |
| ibm-granite/granite-4.1-3b | 1.000±0.000 | 0.875±0.000 | 0.842±0.037 | 0.550±0.000 | 1.000±0.000 | 0.425±0.000 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.575±0.025 | 0.842±0.013 | 0.533±0.037 | 0.442±0.013 | 1.000±0.000 | 0.308±0.025 | 3 seeds, n=5, bash, 4-bit |

## `tasks/t2_repair.jsonl`

| model | `syntax` | `submittability` | `functional` | `resource_fit` | `safety` | `strict_all_levels` | conditions |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-7B-Instruct | 0.983±0.002 | 0.977±0.000 | 0.870±0.002 | 0.965±0.007 | 1.000±0.000 | 0.824±0.002 | 3 seeds, n=5, bash, 4-bit |
| google/gemma-4-12B-it | 1.000±0.000 | 0.876±0.002 | 0.885±0.002 | 0.932±0.005 | 1.000±0.000 | 0.726±0.002 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen3.5-9B | 0.982±0.005 | 0.982±0.005 | 0.947±0.011 | 0.711±0.009 | 1.000±0.000 | 0.691±0.009 | 3 seeds, n=5, bash, 4-bit |
| ibm-granite/granite-4.1-3b | 0.862±0.002 | 1.000±0.000 | 0.750±0.000 | 0.621±0.016 | 1.000±0.000 | 0.529±0.016 | 3 seeds, n=5, bash, 4-bit |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.792±0.016 | 0.886±0.005 | 0.664±0.005 | 0.377±0.011 | 1.000±0.000 | 0.256±0.007 | 3 seeds, n=5, bash, 4-bit |

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

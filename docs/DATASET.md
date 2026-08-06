# The dataset

The files the benchmark is defined by, what is in each of them, and how to tell that the copy in
front of you is the one a published number was measured against.

Eighty-three records in eleven files, all JSONL, all UTF-8, all under 5 KB. It is a small dataset
and the size is the point: every task is hand-authored against a declared cluster topology, and
every fault in the repair set was mechanically induced from a solution that verifies clean and then
kept only if it actually broke.

Licence MIT, the same as the code.

## What is in it

**T1, writing a job script from scratch.** `tasks/t1_slurm.jsonl` holds eight tasks and
`tasks/t1_reference.jsonl` the canonical solution to each. A task record is:

| field | meaning |
|---|---|
| `id` | stable identifier, used by `--task` and in every result file |
| `prompt` | what the model is asked, in the words a support ticket would use |
| `constraints` | the resource spec: `nodes`, `ntasks`, `cpus_per_task`, `time_max_minutes`, `mem_min_mb`, `gpus_min`, any subset |
| `required_directives` | directives the task demands explicitly, even where SLURM has a default |
| `expects_in_body` | strings the script must print when it runs |
| `tags` | topic labels, used by the retrieval ablation and nothing else |

**T2, diagnosing and repairing.** `tasks/t2_repair.jsonl` holds 220 repair tasks: a broken script,
the fault category it carries, and the T1 task it must satisfy once repaired. It is not
hand-written. `anvil induce` applies seven fault injectors to the T1 reference solutions and keeps
only the variants that actually fail verification, because an inducer that produces an
accidentally-valid script is a bug in the inducer rather than a fault worth teaching a model to
repair. `tests/test_repair.py` holds the file to the current inducers, so it cannot drift from the
code that generated it.

**T3, Apptainer recipes.** `tasks/t3_apptainer.jsonl` and `tasks/t3_reference.jsonl`: a different
artifact type with the same shape, a definition file rather than a batch script.

**Two sets that exist for one measurement each.** `tasks/t1_exec.jsonl` states no memory minimum,
so the payload's real need is the only ground truth and a memory under-request can only be caught
by running the job; `tasks/t2_exec_repair.jsonl` is its induced counterpart.
`tasks/t1_coreutils.jsonl` asks for a character count that comes out the same on GNU coreutils and
on `uutils`, and is the one task in the set that two toolchains judge differently. Both live apart
from `tasks/t1_slurm.jsonl` on purpose: those eight are the denominator of every published T1
figure, and adding to them would move every number at once.

**The retrieval corpus.** `tasks/retrieval_corpus.jsonl`, eight short documents about SLURM
semantics, tagged by topic. It is the material the retrieval ablation attaches to prompts, not a
task set: nothing is scored against it.

## Telling which copy you have

`dataset/MANIFEST.json` records the SHA-256, size and record count of every released file. The
first twelve characters of that digest are what the harness writes into every generations file as
`tasks_sha`, and `anvil verify` refuses to grade generations whose digest does not match the task
set in front of it. So the same number identifies the dataset in three places: the manifest, the
saved generations, and the refusal.

```
./scripts/dataset_manifest.py --check
```

`make test` runs the same check, so a task edited without regenerating the manifest fails before it
can quietly invalidate a published figure.

Every number in [RESULTS.md](RESULTS.md) belongs to the digests currently in the manifest. If they
change, those figures need measuring again; regenerating the manifest and leaving the tables in
place would publish results for a dataset that no longer exists.

## Using it

```
pip install git+https://github.com/antoniorotundo2/anvil
anvil run --model oracle          # the upper bound: must score 1.0
anvil run --model broken          # the lower bound: must score 0.0 strict
anvil run --model <a hugging face model id>
```

The task files travel inside the package, so none of this needs a checkout. Grading is only
faithful where a scheduler and GNU coreutils are, which is what the container is for: see the
README.

## What it is not

Eight T1 tasks, which makes each one worth 0.125 of every T1 figure. One reference topology, four
nodes with one GPU each, declared by this project rather than borrowed from a real centre. Prompts
written by one author, in one style, in English. No adversarial or jailbreak material, no
multi-turn dialogue, no site-specific dialects of `#SBATCH`.

It contains no personal data and no third-party content: every prompt, solution and document was
written for this benchmark.

# Anvil

**Executable benchmarking of LLM-generated HPC operational artifacts.**

Anvil asks a question the existing benchmarks don't: *when an LLM writes the SLURM job
script or the container recipe that a supercomputer user actually needs — is it correct?*
Not "does it look like the reference answer" (cosine similarity), but **does it parse, does
the scheduler accept it, does it run, and does it request the right resources?**

> Status: **Phase 1** — task T1 (generation) for SLURM job scripts.
> Planned: T2 (diagnose-and-repair), Apptainer recipes, retrieval ablation. See Roadmap.

---

## Why

Assistants that help HPC users are appearing, but they are evaluated with **semantic
similarity metrics** because no validated HPC benchmark exists — their own authors say so.
Meanwhile, execution-based benchmarks are the norm everywhere else: parallel code, PDE
solvers, quantum SDKs. **Operational HPC artifacts are the empty slot.** Anvil fills it.

A wrong `--mem` doesn't look wrong. It looks *plausible*. Then the job dies at hour six.

---

## The verifier

Five independent levels, weakest to strongest:

| Level | Question | How |
|---|---|---|
| `syntax` | Is it a valid script? | shebang, `bash -n`, misplaced `#SBATCH` |
| `submittability` | Would SLURM accept it? | `sbatch --test-only` |
| `functional` | Does it run and exit 0? | sandboxed execution, expected output |
| `resource_fit` | Does it request what was asked? | parse directives vs. task constraints |
| `safety` | Is it dangerous? | destructive-pattern probes |

Two design choices carry the scientific weight:

- **`skipped` is never `passed`.** No SLURM on your laptop? `submittability` is skipped and
  scored as *not passed*. The metrics stay honest on any machine.
- **Dangerous scripts are never executed.** `safety` gates `functional`.

### The misplaced-directive check
SLURM stops reading `#SBATCH` lines at the first real command. Directives after it are
**silently ignored** — `sbatch` accepts the job and the request is wrong. Anvil catches this;
`sbatch --test-only` cannot.

---

## Quickstart

```bash
pip install -e ".[dev]"

# The oracle returns canonical solutions: it must score 1.0. If it doesn't, the
# benchmark is broken, not the model.
python -m anvil.cli run --model oracle --tasks tasks/t1_slurm.jsonl -v

# The broken model returns deliberately faulty artifacts: it must score 0.0 strict.
python -m anvil.cli run --model broken --tasks tasks/t1_slurm.jsonl -n 3

# A real model. CPU by default; uses your GPU automatically if present.
pip install -e ".[models]"
python -m anvil.cli run --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --tasks tasks/t1_slurm.jsonl -n 5 -k 1 --out results.json
```

Nothing above needs a cluster. Check what your machine can do:

```bash
python -m anvil.cli doctor
```

To unlock `submittability`, either install SLURM locally
(`sudo ./scripts/setup_slurm_single_node.sh`) or use the Linux container, which ships
with SLURM and gives faithful bash/coreutils behaviour on macOS:

```bash
docker build -t anvil docker/
docker run --rm -v "$PWD":/work -w /work anvil pytest -q
```

For batch runs on a GPU machine you only access occasionally:

```bash
N=5 SEEDS="0 1 2" ./scripts/run_experiments.sh
```

See [docs/HARDWARE.md](docs/HARDWARE.md) for the dev-machine / experiment-machine split.

---

## Oracle and broken model

Every benchmark should ship both, and few do.

- **Oracle** — canonical solutions. Proves the tasks are solvable and the verifier isn't
  too strict. CI fails if it drops below 1.0.
- **Broken** — faulty artifacts (missing shebang, misplaced directive, walltime overrun,
  `rm -rf /`, non-zero exit). Proves the verifier isn't too permissive. CI fails if it
  scores above 0.0 strict.

Together they bracket the verifier from both sides. Neither test is decorative: the
oracle caught a real bug during development, where the harness injected
`SLURM_CPUS_PER_TASK=1` into a task that requested 4 cores — the harness was contradicting
the spec it was checking. Environment variables are now derived from task constraints.

---

## Metrics

`pass@k` with the unbiased estimator (Chen et al., 2021), computed per level, plus
`strict_all_levels` (every non-skipped level passed).

---

## Roadmap

- [x] **Phase 1** — verifier (5 levels), 8 T1 tasks, oracle + broken, `pass@k`, CI
- [ ] **Phase 2** — T2 diagnose-and-repair; Apptainer recipes; retrieval ablation
      (zero-shot / vector / vectorless); failure-category breakdown
- [ ] **Phase 3** — QLoRA reference model; SSM arm (Mamba-Codestral); hybrid
      classical-quantum artifacts (simulator-verified)
- [ ] **Phase 4** — dataset release, leaderboard, preprint

Evaluated models will include a **state-space** model alongside transformers, to test
whether architecture matters for operational artifacts.

## Limitations

Failures in T2 will be **partly synthetic**, induced deliberately to obtain ground truth.
We anchor the taxonomy to published HPC-centre FAQs and to failure statistics from public
scheduler datasets, and we say so plainly. Without a cluster, `submittability` is skipped
and `functional` runs under `bash` rather than `sbatch`; results report which mode was used.

## License

MIT.

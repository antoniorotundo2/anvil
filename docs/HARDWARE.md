# Hardware: two machines, two roles

| | MacBook Pro 14" (M1 Pro) | Windows 11 PC |
|---|---|---|
| Access | daily | **occasional** |
| CPU | M1 Pro | i5-12400F |
| RAM | 16 GB **unified** | 64 GB DDR4-3200 |
| GPU | integrated (MPS) | RTX 3060 (Ampere) |
| Disk | 512 GB | 2 TB FireCuda 530 (PCIe Gen4) |
| **Role** | **development machine** | **experiment machine** |

The split is not a compromise: it mirrors an HPC centre, where you develop on your own laptop and
run jobs on nodes you do not control.

---

## Mac — development

This is where 90% of the time goes: authoring tasks, developing the verifier, running the tests,
using `oracle` and `broken`, writing the paper.

**What not to expect from the Mac.** The 16 GB are *unified*: CPU and GPU share them, leaving
roughly 10–11 GB for a model on MPS.

| Model | fp16 on MPS | Fits? |
|---|---|---|
| 1.5B | ~3 GB | yes, comfortably |
| 3B | ~6 GB | yes |
| 7B | ~14 GB | **no** |

So the Mac develops with models up to 3B. `bitsandbytes` does not support MPS: no 4-bit, and
`--load-in-4bit` fails with an explicit message (by design, not by accident).

**The disk is the silent constraint.** 512 GB go quickly: a single 7B in fp16 pulled from Hugging
Face weighs ~15 GB. Move the cache and watch it:

```bash
export HF_HOME="$HOME/hf_cache"      # better still: an external drive
du -sh "$HF_HOME"
huggingface-cli delete-cache
```

**The fidelity problem, and its fix.** macOS ships `bash` 3.2 and BSD coreutils. A generated script
using GNU-isms may pass on the Mac and fail on the cluster — or the reverse. The `functional` level
would return verdicts of no scientific value. For results that go in the paper, run the verifier in
the Linux container:

```bash
docker build -t anvil docker/
docker run --rm -v "$PWD":/work -w /work anvil pytest -q
docker run --rm -v "$PWD":/work -w /work anvil \
    python -m anvil.cli run --model oracle --tasks tasks/t1_slurm.jsonl -v
```

The image ships SLURM, so `submittability` is **active** there. Docker Desktop on Apple Silicon
builds the image natively for arm64.

---

## Windows PC — experiments

This is where the real numbers come from. The RTX 3060 is **Ampere**: it supports bf16, **not** FP8
(which needs Hopper, sm_90+) — the practical confirmation that the Transformer Engine was out of
reach.

VRAM: the desktop 3060 is typically 12 GB (an 8 GB variant exists). Check with `anvil doctor`,
which prints the detected VRAM. With 12 GB:

| Model | fp16 | 4-bit NF4 |
|---|---|---|
| 1.5B | yes | yes |
| 7B | **no** (~14 GB) | yes (~5 GB) |
| 14B | no | tight (~9 GB) |

So the 7B class is evaluated **in 4-bit**. Not a concession: quantization is already an ablation
axis, and now it has a hardware justification for the paper.

The 64 GB of system RAM allow CPU offload and many parallel verification sandboxes. The Gen4 SSD
makes weight loading negligible — put `HF_HOME` there, you have 2 TB.

### Recommended setup: WSL2

One configuration gives you **real SLURM + real GPU** on the same machine:

1. WSL2 with Ubuntu.
2. NVIDIA driver on Windows (CUDA passthrough into WSL2 needs no driver inside WSL).
3. Inside WSL2: `sudo ./scripts/setup_slurm_single_node.sh` → `submittability` active.
4. `pip install -e ".[models,quant]"` → 4-bit available.

Give WSL2 memory via `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=48GB
processors=10
```

Keep the model cache **inside the WSL2 filesystem** (`~/hf_cache`), not on `/mnt/c`: crossing the
filesystem boundary is very costly in I/O.

### Occasional access dictates the method

If you cannot iterate, do not iterate: **launch and walk away.**

```bash
N=5 SEEDS="0 1 2" ./scripts/run_experiments.sh
```

The script is **idempotent**: if the session is interrupted, rerunning resumes the missing runs.
Before spending GPU time it runs the guards (`oracle` must score 1.0, `broken` 0.0): if the
benchmark is broken you find out in two seconds instead of after an hour of inference. It records
`environment.json` on every run, because hardware must be declared in the paper.

---

## The plan, concretely

1. **On the Mac**, now: `pip install -e ".[dev]"`, then `anvil doctor`, then the tests. Develop the
   T2 tasks and the failure inducers with `oracle`/`broken`, which need no GPU.
2. **On the Mac**, for fidelity: build the container and rerun the suite inside it.
3. **On the Mac**, first real model: a 1.5B on MPS. This validates the *path*, not the numbers.
4. **On the PC**, when you get access: WSL2 + SLURM + `run_experiments.sh`. These are the paper's
   numbers, with multiple seeds and spread.
5. **In the paper**: declare both machines. Development on Apple Silicon, experiments on an
   RTX 3060 12 GB with NF4 quantization, single-node SLURM in WSL2. Honest, reproducible, and it
   shows you can tell development from measurement.

## What stays out of reach (and must be said)

- **FP8 / Transformer Engine**: needs Hopper. Not on Ampere, not on Apple Silicon.
- **Real multi-GPU DDP/DeepSpeed**: you have one GPU. You can prove the *correctness* of DDP code
  with `gloo` on CPU across two processes, and state that scaling is not measured. Real scaling
  numbers need Kaggle (2×T4) or an academic allocation.
- **Mamba-Codestral**: `mamba-ssm` and `causal-conv1d` require CUDA compilation. They do not build
  on the Mac. The SSM arm runs **only** on the PC.

#!/usr/bin/env bash
# Run the full experiment matrix (T1 from-scratch + T2 diagnose-and-repair) and
# walk away.
#
# Designed for a GPU machine you access OCCASIONALLY: launch it, leave, come back
# to collected results. Same discipline as an HPC job: you submit, you don't watch.
#
#   ./scripts/run_experiments.sh              # default matrix, T1 + T2
#   MODELS="Qwen/Qwen2.5-Coder-1.5B-Instruct" SEEDS="0 1" ./scripts/run_experiments.sh
#   RUN_T2=0 ./scripts/run_experiments.sh     # T1 only
#
# Results land in results/<timestamp>/ alongside the environment report: hardware
# must always be recorded, it feeds the paper's "setup" section. Each run also
# writes a companion *.generations.jsonl with the raw scripts, so the same
# generations can be re-verified against another base image later with
# `anvil verify` / `anvil verify-repair` (the cross-distribution ablation)
# without spending GPU time twice.

set -euo pipefail
cd "$(dirname "$0")/.."

# Put the model cache on a large disk (NOT the system disk if it is small: a 7B
# model in fp16 weighs ~15GB).
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

# Silences a known-harmless bitsandbytes FutureWarning (an internal, deprecated
# torch API call, not a project or user-facing issue) that repeats on every
# 4-bit generate() call and drowns the actual matrix progress output.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning:bitsandbytes.backends.cuda.ops}"

TASKS="${TASKS:-tasks/t1_slurm.jsonl}"
REPAIR_TASKS="${REPAIR_TASKS:-tasks/t2_repair.jsonl}"
RUN_T2="${RUN_T2:-1}"     # 1 = also run the T2 diagnose-and-repair matrix
N="${N:-5}"          # samples per task
K="${K:-1}"          # pass@k budget
SEEDS="${SEEDS:-0 1 2}"
FOURBIT="${FOURBIT:-1}"   # 1 = 4-bit quantization (requires CUDA)

# Models to evaluate. On a 12GB GPU: 7B fits in 4-bit, not in fp16.
MODELS="${MODELS:-Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/${STAMP}"
mkdir -p "$OUT"

echo "==> Environment"
python -m anvil.cli doctor --json | tee "${OUT}/environment.json" | head -20

echo
echo "==> Regression guards (before spending GPU time)"
python -m anvil.cli run --model oracle --tasks "$TASKS" --out "${OUT}/oracle.json" >/dev/null
python -m anvil.cli run --model broken --tasks "$TASKS" -n 3 --out "${OUT}/broken.json" >/dev/null
python - <<'PY'
import json, sys, glob, os
out = sorted(glob.glob("results/*/"))[-1]
o = json.load(open(os.path.join(out, "oracle.json")))["summary"]
b = json.load(open(os.path.join(out, "broken.json")))["summary"]
bad = [l for l in ("syntax", "functional", "resource_fit", "safety") if o[l]["pass@1"] != 1.0]
if bad:
    sys.exit(f"STOP: oracle not at 1.0 on {bad}. The benchmark is broken, not the models.")
if b["strict_all_levels"]["pass@1"] != 0.0:
    sys.exit("STOP: the verifier promotes defective artifacts.")
print("Guards OK: oracle 1.0, broken 0.0 strict.")
PY

if [[ "$RUN_T2" == "1" ]]; then
  echo
  echo "==> T2 regression guards (before spending GPU time)"
  python -m anvil.cli repair --model oracle --repair-tasks "$REPAIR_TASKS" --tasks "$TASKS" \
    --out "${OUT}/repair_oracle.json" >/dev/null
  python -m anvil.cli repair --model broken --repair-tasks "$REPAIR_TASKS" --tasks "$TASKS" \
    --out "${OUT}/repair_broken.json" >/dev/null
  python - "$OUT" <<'PY'
import json, sys, os
out = sys.argv[1]
o = json.load(open(os.path.join(out, "repair_oracle.json")))["summary"]
b = json.load(open(os.path.join(out, "repair_broken.json")))["summary"]
bad = [l for l in ("syntax", "functional", "resource_fit", "safety") if o[l]["pass@1"] != 1.0]
if bad:
    sys.exit(f"STOP: oracle repair not at 1.0 on {bad}. tasks/t2_repair.jsonl or the repair "
              "verifier is broken, not the models.")
if b["strict_all_levels"]["pass@1"] != 0.0:
    sys.exit("STOP: a no-op repair passes induced faults - the repair verifier is too permissive.")
print("T2 guards OK: oracle repair 1.0, no-op repair 0.0 strict.")
PY
fi

# A string, not an array: bash 3.2 (macOS's /usr/bin/bash) raises "unbound
# variable" under `set -u` when expanding "${empty_array[@]}". There is only
# ever one flag here, so word-splitting an unquoted string is safe and portable.
FLAGS=""
[[ "$FOURBIT" == "1" ]] && FLAGS="--load-in-4bit"

echo
echo "==> T1 matrix: $(echo "$MODELS" | wc -w) models x $(echo "$SEEDS" | wc -w) seeds, n=${N}"
for model in $MODELS; do
  safe="${model//\//_}"
  for seed in $SEEDS; do
    dest="${OUT}/${safe}__seed${seed}.json"
    if [[ -f "$dest" ]]; then
      echo "  [skip] ${safe} seed=${seed} (already present)"
      continue      # idempotent restart: if the session dies, resume
    fi
    echo "  [run ] ${safe} seed=${seed}"
    # --save-generations alongside --out: the raw scripts, not just the scores,
    # so the same generations can be re-verified against another base image
    # later (the cross-distribution ablation) without spending GPU time twice.
    python -m anvil.cli run \
      --model "$model" --tasks "$TASKS" \
      -n "$N" -k "$K" --seed "$seed" \
      $FLAGS --out "$dest" --save-generations "${dest%.json}.generations.jsonl" \
      || echo "  [FAIL] ${safe} seed=${seed}"
  done
done

if [[ "$RUN_T2" == "1" ]]; then
  echo
  echo "==> T2 matrix: $(echo "$MODELS" | wc -w) models x $(echo "$SEEDS" | wc -w) seeds, n=${N}"
  for model in $MODELS; do
    safe="${model//\//_}"
    for seed in $SEEDS; do
      dest="${OUT}/repair__${safe}__seed${seed}.json"
      if [[ -f "$dest" ]]; then
        echo "  [skip] repair ${safe} seed=${seed} (already present)"
        continue      # idempotent restart: if the session dies, resume
      fi
      echo "  [run ] repair ${safe} seed=${seed}"
      python -m anvil.cli repair \
        --model "$model" --repair-tasks "$REPAIR_TASKS" --tasks "$TASKS" \
        -n "$N" -k "$K" --seed "$seed" \
        $FLAGS --out "$dest" --save-generations "${dest%.json}.generations.jsonl" \
        || echo "  [FAIL] repair ${safe} seed=${seed}"
    done
  done
fi

echo
echo "==> T1 aggregation"
python - "$OUT" <<'PY'
import json, sys, glob, os
from pathlib import Path
out = Path(sys.argv[1])
rows = []
# "repair__*__seed*.json" also matches "*__seed*.json": exclude it, T2 has its
# own aggregation below.
for f in sorted(out.glob("*__seed*.json")):
    if f.name.startswith("repair__"):
        continue
    d = json.load(open(f))
    rows.append((d["model"], d["summary"]))
if not rows:
    sys.exit("no results")
models = sorted({m for m, _ in rows})
levels = ["syntax", "submittability", "functional", "resource_fit", "safety", "strict_all_levels"]
print(f"\n{'model':<40}" + "".join(f"{l[:12]:<14}" for l in levels))
print("-" * (40 + 14 * len(levels)))
for m in models:
    ss = [s for mm, s in rows if mm == m]
    line = f"{m:<40}"
    for l in levels:
        vals = [s[l][f"pass@{list(s[l])[0].split('@')[1]}"] for s in ss]
        mean = sum(vals) / len(vals)
        spread = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0
        line += f"{mean:.2f}±{spread:.2f}    "
    print(line)
json.dump({"models": models}, open(out / "index.json", "w"), indent=2)
print(f"\nResults in {out}/")
PY

if [[ "$RUN_T2" == "1" ]]; then
  echo
  echo "==> T2 aggregation"
  python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for f in sorted(out.glob("repair__*__seed*.json")):
    d = json.load(open(f))
    rows.append((d["model"], d["summary"]))
if not rows:
    sys.exit("no repair results")
models = sorted({m for m, _ in rows})
levels = ["syntax", "submittability", "functional", "resource_fit", "safety", "strict_all_levels"]
print(f"\n{'model':<40}" + "".join(f"{l[:12]:<14}" for l in levels))
print("-" * (40 + 14 * len(levels)))
for m in models:
    ss = [s for mm, s in rows if mm == m]
    line = f"{m:<40}"
    for l in levels:
        vals = [s[l][f"pass@{list(s[l])[0].split('@')[1]}"] for s in ss]
        mean = sum(vals) / len(vals)
        spread = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0
        line += f"{mean:.2f}±{spread:.2f}    "
    print(line)
print(f"\nRepair results in {out}/")
PY
fi

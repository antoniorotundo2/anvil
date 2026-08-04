#!/usr/bin/env bash
# Compare zero-shot / vector / vectorless retrieval on the same model, seeds
# and task set, and print pass@k side by side.
#
#   ./scripts/retrieval_ablation.sh
#   MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct SEEDS="0 1 2" N=5 ./scripts/retrieval_ablation.sh
#   STRATEGIES=vectorless CORPUS=results/reordered.jsonl ./scripts/retrieval_ablation.sh
#
# Every cell also writes its generated scripts beside its scores, so a finished sweep can
# be handed to crossdist_ablation.sh without spending inference time again.
#
# To resume an interrupted sweep, point OUT at the directory it was writing to:
#   OUT=results/retrieval_20260726_101500 ./scripts/retrieval_ablation.sh
# Cells already on disk are skipped. Without OUT each invocation starts a fresh
# directory, so the skip never fires and the sweep restarts from nothing: the
# header used to claim resumability the timestamp made impossible.

set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning:bitsandbytes.backends.cuda.ops}"

# Prefer the project venv, same rule as the Makefile. Ubuntu ships only `python3`, so a
# bare `python` is not merely the wrong interpreter, it does not exist: nine cells once
# failed in a row on `python: command not found`. Overridable for a different venv.
PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"

TASKS="${TASKS:-tasks/t1_slurm.jsonl}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
N="${N:-5}"
K="${K:-1}"
SEEDS="${SEEDS:-0 1 2}"
FOURBIT="${FOURBIT:-1}"
STRATEGIES="${STRATEGIES:-zero-shot vector vectorless}"

# A corpus other than the default one. What the arms retrieve depends on the order the
# documents sit in, not only on their content: `vectorless` fills its remaining slots from
# the documents tagged `general` in corpus order, so the first of them is attached to every
# task and the rest are never seen. Pointing this at a reordered copy is how that is
# measured rather than argued (see DESIGN.md, "What the level breaks on").
CORPUS="${CORPUS:-}"

# How many documents an arm may attach. Raising it is the way to tell a document's own
# effect from the effect of the one it displaced: at the default the general slot holds
# exactly one document, so swapping the corpus order gains one and loses another at the
# same time, and the two cannot be told apart.
RETRIEVAL_K="${RETRIEVAL_K:-}"

# Where the retrieved documents sit relative to the task. Every published arm used the
# default, `append`, which is also the only position the oracle could resolve before
# `models.OracleModel` learned to find the task prompt anywhere in what it receives.
RETRIEVAL_POSITION="${RETRIEVAL_POSITION:-}"

# A string, not an array: see run_experiments.sh for why (bash 3.2 on macOS).
FLAGS=""
[[ "$FOURBIT" == "1" ]] && FLAGS="--load-in-4bit"
[[ -n "$CORPUS" ]] && FLAGS="$FLAGS --retrieval-corpus $CORPUS"
[[ -n "$RETRIEVAL_K" ]] && FLAGS="$FLAGS --retrieval-k $RETRIEVAL_K"
[[ -n "$RETRIEVAL_POSITION" ]] && FLAGS="$FLAGS --retrieval-position $RETRIEVAL_POSITION"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/retrieval_${STAMP}}"
mkdir -p "$OUT"

echo "==> Retrieval ablation: ${MODEL}, $(echo "$SEEDS" | wc -w) seeds, n=${N}"
for strategy in $STRATEGIES; do
  for seed in $SEEDS; do
    safe="${strategy//-/_}"
    dest="${OUT}/${safe}__seed${seed}.json"
    if [[ -f "$dest" ]]; then
      echo "  [skip] ${strategy} seed=${seed} (already present)"
      continue      # idempotent restart: if the session dies, resume
    fi
    echo "  [run ] ${strategy} seed=${seed}"
    "$PYTHON" -m anvil.cli run --model "$MODEL" --tasks "$TASKS" \
      --retrieval "$strategy" -n "$N" -k "$K" --seed "$seed" \
      $FLAGS --out "$dest" --save-generations "${dest%.json}.generations.jsonl" \
      || echo "  [FAIL] ${strategy} seed=${seed}"
  done
done

echo
echo "==> Aggregation"
"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for f in sorted(out.glob("*__seed*.json")):
    d = json.load(open(f))
    rows.append((d.get("retrieval", "?"), d["summary"]))
if not rows:
    sys.exit("no results")

order = ("zero-shot", "vector", "vectorless")
strategies = sorted({s for s, _ in rows}, key=lambda s: order.index(s) if s in order else 99)
levels = ["syntax", "submittability", "functional", "resource_fit", "safety", "strict_all_levels"]
print(f"\n{'strategy':<14}" + "".join(f"{l[:12]:<14}" for l in levels))
print("-" * (14 + 14 * len(levels)))
for s in strategies:
    summaries = [summ for strat, summ in rows if strat == s]
    line = f"{s:<14}"
    for level in levels:
        vals = [x[level][f"pass@{list(x[level])[0].split('@')[1]}"] for x in summaries]
        mean = sum(vals) / len(vals)
        spread = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0
        line += f"{mean:.2f}±{spread:.2f}    "
    print(line)
print(f"\nResults in {out}/")
PY

#!/usr/bin/env bash
# Verify one set of generations inside several base images and report where the
# verdicts diverge. This is the ablation the generate/verify split exists for: the
# scripts are written once, on the machine with the accelerator, then judged again
# in each environment without spending inference time twice.
#
#   ./scripts/crossdist_ablation.sh results/20260724_124032
#   BASES="ubuntu:24.04 ubuntu:26.04" ./scripts/crossdist_ablation.sh results/<run>
#
# The argument is a directory holding *.generations.jsonl, which `run_experiments.sh`
# writes next to every cell of its matrix. Every seed present is compared, so the
# result is multi-seed whenever the run it reads was.
#
# To resume, point OUT at the directory an interrupted attempt was writing to:
#   OUT=results/crossdist_20260727_161500 ./scripts/crossdist_ablation.sh results/<run>
# Cells already on disk are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the project venv, same rule as the Makefile. Only the aggregation runs on the
# host; the verification itself runs inside the images.
PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"

TASKS="${TASKS:-tasks/t1_slurm.jsonl}"
BASES="${BASES:-ubuntu:24.04 ubuntu:26.04}"

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "usage: $0 <directory containing *.generations.jsonl>" >&2
  exit 2
fi

case "$(cd "$RUN_DIR" && pwd -P)" in
  "$(pwd -P)"/*) ;;
  *) echo "${RUN_DIR} is outside $(pwd -P)." >&2
     echo "Only the repository is mounted into the verification containers, so a path" >&2
     echo "outside it cannot be read there. Move or copy the run under results/." >&2
     exit 2 ;;
esac

shopt -s nullglob
GENERATIONS=("$RUN_DIR"/*.generations.jsonl)
if [[ ${#GENERATIONS[@]} -eq 0 ]]; then
  echo "no *.generations.jsonl in ${RUN_DIR}: run run_experiments.sh first," >&2
  echo "or pass the directory of a run that used --save-generations" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/crossdist_${STAMP}}"
mkdir -p "$OUT"

echo "==> Cross-distribution ablation"
echo "    generations: ${#GENERATIONS[@]} cells from ${RUN_DIR}"
echo "    base images: ${BASES}"

echo
echo "==> Images"
for base in $BASES; do
  tag="anvil:$(echo "$base" | tr ':.' '--')"
  echo "  [build] ${tag} from ${base}"
  docker build -q -t "$tag" --build-arg BASE_IMAGE="$base" docker/ >/dev/null
done

echo
echo "==> Verification"
for gen in "${GENERATIONS[@]}"; do
  cell="$(basename "$gen" .generations.jsonl)"
  for base in $BASES; do
    tag="anvil:$(echo "$base" | tr ':.' '--')"
    safe="$(echo "$base" | tr ':.' '--')"
    dest="${OUT}/${cell}__${safe}.json"
    if [[ -f "$dest" ]]; then
      echo "  [skip] ${cell} in ${base}"
      continue
    fi
    echo "  [run ] ${cell} in ${base}"
    docker run --rm -v "$PWD":/work -w /work "$tag" \
      python -m anvil.cli verify --generations "$gen" --tasks "$TASKS" --out "$dest" \
      >/dev/null || echo "  [FAIL] ${cell} in ${base}"
  done
done

echo
echo "==> Comparison"
"$PYTHON" - "$OUT" <<'PY'
"""Compare the verdicts image by image, per sample and per level.

Agreement between the summaries would be much weaker evidence: two environments can
reach the same pass@k while disagreeing about which samples pass. The per-sample
comparison is the claim worth making, and it is well defined because `verify` walks
the generations file in order, so index i is the same generated script everywhere.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
cells = defaultdict(dict)              # cell -> base -> data
for f in sorted(out.glob("*__*.json")):
    cell, base = f.stem.rsplit("__", 1)
    cells[cell][base] = json.load(open(f))
if not cells:
    sys.exit("no results to compare")

bases = sorted({b for per in cells.values() for b in per})
if len(bases) < 2:
    sys.exit(f"only one base image present ({bases}): nothing to compare")

# The ablation is vacuous unless the environments actually differ. Comparing an image
# with itself would "confirm portability" while testing nothing, the same trap the
# scheduler canary guards against on the submittability level.
print("environments compared:")
fingerprints = {}
for base in bases:
    env = next(per[base]["environment"] for per in cells.values() if base in per)
    fingerprints[base] = (env.get("coreutils"), env.get("gnu_faithful"), env.get("bash"))
    print(f"  {base:<16} coreutils={env.get('coreutils')}  "
          f"gnu_faithful={env.get('gnu_faithful')}  bash={env.get('bash')}")
if len(set(fingerprints.values())) == 1:
    print("\n  WARNING: every image reports the same toolchain. Agreement below would say")
    print("           nothing about portability. Check that BASE_IMAGE reached the build.")

samples = levels = divergent = 0
report = []
for cell, per in sorted(cells.items()):
    if len(per) < len(bases):
        report.append(f"  {cell}: incomplete, only {sorted(per)}")
        continue
    ref_base = bases[0]
    ref = per[ref_base]["results"]
    for other in bases[1:]:
        cmp = per[other]["results"]
        if len(ref) != len(cmp):
            report.append(f"  {cell}: {len(ref)} results in {ref_base} vs {len(cmp)} in {other}")
            continue
        for i, (a, b) in enumerate(zip(ref, cmp)):
            if a["task_id"] != b["task_id"]:
                report.append(f"  {cell}[{i}]: task_id mismatch, results are not aligned")
                continue
            samples += 1
            la = {x["level"]: x for x in a["levels"]}
            lb = {x["level"]: x for x in b["levels"]}
            for level in sorted(set(la) | set(lb)):
                levels += 1
                pa, pb = la.get(level), lb.get(level)
                if pa is None or pb is None:
                    divergent += 1
                    report.append(f"  {cell} {a['task_id']}[{i}] {level}: present in only one image")
                elif (pa["passed"], pa["skipped"]) != (pb["passed"], pb["skipped"]):
                    divergent += 1
                    report.append(
                        f"  {cell} {a['task_id']}[{i}] {level}: "
                        f"{ref_base} passed={pa['passed']} skipped={pa['skipped']} / "
                        f"{other} passed={pb['passed']} skipped={pb['skipped']}"
                    )

print(f"\n{len(cells)} cells, {samples} sample comparisons, {levels} level comparisons")
if divergent == 0 and not report:
    print("every level of every sample agreed across all base images")
else:
    print(f"{divergent} level comparisons diverged:")
    for line in report[:40]:
        print(line)
    if len(report) > 40:
        print(f"  ... and {len(report) - 40} more")

print(f"\nResults in {out}/")
PY

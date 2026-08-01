#!/usr/bin/env bash
# Verify one set of generations twice, once under each executor, and report the samples
# where the two disagree. This is the measurement the sbatch executor exists for: `bash`
# ignores every #SBATCH line and simulates three variables, real submission enforces the
# walltime and the allocation, so the interesting number is not either pass@k but the set
# of scripts one promotes and the other stops.
#
#   ./scripts/executor_ablation.sh results/<run>
#   SCHED_IMAGE=anvil:sched ./scripts/executor_ablation.sh results/<run>
#
# The argument is a directory holding *.generations.jsonl, which `run_experiments.sh`
# writes next to every cell of its matrix, T1 as `<model>__seed<N>.generations.jsonl` and
# T2 as `repair__<model>__seed<N>.generations.jsonl`. Every seed present is compared, so
# the result is multi-seed whenever the run it reads was.
#
# Both arms run inside the same image on purpose. Verifying with bash on the host and with
# sbatch in the container would vary the toolchain along with the executor, and any
# divergence would then have two candidate causes instead of one.
#
# To resume, point OUT at the directory an interrupted attempt was writing to:
#   OUT=results/executor_20260801_101500 ./scripts/executor_ablation.sh results/<run>
# Cells already on disk are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the project venv, same rule as the Makefile. Only the aggregation runs on the
# host; both verifications run inside the image.
PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"

TASKS="${TASKS:-tasks/t1_slurm.jsonl}"
REPAIR_TASKS="${REPAIR_TASKS:-tasks/t2_repair.jsonl}"
# Not the default image: that one accepts jobs and never runs them, so the sbatch arm
# would come back entirely skipped. See docs/REFERENCE_CLUSTER.md.
SCHED_IMAGE="${SCHED_IMAGE:-anvil:sched}"
# slurmd creates its cgroup scope under /sys/fs/cgroup, which a plain `docker run` mounts
# read-only, and the enforcement this ablation measures needs the controllers delegated.
DOCKER_RUN=(docker run --rm --privileged --cgroupns=host -v "$PWD":/work -w /work)

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

if ! docker image inspect "$SCHED_IMAGE" >/dev/null 2>&1; then
  echo "${SCHED_IMAGE} is not built: run \`make docker-build-sched\` first." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/executor_${STAMP}}"
mkdir -p "$OUT"

echo "==> Executor ablation"
echo "    generations: ${#GENERATIONS[@]} cells from ${RUN_DIR}"
echo "    image:       ${SCHED_IMAGE}"

echo
echo "==> Verification"
for gen in "${GENERATIONS[@]}"; do
  cell="$(basename "$gen" .generations.jsonl)"
  for executor in bash sbatch; do
    dest="${OUT}/${cell}__${executor}.json"
    if [[ -f "$dest" ]]; then
      echo "  [skip] ${cell} under ${executor}"
      continue
    fi
    echo "  [run ] ${cell} under ${executor}"
    # The repair cells carry a different verb and a second task file: their generations
    # are repairs of induced faults, graded by the same verifier as a from-scratch T1.
    if [[ "$(basename "$gen")" == repair__* ]]; then
      "${DOCKER_RUN[@]}" "$SCHED_IMAGE" python -m anvil.cli verify-repair \
        --generations "$gen" --repair-tasks "$REPAIR_TASKS" --tasks "$TASKS" \
        --executor "$executor" --out "$dest" >/dev/null \
        || echo "  [FAIL] ${cell} under ${executor}"
    else
      "${DOCKER_RUN[@]}" "$SCHED_IMAGE" python -m anvil.cli verify \
        --generations "$gen" --tasks "$TASKS" \
        --executor "$executor" --out "$dest" >/dev/null \
        || echo "  [FAIL] ${cell} under ${executor}"
    fi
  done
done

echo
echo "==> Comparison"
"$PYTHON" - "$OUT" <<'PY'
"""Compare the two executors sample by sample, and name what stopped each one.

Summary agreement would say almost nothing here: the two arms can reach the same pass@k
while disagreeing about which scripts run, and the disagreement is the finding. The
comparison is well defined because `verify` walks the generations file in order, so index
i is the same generated script under both executors.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

out = Path(sys.argv[1])
cells = defaultdict(dict)              # cell -> executor -> data
for f in sorted(out.glob("*__*.json")):
    cell, executor = f.stem.rsplit("__", 1)
    if executor in ("bash", "sbatch"):
        cells[cell][executor] = json.load(open(f))
if not cells:
    sys.exit("no results to compare")

print("environment:")
env = next(iter(next(iter(cells.values())).values()))["environment"]
print(f"  base_image={env.get('base_image')}  coreutils={env.get('coreutils')}  "
      f"sbatch={env.get('sbatch')}")

def level(result, name):
    return next((x for x in result["levels"] if x["level"] == name), None)

samples = 0
stopped = []          # passed under bash, failed under sbatch: the finding
unverifiable = 0      # skipped under sbatch: says nothing about the script
rescued = []          # failed under bash, passed under sbatch: worth knowing if ever
other_levels = 0      # any disagreement outside `functional`
reasons = Counter()

for cell, per in sorted(cells.items()):
    if len(per) < 2:
        print(f"  {cell}: incomplete, only {sorted(per)}")
        continue
    a, b = per["bash"]["results"], per["sbatch"]["results"]
    if len(a) != len(b):
        print(f"  {cell}: {len(a)} results under bash vs {len(b)} under sbatch")
        continue
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra["task_id"] != rb["task_id"]:
            print(f"  {cell}[{i}]: task_id mismatch, results are not aligned")
            continue
        samples += 1
        fa, fb = level(ra, "functional"), level(rb, "functional")
        if fa is None or fb is None:
            continue
        if fb["skipped"]:
            unverifiable += 1
        elif fa["passed"] and not fb["passed"]:
            stopped.append((cell, rb["task_id"], i, fb["detail"]))
            # The detail carries the job id, which is unique per sample; the terminal
            # state is what groups them into kinds of failure.
            for kind in ("OUT_OF_MEMORY", "TIMEOUT", "FAILED", "expected output not found",
                         "still running", "wrote nothing", "sbatch refused"):
                if kind in fb["detail"]:
                    reasons[kind] += 1
                    break
            else:
                reasons["other"] += 1
        elif fb["passed"] and not fa["passed"]:
            rescued.append((cell, rb["task_id"], i, fa["detail"]))
        for name in ("syntax", "submittability", "resource_fit", "safety"):
            la, lb = level(ra, name), level(rb, name)
            if la and lb and (la["passed"], la["skipped"]) != (lb["passed"], lb["skipped"]):
                other_levels += 1

print(f"\n{len(cells)} cells, {samples} sample comparisons")
print(f"  {len(stopped)} promoted by bash and stopped by real submission")
print(f"  {len(rescued)} failed under bash and passed under real submission")
print(f"  {unverifiable} not verifiable under sbatch (skipped, says nothing about the script)")
print(f"  {other_levels} disagreements outside `functional`, which should be zero: "
      "the other four levels do not depend on the executor")

if unverifiable == samples:
    print("\n  WARNING: every sample was skipped under sbatch, so this compares nothing.")
    print("           Check that the image runs jobs (make docker-guards-sbatch).")

if reasons:
    print("\nWhat stopped them:")
    for kind, count in reasons.most_common():
        print(f"  {count:>4}  {kind}")

if stopped:
    print("\nFirst samples that only real execution rejects:")
    for cell, task_id, i, detail in stopped[:15]:
        flat = " ".join(detail.split())[:100]
        print(f"  {cell} {task_id}[{i}]: {flat}")
    if len(stopped) > 15:
        print(f"  ... and {len(stopped) - 15} more")

if rescued:
    print("\nSamples the bash sandbox rejects and the real scheduler accepts:")
    for cell, task_id, i, detail in rescued[:5]:
        flat = " ".join(detail.split())[:100]
        print(f"  {cell} {task_id}[{i}]: {flat}")

print(f"\nResults in {out}/")
PY

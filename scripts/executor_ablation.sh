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

# Same knob as the Makefile: RUNTIME=podman runs these against Podman instead.
RUNTIME="${RUNTIME:-docker}"
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
DOCKER_RUN=("$RUNTIME" run --rm --privileged --cgroupns=host -v "$PWD":/work -w /work)

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

# Two different problems with one symptom: `"$RUNTIME" image inspect` fails the same way when
# the image is missing and when the daemon is down, and the second answer sends the reader
# to build an image that is already there.
if ! "$RUNTIME" info >/dev/null 2>&1; then
  echo "the ${RUNTIME} daemon is not reachable: start it and run this again." >&2
  exit 2
fi
if ! "$RUNTIME" image inspect "$SCHED_IMAGE" >/dev/null 2>&1; then
  echo "${SCHED_IMAGE} is not built: run \`make docker-build-sched\` first." >&2
  exit 2
fi

# The entrypoint is COPYed into the image, not mounted from the checkout, so pulling a fix
# to it changes nothing until the image is rebuilt. That cost an hour of verification once
# and gave back numbers identical to the run it was meant to correct, which is the worst
# shape a stale result can take: it looks like a finding. cksum is POSIX and behaves the
# same on both machines this runs on.
disk_sum="$(cksum <docker/entrypoint.sh | cut -d' ' -f1)"
image_sum="$("$RUNTIME" run --rm --entrypoint cat "$SCHED_IMAGE" /usr/local/bin/entrypoint.sh \
  2>/dev/null | cksum | cut -d' ' -f1)"
if [[ "$disk_sum" != "$image_sum" ]]; then
  echo "${SCHED_IMAGE} was built from a different docker/entrypoint.sh than the one here." >&2
  echo "Rebuild it before measuring: make docker-build-sched" >&2
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

# A run assembled over days can pick up a verifier change halfway, and the two gradings
# then differ for a reason that has nothing to do with the executor. That is the whole
# point of the comparison, so it has to be ruled out rather than assumed.
rules = {d.get("verifier_sha", "unstamped") for per in cells.values() for d in per.values()}
if len(rules) > 1:
    sys.exit(f"reports were graded by different verifiers, not comparable: {sorted(rules)}")
print(f"  verifier_sha={rules.pop()}")

def level(result, name):
    return next((x for x in result["levels"] if x["level"] == name), None)

samples = 0
stopped = []          # passed under bash, failed under sbatch: the finding
unverifiable = 0      # bash could judge it and sbatch could not
rescued = []          # failed under bash, passed under sbatch: worth knowing if ever
other_levels = 0      # any disagreement outside `functional`
strict_flips = []     # the verdict on the whole artifact changed
reasons = Counter()
# summary[family][model][level][executor] -> list of pass@1, one per seed
scores: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

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
        # Skipped under both arms is not the executor's doing: a script that fails
        # `syntax` never reaches either. Counting those here once suggested a third of
        # the run was beyond real submission's reach.
        if fb["skipped"]:
            unverifiable += 0 if fa["skipped"] else 1
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
        if ra["all_passed"] != rb["all_passed"]:
            strict_flips.append((cell, rb["task_id"], i, rb["all_passed"]))
        for name in ("syntax", "submittability", "resource_fit", "safety"):
            la, lb = level(ra, name), level(rb, name)
            if la and lb and (la["passed"], la["skipped"]) != (lb["passed"], lb["skipped"]):
                other_levels += 1

# What each arm says, not only where they differ. The `bash` column is the corrected
# table: these cells were graded inside the image, against the declared topology, which
# is the whole reason the run is verified here rather than wherever it was generated.
for cell, per in sorted(cells.items()):
    family = "T2 repair" if cell.startswith("repair__") else "T1"
    model = cell[len("repair__"):] if cell.startswith("repair__") else cell
    model = model.rsplit("__seed", 1)[0]
    for executor, data in per.items():
        for level, row in data["summary"].items():
            scores[family][model][level][executor].append(row["pass@1"])

for family in sorted(scores):
    print(f"\n{family}: pass@1 per arm, mean over seeds (half-range in brackets)")
    levels = ["syntax", "submittability", "functional", "resource_fit", "safety",
              "strict_all_levels"]
    print(f"  {'model':<34}{'level':<20}{'bash':>16}{'sbatch':>16}")
    for model in sorted(scores[family]):
        for level in levels:
            cell_scores = scores[family][model][level]
            if not cell_scores:
                continue
            out_cols = []
            for executor in ("bash", "sbatch"):
                vals = cell_scores.get(executor, [])
                if not vals:
                    out_cols.append("n/a".rjust(16))
                    continue
                mean = sum(vals) / len(vals)
                half = (max(vals) - min(vals)) / 2
                out_cols.append(f"{mean:.3f}+-{half:.3f}".rjust(16))
            print(f"  {model[:33]:<34}{level:<20}" + "".join(out_cols))

print(f"\n{len(cells)} cells, {samples} sample comparisons")
print(f"  {len(strict_flips)} artifacts whose strict verdict changes with the executor")
print(f"  {len(stopped)} promoted by bash and stopped by real submission")
print(f"  {len(rescued)} failed under bash and passed under real submission")
print(f"  {unverifiable} judged by bash and not by sbatch (skipped, says nothing "
      "about the script)")
print(f"  {other_levels} disagreements outside `functional`, which should be zero: "
      "the other four levels do not depend on the executor")

if unverifiable == samples:
    print("\n  WARNING: every sample was skipped under sbatch, so this compares nothing.")
    print("           Check that the image runs jobs (make docker-guards-sbatch).")

if reasons:
    print("\nWhat stopped them:")
    for kind, count in reasons.most_common():
        print(f"  {count:>4}  {kind}")

if strict_flips:
    print("\nArtifacts the two executors disagree about, as whole artifacts:")
    for cell, task_id, i, now_passes in strict_flips[:10]:
        verdict = "only real submission accepts it" if now_passes else "only bash accepts it"
        print(f"  {cell} {task_id}[{i}]: {verdict}")
    if len(strict_flips) > 10:
        print(f"  ... and {len(strict_flips) - 10} more")

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

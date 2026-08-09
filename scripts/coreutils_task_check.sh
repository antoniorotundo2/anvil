#!/usr/bin/env bash
# Can the benchmark see a coreutils divergence at all?
#
#   ./scripts/coreutils_task_check.sh
#   BASES="ubuntu:24.04 ubuntu:26.04" ./scripts/coreutils_task_check.sh
#
# `scripts/coreutils_divergence.sh` shows that GNU 9.4 and uutils 0.8.0 answer `wc -m`
# differently on a non-ASCII payload under the C locale. That is a fact about the
# toolchains; this script asks whether the harness can turn it into a verdict.
#
# Two assertions, and the second is the one worth having:
#
#   the reference solution passes in both images. It pins a UTF-8 locale, so the count
#   means the same thing on either implementation, and a task whose own solution were
#   environment-dependent would be a defective task rather than a sensitive one;
#
#   the same script with `LC_ALL=C` instead passes in one image and fails in the other.
#   Every other T1 task returns the same verdict in both, which is what the
#   cross-distribution ablation reports, and that agreement has never been able to
#   distinguish two interchangeable toolchains from eight tasks that never ask.
#
# The non-portable variant is derived from the reference by substitution rather than
# written out, so the two cannot drift apart into a comparison of two different scripts.
#
# This is a bracket, not evidence: it is meant to fail loudly if the task stops being
# able to tell the implementations apart.

set -euo pipefail

# Same knob as the Makefile: RUNTIME=podman runs this against Podman instead.
RUNTIME="${RUNTIME:-docker}"
cd "$(dirname "$0")/.."

BASES="${BASES:-ubuntu:24.04 ubuntu:26.04}"
TASKS="tasks/t1_coreutils.jsonl"
PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"

if ! "$RUNTIME" info >/dev/null 2>&1; then
  echo "the ${RUNTIME} daemon is not reachable: start it and run this again." >&2
  exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Both artifacts as generations, so `anvil verify` grades them exactly as it grades a
# model's output: same levels, same executor, no special path for a hand-written script.
"$PYTHON" - "$WORK" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from anvil.cli import _file_sha  # noqa: E402

out = Path(sys.argv[1])
ref = json.loads(Path("tasks/t1_coreutils_reference.jsonl").read_text(encoding="utf-8"))
portable = ref["script"]
naive = portable.replace("export LC_ALL=C.utf8\n", "export LC_ALL=C\n")
if naive == portable:
    sys.exit("the reference no longer pins LC_ALL=C.utf8: this check has nothing to vary")

rows = []
for sample, script in enumerate((portable, naive)):
    rows.append({
        "task_id": ref["id"],
        "sample": sample,
        "model": "coreutils-task-check",
        "seed": 0,
        # `verify` refuses generations whose task file is not the one in front of it, and
        # a hand-written pair has to satisfy that rule like any other.
        "tasks_sha": _file_sha("tasks/t1_coreutils.jsonl"),
        "script": script,
    })
out.joinpath("gen.jsonl").write_text(
    "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
)
PY

echo "==> Coreutils task check"
echo "    task:        ${TASKS}"
echo "    base images: ${BASES}"
echo

for base in $BASES; do
  tag="anvil:$(echo "$base" | tr ':.' '--')"
  echo "  [build] ${tag} from ${base}"
  "$RUNTIME" build -q -t "$tag" --build-arg BASE_IMAGE="$base" docker/ >/dev/null
  "$RUNTIME" run --rm -v "$PWD":/work -v "$WORK":/gen -w /work "$tag" \
    python -m anvil.cli verify --generations /gen/gen.jsonl --tasks "$TASKS" \
    --out "/gen/$(echo "$base" | tr ':.' '--').json" >/dev/null
done

echo
"$PYTHON" - "$WORK" $BASES <<'PY'
import json
import sys
from pathlib import Path

work, bases = Path(sys.argv[1]), sys.argv[2:]
verdicts: dict[str, list[bool]] = {}
for base in bases:
    payload = json.loads(work.joinpath(base.replace(":", "-").replace(".", "-") + ".json")
                         .read_text(encoding="utf-8"))
    # Sample 0 is the reference, sample 1 the LC_ALL=C variant, in the order written above.
    verdicts[base] = [
        all(lv["passed"] or (lv["skipped"] and lv.get("skip_scope") == "environment")
            for lv in r["levels"])
        for r in payload["results"]
    ]

for base in bases:
    portable, naive = verdicts[base]
    print(f"  {base:<14} reference={'pass' if portable else 'FAIL'}   "
          f"LC_ALL=C variant={'pass' if naive else 'fail'}")

problems = [b for b in bases if not verdicts[b][0]]
if problems:
    sys.exit(f"\nthe reference solution fails in {', '.join(problems)}: the task is defective, "
             "not sensitive")

naive_verdicts = {verdicts[b][1] for b in bases}
if len(naive_verdicts) < 2:
    sys.exit("\nthe LC_ALL=C variant is judged the same way in every image: this task no longer "
             "distinguishes the toolchains, which is the only reason it exists")

print("\nCoreutils task check OK: the reference passes everywhere and the same script with "
      "LC_ALL=C\ndoes not, so a toolchain divergence reaches the verdict.")
PY

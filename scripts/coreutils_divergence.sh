#!/usr/bin/env bash
# Where do GNU coreutils and uutils actually differ, on the operations a job script uses?
#
#   ./scripts/coreutils_divergence.sh
#   BASES="ubuntu:24.04 ubuntu:26.04" ./scripts/coreutils_divergence.sh
#
# The cross-distribution ablation reports zero divergence across 360 level comparisons, and
# that number is only as strong as the tasks behind it: none of the eight touches a place
# where the two implementations part company, so the agreement measures the task set as
# much as the toolchains. This script asks the question directly, with no tasks and no
# model in the way: run the same 73 invocations in both images and diff the output.
#
# It is evidence for a claim, not part of the bracket. Nothing here touches the verifier.

set -euo pipefail
cd "$(dirname "$0")/.."

BASES="${BASES:-ubuntu:24.04 ubuntu:26.04}"
PROBE="$(mktemp -d)/probe.sh"
trap 'rm -rf "$(dirname "$PROBE")"' EXIT

# Two families on purpose. The first is what a job script does with its results: sort,
# count, format, slice, hash. The second is what a *careful* job script does around them:
# check exit codes, read error text, set a locale. Divergences hide in the second.
cat >"$PROBE" <<'PROBE_END'
run() { printf '%-30s|%s\n' "$1" "$(eval "$2" 2>&1 | head -2 | tr '\n' '~')"; }
cd /tmp && mkdir -p probe && cd probe
printf 'b\na\nB\nA\n10\n9\n' > f.txt

run "sort"                    "sort f.txt | tr '\n' ' '"
run "sort -n"                 "sort -n f.txt | tr '\n' ' '"
run "sort -V"                 "printf '1.10\n1.9\n1.2\n' | sort -V | tr '\n' ' '"
run "sort -u -f"              "sort -uf f.txt | tr '\n' ' '"
run "sort -h"                 "printf '2K\n1M\n900\n' | sort -h | tr '\n' ' '"
run "sort -s -k1,1"           "printf 'b 1\na 2\nb 0\n' | sort -s -k1,1 | tr '\n' '~'"
run "sort -t: -k2,2n"         "printf 'a:2\nb:1\n' | sort -t: -k2,2n | tr '\n' ' '"
run "wc -l"                   "wc -l < f.txt"
run "wc -L"                   "wc -L < f.txt"
run "uniq -c"                 "printf 'a\na\nb\n' | uniq -c | tr -s ' ' | tr '\n' '~'"
run "cut -d -f"               "echo 'a:b:c' | cut -d: -f2,3"
run "tr -s"                   "echo 'aaabbb' | tr -s 'ab'"
run "head -c"                 "head -c 3 f.txt"
run "tail -c +2"              "tail -c +2 f.txt | head -1"
run "fold -w"                 "echo abcdef | fold -w2 | tr '\n' ' '"
run "paste -d"                "printf 'a\nb\n' | paste -d, - -"
run "join"                    "printf '1 a\n' > j1; printf '1 x\n' > j2; join j1 j2"
run "comm"                    "printf 'a\nb\n' > c1; printf 'b\nc\n' > c2; comm c1 c2 | tr '\n' '~'"
run "split -n"                "printf 'abcdef' > s.txt; split -n 2 s.txt sp_ && ls sp_* | tr '\n' ' '"
run "expand -t4"              "printf 'a\tb\n' | expand -t4 | cat -A"
run "od -An -tx1"             "printf 'AB' | od -An -tx1"
run "base64 -w0"              "printf abc | base64 -w0"
run "sha256sum"               "sha256sum f.txt | cut -c1-16"
run "du -h"                   "du -h f.txt"
run "du -B1 --apparent-size"  "du -B1 --apparent-size f.txt"
run "df -h /"                 "df -h / | tail -1 | awk '{print \$5}'"
run "df -x tmpfs"             "df -x tmpfs -h / | tail -1 | awk '{print \$1}'"
run "stat -c '%s %F'"         "stat -c '%s %F' f.txt"
run "truncate -s"             "truncate -s 3 t.bin && stat -c %s t.bin"
run "readlink -f"             "readlink -f ./f.txt"
run "realpath --relative-to"  "realpath --relative-to=/tmp /tmp/probe/f.txt"
run "ls -l field count"       "ls -l f.txt | awk '{print NF}'"
run "ls --time-style"         "ls --time-style=+%Y f.txt"
run "mktemp -d"               "d=\$(mktemp -d) && test -d \$d && echo ok"
run "install -D"              "install -D f.txt d1/d2/f.txt && echo ok"
run "cp --reflink=auto"       "cp --reflink=auto f.txt g.txt && echo ok"
run "seq -w"                  "seq -w 8 11 | tr '\n' ' '"
run "seq -s"                  "seq -s, 1 4"
run "seq float"               "seq 0 0.5 1.5 | tr '\n' ' '"
run "printf %b octal"         "printf '%b' 'a\\0142c'"
run "printf %-5s"             "printf '[%-5s]' ab"
run "printf %q"               "printf '%q' 'a b'"
run "numfmt --to=iec"         "numfmt --to=iec 1536"
run "shuf --random-source"    "printf '1\n2\n3\n' | shuf --random-source=/dev/zero | tr '\n' ' '"
run "xargs -I"                "echo a b | xargs -I{} echo '[{}]'"
run "timeout"                 "timeout 1 sleep 0.1 && echo ok"
run "nproc"                   "nproc >/dev/null && echo ok"
run "date -d absolute"        "TZ=UTC date -u -d '2026-01-02 03:04:05' +%Y%m%dT%H%M%S"
run "date -R"                 "TZ=UTC date -R -d @0"
run "date -d relative"        "TZ=UTC date -u -d 'now + 1 hour' +%H >/dev/null && echo ok"
run "date '+%-d %j %U %s'"    "TZ=UTC date -u -d @0 '+%-d %j %U %s'"
run "date +%N width"          "date -u +%N | wc -c"

# What a careful script does around the work: check codes, read messages, set a locale.
run "rc: cat missing"         "cat /nope; echo rc=\$?"
run "rc: mkdir exists"        "mkdir /tmp; echo rc=\$?"
run "rc: rm missing"          "rm /nope; echo rc=\$?"
run "rc: stat missing"        "stat /nope; echo rc=\$?"
run "rc: ls bad flag"         "ls --nonsense; echo rc=\$?"
run "rc: cp no dest"          "cp f.txt; echo rc=\$?"
run "rc: head bad number"     "head -c abc f.txt; echo rc=\$?"
run "rc: sort bad key"        "sort -k f.txt; echo rc=\$?"
run "rc: du missing"          "du /nope; echo rc=\$?"
run "rc: date bad input"      "date -d 'not a date'; echo rc=\$?"
run "rc: timeout expiry"      "timeout 0.1 sleep 5; echo rc=\$?"
run "rc: timeout --preserve"  "timeout --preserve-status 0.1 sleep 5; echo rc=\$?"
run "rc: tail -f killed"      "timeout 0.3 tail -f f.txt >/dev/null; echo rc=\$?"
run "locale -a"               "locale -a | tr '\n' ' '"
run "sort LC_ALL=C"           "LC_ALL=C sort f.txt | tr '\n' ' '"
run "sort LC_ALL=en_US.UTF-8" "LC_ALL=en_US.UTF-8 sort f.txt | tr '\n' ' '"
run "sort inherited locale"   "sort f.txt | tr '\n' ' '"
run "numfmt --grouping C"     "LC_ALL=C numfmt --grouping 1234567"
run "numfmt --grouping en_US" "LC_ALL=en_US.UTF-8 numfmt --grouping 1234567"
run "ls --version"            "ls --version | head -1"
PROBE_END

work="$(dirname "$PROBE")"
for base in $BASES; do
  echo "==> ${base}"
  docker run --rm -v "${work}":/probe "$base" bash /probe/probe.sh \
    >"${work}/$(echo "$base" | tr ':/' '--').txt" 2>/dev/null
done

first=""
echo
echo "==> Divergences"
for base in $BASES; do
  out="${work}/$(echo "$base" | tr ':/' '--').txt"
  if [[ -z "$first" ]]; then
    first="$out"
    echo "    reference: ${base}, $(wc -l <"$first") invocations"
    continue
  fi
  if diff -q "$first" "$out" >/dev/null; then
    echo "    ${base}: identical on every invocation"
  else
    echo "    ${base}:"
    diff "$first" "$out" | sed 's/^/      /'
  fi
done

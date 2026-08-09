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
# model in the way: run the same 101 invocations in both images and diff the output.
#
# It is evidence for a claim, not part of the bracket. Nothing here touches the verifier.

set -euo pipefail

# Same knob as the Makefile: RUNTIME=podman runs these against Podman instead.
RUNTIME="${RUNTIME:-docker}"
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

# A third family, chosen differently from the first two. Those were picked blind, for
# being what job scripts do, and turned up nothing behavioural. These were picked to
# hunt, after the first sweep: byte-versus-character width, number formatting, and the
# newer digest and slicing flags, which is where two independent implementations of the
# same tools have room to disagree without either being wrong.
printf 'caf\xc3\xa9 na\xc3\xafve\n' > u.txt

run "wc -m utf8"              "wc -m < u.txt"
run "wc -c utf8"              "wc -c < u.txt"
run "cut -c2-4 utf8"          "cut -c2-4 u.txt"
run "cut -b2-4 utf8"          "cut -b2-4 u.txt | od -An -tx1"
run "fold -w3 utf8"           "fold -w3 u.txt | head -1 | od -An -tx1"
run "tr class utf8"           "tr '[:lower:]' '[:upper:]' < u.txt"
run "expand -t4 before utf8"  "printf 'caf\xc3\xa9\tx\n' | expand -t4 | cat -A"
run "sort accents LC_ALL=C"   "printf 'e\n\xc3\xa9\nf\n' | LC_ALL=C sort | od -An -tx1 | tr '\n' '~'"
run "numfmt --to=si"          "numfmt --to=si 1536"
run "numfmt --from=auto"      "numfmt --from=auto 1.5K"
run "numfmt --to=iec --format" "numfmt --to=iec --format='%8.2f' 1536"
run "seq -f %03g"             "seq -f '%03g' 1 3 | tr '\n' ' '"
run "sort -g"                 "printf '1e3\n5\n2e1\n' | sort -g | tr '\n' ' '"
run "sort -n signed"          "printf '+2\n-1\n 3\n' | sort -n | tr '\n' '~'"
run "sort -h negative"        "printf -- '-1K\n2K\n-3M\n' | sort -h | tr '\n' ' '"
run "sort -k2.2,2.3"          "printf 'x abcd\nx abzz\n' | sort -k2.2,2.3 | tr '\n' '~'"
run "printf %5.2f"            "printf '%5.2f' 3.14159"
run "printf %d hex"           "printf '%d' 0x1f"
run "md5sum --tag"            "md5sum --tag f.txt | cut -c1-12"
run "cksum -a sha256"         "cksum -a sha256 f.txt 2>&1 | cut -c1-20"
run "b2sum"                   "b2sum f.txt 2>&1 | cut -c1-12"
run "basenc --base32"         "printf abc | basenc --base32 2>&1"
run "ls -v"                   "touch v1 v10 v2; ls -v v1 v10 v2 | tr '\n' ' '"
run "head -n -1"              "head -n -1 f.txt | tr '\n' ' '"
run "uniq --all-repeated"     "printf 'a\na\nb\n' | uniq --all-repeated=separate | tr '\n' '~'"
run "split -n l/2"            "printf 'a\nb\nc\nd\n' > s2.txt; split -n l/2 s2.txt L_ && head -c 4 L_aa | tr '\n' ' '"
run "join -a1 -e X -o"        "printf '1 a\n2 b\n' > k1; printf '1 x\n' > k2; join -a1 -e X -o 0,1.2,2.2 k1 k2 | tr '\n' '~'"
run "date +%q"                "TZ=UTC date -u -d @0 +%q 2>&1"

# Not coreutils, and here on purpose: the images differ in bash as well (5.2 against
# 5.3), so a divergence found by this script is only attributable to coreutils if the
# shell is ruled out. This one is the shell.
run "bash: printf overflow"   "printf '%d' 99999999999999999999; echo rc=\$?"
PROBE_END

work="$(dirname "$PROBE")"
for base in $BASES; do
  echo "==> ${base}"
  "$RUNTIME" run --rm -v "${work}":/probe "$base" bash /probe/probe.sh \
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

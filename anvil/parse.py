"""Extract the script from raw LLM output and parse #SBATCH directives.

Extraction is deliberately permissive (models fence code in unpredictable ways);
validation is strict (see checks in verifier.py).
"""

from __future__ import annotations

import re
import shlex

# ```bash ... ```  or ``` ... ```
_FENCE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)```", re.DOTALL)

# SLURM time formats: mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm, d-hh:mm:ss
_TIME_PATTERNS = [
    (re.compile(r"^(\d+)$"), lambda m: int(m[1])),                          # mm
    (re.compile(r"^(\d+):(\d+)$"), lambda m: int(m[1])),                    # mm:ss
    (re.compile(r"^(\d+):(\d+):(\d+)$"),
     lambda m: int(m[1]) * 60 + int(m[2])),                                 # hh:mm:ss
    (re.compile(r"^(\d+)-(\d+)$"),
     lambda m: int(m[1]) * 1440 + int(m[2]) * 60),                          # d-hh
    (re.compile(r"^(\d+)-(\d+):(\d+)$"),
     lambda m: int(m[1]) * 1440 + int(m[2]) * 60 + int(m[3])),              # d-hh:mm
    (re.compile(r"^(\d+)-(\d+):(\d+):(\d+)$"),
     lambda m: int(m[1]) * 1440 + int(m[2]) * 60 + int(m[3])),              # d-hh:mm:ss
]

_MEM_RE = re.compile(r"^(\d+)([KMGT]?)B?$", re.IGNORECASE)
_MEM_MULT = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}


def extract_script(raw: str) -> str:
    """Recover the shell script from raw model output."""
    fences = _FENCE.findall(raw)
    if fences:
        # with several blocks, pick the one that looks like a job script
        for block in fences:
            if "#SBATCH" in block or block.lstrip().startswith("#!"):
                return block.strip("\n")
        return fences[0].strip("\n")
    return raw.strip()


def parse_time_to_minutes(value: str) -> int | None:
    value = value.strip()
    for pat, fn in _TIME_PATTERNS:
        m = pat.match(value)
        if m:
            return fn(m)
    return None


def parse_mem_to_mb(value: str) -> float | None:
    m = _MEM_RE.match(value.strip())
    if not m:
        return None
    num, unit = int(m.group(1)), m.group(2).upper()
    return num * _MEM_MULT[unit]


# SLURM short options and their long equivalents. `-t` and `--time` are the same
# request; a benchmark that measures semantics must not care which was written.
_SHORT_TO_LONG = {
    "-N": "--nodes",
    "-n": "--ntasks",
    "-c": "--cpus-per-task",
    "-t": "--time",
    "-G": "--gpus",
    "-J": "--job-name",
    "-o": "--output",
    "-e": "--error",
    "-a": "--array",
    "-d": "--dependency",
    "-p": "--partition",
    "-A": "--account",
    "-w": "--nodelist",
    "-D": "--chdir",
    "-C": "--constraint",
}


def _normalise(key: str) -> str:
    return _SHORT_TO_LONG.get(key, key)


def parse_directives(script: str) -> dict[str, str]:
    """Collect #SBATCH directives into a {--long-key: value} mapping.

    Two things a naive parser gets wrong, and both produce false negatives:

    1. **SLURM allows several options on one `#SBATCH` line** - the line is parsed
       like a command line. Reading only the first option swallows the rest:
       `#SBATCH --ntasks=1 --time=00:01:00` would report `--time` as missing while
       `sbatch` reports its value as invalid. The two disagree, and the parser is wrong.
    2. **Short and long forms are the same request.** `-t` is `--time`. Demanding the
       long spelling measures style, not correctness.

    Note: SLURM stops reading directives at the first real command. We collect them
    all here; `misplaced_directives` reports the late ones.
    """
    out: dict[str, str] = {}
    for line in script.splitlines():
        s = line.strip()
        if not s.startswith("#SBATCH"):
            continue
        body = s[len("#SBATCH"):].strip()
        if not body:
            continue
        body = body.split("#", 1)[0].strip()   # trailing comment
        try:
            tokens = shlex.split(body)
        except ValueError:                     # unbalanced quotes: fall back
            tokens = body.split()

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if not tok.startswith("-"):        # stray word (e.g. leaked prose)
                i += 1
                continue

            if tok.startswith("--"):
                if "=" in tok:
                    key, _, val = tok.partition("=")
                    out[_normalise(key)] = val
                    i += 1
                    continue
                key = tok
            else:                              # short option
                if len(tok) > 2:               # attached value, e.g. -c4
                    out[_normalise(tok[:2])] = tok[2:]
                    i += 1
                    continue
                key = tok

            # value is the next token, unless it is another option (i.e. a flag)
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                out[_normalise(key)] = tokens[i + 1]
                i += 2
            else:
                out[_normalise(key)] = ""
                i += 1
    return out


def misplaced_directives(script: str) -> list[str]:
    """#SBATCH lines appearing AFTER the first command: SLURM silently ignores
    them. A real and frequent error, invisible to `sbatch --test-only`."""
    seen_command = False
    bad: list[str] = []
    for line in script.splitlines():
        s = line.strip()
        if not s or s.startswith("#!"):
            continue
        if s.startswith("#SBATCH"):
            if seen_command:
                bad.append(s)
            continue
        if s.startswith("#"):
            continue
        seen_command = True
    return bad


def directive_value(directives: dict[str, str], *aliases: str) -> str | None:
    for a in aliases:
        if a in directives:
            return directives[a]
    return None

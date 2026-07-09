"""Extract the script from raw LLM output and parse #SBATCH directives.

Extraction is deliberately permissive (models fence code in unpredictable ways);
validation is strict (see checks in verifier.py).
"""

from __future__ import annotations

import re

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


def parse_directives(script: str) -> dict[str, str]:
    """Collect #SBATCH directives into a {--key: value} mapping.

    Note: SLURM stops reading directives at the first real command. We collect
    them all here; `misplaced_directives` reports the late ones.
    """
    out: dict[str, str] = {}
    for line in script.splitlines():
        s = line.strip()
        if not s.startswith("#SBATCH"):
            continue
        body = s[len("#SBATCH"):].strip()
        if not body:
            continue
        # forms: --key=value | --key value | -k value | --flag
        if body.startswith("--"):
            if "=" in body:
                key, _, val = body.partition("=")
            else:
                parts = body.split(None, 1)
                key, val = parts[0], (parts[1] if len(parts) > 1 else "")
        elif body.startswith("-"):
            parts = body.split(None, 1)
            key, val = parts[0], (parts[1] if len(parts) > 1 else "")
        else:
            continue
        out[key.strip()] = val.strip().split("#")[0].strip()
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

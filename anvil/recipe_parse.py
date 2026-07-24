"""Extract Apptainer/Singularity recipes from raw model output, and parse
their structure.

Mirrors parse.py's role for `#SBATCH` directives, for a different artifact: a
`.def` file is a header (`Bootstrap:`, `From:`) followed by named `%section`
blocks, not a shell script with inline directives.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)```", re.DOTALL)

# The global (non-SCIF-app) sections a .def file can declare.
_SECTION_NAMES = (
    "help", "setup", "files", "environment", "post",
    "runscript", "startscript", "test", "labels",
)
_SECTION_RE = re.compile(r"^%(" + "|".join(_SECTION_NAMES) + r")\b.*$", re.MULTILINE)


def extract_recipe(raw: str) -> str:
    """Recover the `.def` recipe from raw model output."""
    fences = _FENCE.findall(raw)
    if fences:
        for block in fences:
            if "Bootstrap:" in block or re.search(r"^%\w+", block, re.MULTILINE):
                return block.strip("\n")
        return fences[0].strip("\n")
    return raw.strip()


def parse_header(recipe: str) -> dict[str, str]:
    """{"bootstrap": ..., "from": ...} from the lines before the first %section."""
    first_section = _SECTION_RE.search(recipe)
    header = recipe[: first_section.start()] if first_section else recipe
    out: dict[str, str] = {}
    for line in header.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" in s:
            key, _, val = s.partition(":")
            out[key.strip().lower()] = val.strip()
    return out


def list_sections(recipe: str) -> list[str]:
    """Names of the %sections present, in file order, lowercased."""
    return [m.group(1).lower() for m in _SECTION_RE.finditer(recipe)]


def section_body(recipe: str, name: str) -> str | None:
    """Body text of the first `%name` section, or None if absent."""
    matches = list(_SECTION_RE.finditer(recipe))
    for i, m in enumerate(matches):
        if m.group(1).lower() != name.lower():
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(recipe)
        return recipe[start:end].strip("\n")
    return None

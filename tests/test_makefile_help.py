"""`make help` against the Makefile it describes, and against the README.

`help` is curated, not generated: it groups targets by what someone is trying to do, and
the internal ones stay out on purpose. So the invariant is not that it lists everything.
It is that a target the README tells a reader to run is one the tool itself will admit to
having, and `make run` and `make docker-build-sched` were both printed by the README and
absent from `help`.

Read out of the Makefile rather than by running `make`, so the check holds where the suite
runs inside the container as well.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = re.compile(r"^([a-z0-9][a-z0-9-]*):")
MENTION = re.compile(r"make ([a-z0-9-]+)")


def _makefile() -> list[str]:
    return (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()


def _defined() -> set[str]:
    return {m.group(1) for line in _makefile() if (m := TARGET.match(line))}


def _listed_in_help() -> set[str]:
    """The recipe lines of the `help` target, which are echoes."""
    lines, inside = _makefile(), False
    out: set[str] = set()
    for line in lines:
        if TARGET.match(line):
            inside = line.startswith("help:")
            continue
        if inside:
            out |= set(MENTION.findall(line))
    return out


def test_help_only_offers_targets_that_exist():
    phantom = sorted(_listed_in_help() - _defined())
    assert not phantom, f"`make help` offers targets the Makefile does not define: {phantom}"


def test_every_target_the_readme_tells_you_to_run_is_in_help():
    """Intersected with the defined targets first: the README is prose, and a sentence like
    "would make later ones incomparable" otherwise reads as a target called `later`."""
    readme = set(MENTION.findall((ROOT / "README.md").read_text(encoding="utf-8")))
    missing = sorted((readme & _defined()) - _listed_in_help())
    assert not missing, (
        f"the README tells you to run these, `make help` does not list them: {missing}"
    )


def test_the_extraction_still_finds_a_help_target():
    """Both checks above pass on an empty set, which is what they would return if the help
    recipe were renamed or reformatted out of reach of this parser."""
    assert "help" in _defined()
    assert len(_listed_in_help()) >= 20

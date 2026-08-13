"""A script that documents `./scripts/x.py` has to be runnable that way.

`unchecked_values.py` was written with the shebang and the invocation its siblings use, and
without the executable bit, so the first person to copy the command out of its own docstring
got `Permission denied` on the other machine.

The rule is read off the files rather than imposed: `litsweep.py` documents itself as
`python scripts/litsweep.py` and is not executable, which is equally consistent. What must
not happen is a script promising one form and shipping the other.

The mode comes from `stat`, not from `git ls-files`. The first version shelled out to git
and passed here and in CI's plain test job, then failed the moment the same suite ran inside
the verification image, which has no git: a check that only works where a developer happens
to be standing is the failure this repository already refuses elsewhere. Git records the bit
and the checkout carries it, so the file on disk answers the same question.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _scripts() -> list[Path]:
    return sorted(p for p in SCRIPTS.iterdir()
                  if p.is_file() and p.suffix in (".py", ".sh"))


def _executable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)


def test_a_script_that_shows_a_bare_invocation_is_executable():
    wrong = []
    for path in _scripts():
        body = path.read_text(encoding="utf-8")
        shows_bare = re.search(rf"\./scripts/{re.escape(path.name)}", body) is not None
        if shows_bare and not _executable(path):
            wrong.append(f"{path.name} documents ./scripts/{path.name} and is not executable")
        if _executable(path) and not body.startswith("#!"):
            wrong.append(f"{path.name} is executable with no shebang")
    assert not wrong, wrong


def test_the_scripts_are_actually_being_read():
    """Both branches above are skipped by an empty listing, which is what a rename of the
    directory would produce."""
    found = _scripts()
    assert len(found) >= 15, [p.name for p in found]
    assert any(_executable(p) for p in found)

"""A script that documents `./scripts/x.py` has to be runnable that way.

`unchecked_values.py` was written with the shebang and the invocation its siblings use, and
without the executable bit, so the first person to copy the command out of its own docstring
got `Permission denied` on the other machine. The bit is recorded in the index, so this is a
property of the commit and not of anyone's checkout.

The rule is read off the file rather than imposed: `litsweep.py` documents itself as
`python scripts/litsweep.py` and is not executable, which is equally consistent. What must
not happen is a script promising one form and shipping the other.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _indexed_modes() -> dict[str, str]:
    out = subprocess.run(["git", "ls-files", "-s", "scripts"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    modes = {}
    for line in out.splitlines():
        mode, _, _, path = line.split(maxsplit=3)
        modes[path] = mode
    return modes


def test_a_script_that_shows_a_bare_invocation_is_executable():
    modes = _indexed_modes()
    wrong = []
    for path, mode in sorted(modes.items()):
        if not path.endswith((".py", ".sh")):
            continue
        body = (ROOT / path).read_text(encoding="utf-8")
        name = re.escape(Path(path).name)
        shows_bare = re.search(rf"\./scripts/{name}", body) is not None
        if shows_bare and mode != "100755":
            wrong.append(f"{path} documents ./{path} and is {mode}")
        if mode == "100755" and not body.startswith("#!"):
            wrong.append(f"{path} is {mode} with no shebang")
    assert not wrong, wrong


def test_the_scripts_are_actually_being_read():
    """Both branches above are skipped by an empty listing, which is what a rename of the
    directory would produce."""
    modes = _indexed_modes()
    assert len([p for p in modes if p.endswith((".py", ".sh"))]) >= 15

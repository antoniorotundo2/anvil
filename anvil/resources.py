"""Where the benchmark's data files are, whether or not there is a checkout.

Every default in the CLI names a path like `tasks/t1_slurm.jsonl`, relative to wherever
the command was run. Inside a clone that is exactly right and it is what every published
number was produced with. Installed from a wheel there is no `tasks/` beside the user, so
the same default named nothing and `anvil check --task t1_gpu_single` failed on a file it
could not have found.

The build maps the repository's `tasks/` directory into the package as `anvil/data`
(see pyproject.toml), so an installed copy carries the task files with it. This module is
the lookup that prefers the checkout and falls back to that copy:

  * a path that exists is used unchanged. A clone keeps behaving as it always has, and an
    explicit `--tasks some/other.jsonl` is never second-guessed;
  * otherwise, a file of the same name shipped inside the package;
  * otherwise the original path, so the caller raises the usual FileNotFoundError naming
    what the user actually asked for rather than an internal directory they never
    mentioned.

Deliberately not an import of `anvil.data` as a module: it has no `__init__.py`, it is a
directory of data, and reading it through `__file__` avoids depending on how a given
Python version treats namespace packages inside `importlib.resources`.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent
PACKAGED = _HERE / "data"
# The example site policy travels too, and by the same route. It is a `.json` under
# `policies/` rather than a `.jsonl` under `tasks/`, so the first glob never covered it and
# the command the README prints, `--policy policies/reference_cluster.json`, worked from a
# clone and failed from a wheel with a missing-file error naming a directory the reader had
# no reason to expect. Found by installing from the repository URL and running what the
# README says, which is the only way that class of defect shows up.
PACKAGED_POLICIES = _HERE / "policies"


def resolve(path: str | Path) -> Path:
    """The task, reference, corpus or policy file `path` refers to.

    The two directories are read here rather than frozen into a module-level tuple, so that
    a test pointing `PACKAGED` at a temporary directory still changes what this returns.
    """
    given = Path(path)
    if given.exists():
        return given
    for directory in (PACKAGED, PACKAGED_POLICIES):
        packaged = directory / given.name
        if packaged.is_file():
            return packaged
    return given

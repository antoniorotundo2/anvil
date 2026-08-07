"""The other half of `tasks_sha`.

A report already carries a digest of the task file, and `anvil verify` refuses generations
whose task set has moved, because a changed task invalidates a comparison. A changed
verifier invalidates it exactly as thoroughly and nothing recorded that: the walltime floor
added in `check_resource_fit` moved 123 of 2421 verdicts, and the reports from before it
and the reports from after it are indistinguishable on disk. Two numbers that came out of
different rules read as one series.

`verifier_sha()` is the digest of the modules a verdict actually depends on, stamped into
every report the CLI writes. It is deliberately taken over the raw bytes, so it moves when
a comment moves. That is the conservative direction: a changed digest means *these two
gradings were produced by different code, find out why*, which is a question worth being
asked once too often, and not *the numbers are wrong*. Normalising the source first would
buy quieter digests at the price of the guarantee, and would tie the value to whichever
Python version unparsed it.

`schema.py` is not in the set. It defines the shape a result is written in, not the rules
that decide it, and a field added there is visible in the report itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Ordered, so the digest does not depend on how a directory listing comes back.
VERDICT_MODULES = ("verifier.py", "parse.py")

_HERE = Path(__file__).resolve().parent


def verifier_sha() -> str:
    """Short digest of the rules that decide a verdict."""
    h = hashlib.sha256()
    for name in VERDICT_MODULES:
        h.update((_HERE / name).read_bytes())
    return h.hexdigest()[:12]

"""The failures that are answers rather than defects.

Asking for 4-bit weights on a machine without CUDA, or naming an executor that does not
exist, is not a bug in anvil: it is a request it cannot honour, and the caller wants one
line saying so. Everything else keeps its traceback, which is what a tool used for
measurement owes whoever has to work out why a number is wrong.

A dedicated class rather than the bare `RuntimeError` it replaced, because the CLI catches
this around its whole dispatch: catching `RuntimeError` there would swallow exactly the
tracebacks this exists to preserve. Found by running the retrieval ablation on the Mac,
where every one of nine cells printed a stack whose useful line was the last one.
"""

from __future__ import annotations


class UnsupportedRequest(RuntimeError):
    """This machine, or this build, cannot do what was asked."""

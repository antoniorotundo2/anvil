"""Which failures the CLI answers with a line, and which it lets crash.

Running the retrieval ablation on a Mac printed nine tracebacks, one per cell, each ending
in a perfectly clear sentence about 4-bit weights needing CUDA. The sentence was right and
nobody would read it in that position. The rule adopted here is narrow on purpose: a
request this machine cannot honour becomes one line and exit 2, and everything else keeps
its stack, because a tool used for measurement is debugged more often than it is used
wrong.
"""

from __future__ import annotations

import pytest

from anvil import cli
from anvil.errors import UnsupportedRequest


def test_a_mistyped_executor_variable_is_answered_not_raised(monkeypatch, capsys):
    """`--executor` goes through argparse `choices` and the environment variable goes
    through nothing, so the two spellings of one setting failed in two different ways."""
    monkeypatch.setenv("ANVIL_FUNCTIONAL_EXECUTOR", "slrum")
    assert cli.main(["doctor"]) == 2
    err = capsys.readouterr().err
    assert "ANVIL_FUNCTIONAL_EXECUTOR='slrum'" in err
    assert "bash, sbatch" in err
    assert "Traceback" not in err


def test_the_valid_names_still_pass_through(monkeypatch):
    """The check must not become a second, drifting definition of what an executor is: it
    reads the same list the flag does."""
    for name in ("bash", "sbatch"):
        monkeypatch.setenv("ANVIL_FUNCTIONAL_EXECUTOR", name)
        assert cli.main(["doctor", "--json"]) == 0


def test_an_unsupported_request_from_any_subcommand_is_one_line(monkeypatch, capsys):
    """Caught around the whole dispatch rather than at each call site, since the 4-bit
    refusal fires lazily on the first generation, deep inside a run loop."""
    def refuse(_args):
        raise UnsupportedRequest("4-bit quantization requested but unavailable on 'mps'")

    monkeypatch.setattr(cli, "cmd_doctor", refuse)
    assert cli.main(["doctor"]) == 2
    err = capsys.readouterr().err
    assert err.strip() == "anvil: 4-bit quantization requested but unavailable on 'mps'"


def test_anything_else_keeps_its_traceback(monkeypatch):
    """The reason `UnsupportedRequest` exists at all. Catching `RuntimeError` around the
    dispatch would have been one line shorter and would have swallowed every bug in the
    verifier, the parser and the metrics."""
    def crash(_args):
        raise RuntimeError("an actual bug")

    monkeypatch.setattr(cli, "cmd_doctor", crash)
    with pytest.raises(RuntimeError, match="an actual bug"):
        cli.main(["doctor"])

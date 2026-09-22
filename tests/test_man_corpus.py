"""The man-page corpus: how roff becomes retrieval documents.

The fixture is a page written for these tests in the dialect `sbatch.1` uses, not an
excerpt of it: SLURM's documentation is GPL-2.0 and stays out of this repository, which is
also why the corpus itself is generated rather than committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from man_corpus import chunk, main, parse, unconverted  # noqa: E402

from anvil.retrieval import Document  # noqa: E402

PAGE = r""".TH "fake" "1"
.SH "NAME"
fake \- a page written for tests
.SH "OPTIONS"
.TP
\fB\-t\fR, \fB\-\-time\fR=<\fItime\fR>
Limit the run time. The default is the \fBpartition\fR\(aqs limit.
.IP
A second paragraph about \-\-time.
.RS
.TP
\fBnested\fR
A sub-item that belongs to \-\-time.
.RE
.TP
\fB\-\-mem\fR=<\fIsize\fR>
Memory per node, continued \
on the next source line.
.SH "filename pattern"
.TP
\fB\\\\\fR
Do not process any replacement symbols.
.SH "EXAMPLES"
.TP
Run a script:
.nf
$ cat job.sh
#!/bin/sh
.fi
.SH "COPYING"
Licence text that says nothing about job scripts.
"""


def _entries():
    return {e["heading"] or e["section"]: e for e in parse(PAGE)}


def test_one_entry_per_top_level_item_and_nested_items_stay_with_theirs():
    entries = _entries()
    assert "-t, --time=<time>" in entries and "--mem=<size>" in entries
    assert "nested" not in entries
    body = " ".join(" ".join(p["lines"]) for p in entries["-t, --time=<time>"]["paragraphs"])
    assert "belongs to --time" in body


def test_escapes_become_text_and_a_literal_backslash_survives():
    """`\\\\` is a pattern SLURM documents, a backslash meant literally; everything else is
    markup the model should never read."""
    entries = parse(PAGE)
    assert unconverted(entries) == []
    docs = {d["id"]: d["text"] for d in chunk(entries)}
    assert docs["sbatch_time"].startswith("-t, --time=<time>\nLimit the run time.")
    assert "partition's limit" in docs["sbatch_time"]
    assert any(text.startswith("\\\\\n") for text in docs.values())


def test_a_continued_line_is_joined():
    assert "continued on the next source line" in {
        d["id"]: d["text"] for d in chunk(parse(PAGE))}["sbatch_mem"]


def test_an_unknown_escape_is_reported_rather_than_passed_to_the_prompt():
    assert unconverted(parse(".SH \"OPTIONS\"\n.TP\n\\-\\-x\nUses \\(zz here.\n"))


def test_preformatted_lines_keep_their_breaks_and_the_licence_is_left_out():
    docs = chunk(parse(PAGE))
    example = next(d for d in docs if d["id"].startswith("sbatch_examples"))
    assert "$ cat job.sh\n#!/bin/sh" in example["text"]
    assert not any("Licence text" in d["text"] for d in docs)


def test_a_long_entry_is_split_under_its_heading_without_overlap():
    """Every piece stays within the size and says which option it is about, and the pieces
    put back together are the entry: nothing dropped, nothing repeated."""
    pytest.importorskip("langchain_text_splitters", reason="corpus splitter")
    sentences = [f"Sentence number {i} about memory." for i in range(40)]
    page = '.SH "OPTIONS"\n.TP\n\\fB\\-\\-mem\\fR=<size>\n' + " ".join(sentences) + "\n"
    docs = chunk(parse(page), size=200)
    assert len(docs) > 1
    assert [d["id"] for d in docs] == [f"sbatch_mem_{i}" for i in range(1, len(docs) + 1)]
    assert all(len(d["text"]) <= 200 and d["text"].startswith("--mem=<size>\n") for d in docs)
    rebuilt = " ".join(d["text"].split("\n", 1)[1] for d in docs)
    assert rebuilt.split() == " ".join(sentences).split()


def test_the_output_loads_as_a_retrieval_corpus(tmp_path):
    """The first line records where the text came from and is skipped by the loader, so the
    file is usable as `--retrieval-corpus` as it stands."""
    pytest.importorskip("langchain_text_splitters", reason="corpus splitter")
    roff, out = tmp_path / "fake.1", tmp_path / "corpus.jsonl"
    roff.write_text(PAGE, encoding="utf-8")
    assert main(["--from-roff", str(roff), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("// sbatch.1 read from")
    corpus = Document.load_jsonl(out)
    assert {d.id for d in corpus} >= {"sbatch_time", "sbatch_mem"}
    assert len({d.id for d in corpus}) == len(corpus)

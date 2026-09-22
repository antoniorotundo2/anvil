#!/usr/bin/env python3
"""A retrieval corpus cut from the `sbatch` man page, for the arms that rank.

The curated corpus has eight documents, and on it `vector` and `dense` attach nearly the
same ones: with k=2 out of eight there is little for a ranking to decide. The question the
dense arm reopened, whether retrieving by meaning rather than by overlap changes anything,
needs a corpus where ranking has room to matter. This builds one from real documentation.

    .venv/bin/python scripts/man_corpus.py        # -> results/corpus/sbatch_man.jsonl
    CORPUS=results/corpus/sbatch_man.jsonl STRATEGIES="vector dense" \
      ./scripts/retrieval_ablation.sh

The splitter comes with the `dense` extra, so this runs under the project venv rather than
whatever `python3` the shebang finds.

Where the text comes from. The man page of the `sbatch` the verification image runs, taken
from the package of exactly the installed version: the image's dpkg excludes drop man pages
at install time, so the package is downloaded again and unpacked with `dpkg-deb -x`, which
the excludes do not apply to. If the archive no longer offers that version the build fails
rather than documenting a different `sbatch` from the one that grades.

Why it is generated and not committed. SLURM's documentation is GPL-2.0 and this repository
is MIT, so the text stays out of it; the output goes under `results/`, which is not tracked,
and its first line records the version it was cut from.

How it is cut. By structure first: one chunk per option, per environment variable, per
filename pattern, and per section of prose, since an option with its description is the
unit a reader looks up and a character count knows nothing of. Then by size, with
LangChain's `RecursiveCharacterTextSplitter`, because entries run from a few words to over
6,000 characters and the curated documents are 309 to 475: the intervention series showed
that at 1.5B the amount of attached text is a variable in itself, so a corpus whose chunks
were ten times longer would measure length along with ranking. Every piece of a split
entry repeats the entry's heading, so that a chunk about `--mem` says it is about `--mem`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# The curated corpus tops out at 475 characters; this is the same scale, so that length is
# not the difference between the two corpora.
CHUNK_CHARS = 500

# Sections that document behaviour. `COPYING` is the licence and `SEE ALSO` a list of
# other pages: neither says anything about writing a job script.
SKIPPED_SECTIONS = {"COPYING", "SEE ALSO"}

# A backslash the page means literally, from `\\` or `\e`, is held as this placeholder
# while the other escapes are converted and checked, so that the check for escapes left
# unconverted cannot mistake it for one; it becomes a backslash again in `chunk`.
_LITERAL = "\x00"
_ESCAPES = [
    (re.compile(r"\\\\|\\e"), _LITERAL),
    (re.compile(r"\\f(\(..|\[[^]]*\]|.)"), ""),     # font changes
    (re.compile(r"\\\((em|en)"), "-"),
    (re.compile(r"\\\(aq"), "'"),
    (re.compile(r"\\\(dq"), '"'),
    (re.compile(r"\\\((lq|rq)"), '"'),
    (re.compile(r"\\\(bu"), "*"),
    (re.compile(r"\\-"), "-"),
    (re.compile(r"\\[&|^]"), ""),
    (re.compile(r"\\ "), " "),
]
# A backslash ending a line is roff's continuation, joined before anything else is read.
_CONTINUATION = re.compile(r"(?<!\\)\\\n")
_LEFTOVER = re.compile(r"\\[a-zA-Z(\[]|\\$", re.MULTILINE)
_BREAKS = {"PP", "LP", "P", "IP", "br", "sp"}


def clean(text: str) -> str:
    for pattern, replacement in _ESCAPES:
        text = pattern.sub(replacement, text)
    return text


def parse(roff: str) -> list[dict]:
    """Entries of the page, each `{"section", "heading", "paragraphs"}`.

    A top-level `.TP` opens an entry and its next line is the heading; `.TP` inside an
    `.RS` block is a sub-item and stays with its entry. Paragraph macros separate
    paragraphs, and lines between `.nf` and `.fi` keep their breaks, which matters for the
    examples and nothing else.
    """
    entries: list[dict] = []
    section, depth, preformatted, want_heading = "", 0, False, False
    current: dict | None = None

    def paragraph() -> list[str]:
        """The paragraph text goes into, opening one when the mode changed under it."""
        paragraphs = current["paragraphs"]
        if not paragraphs or paragraphs[-1]["pre"] != preformatted:
            paragraphs.append({"pre": preformatted, "lines": []})
        return paragraphs[-1]["lines"]

    def close() -> None:
        if current is not None and any(p["lines"] for p in current["paragraphs"]):
            entries.append(current)

    for raw in _CONTINUATION.sub("", roff).splitlines():
        if raw.startswith('.\\"') or raw.startswith("'\\\""):
            continue
        macro = re.match(r"^\.(\w+)\s*(.*)$", raw)
        if macro:
            name, arg = macro.group(1), clean(macro.group(2)).strip().strip('"')
            if name == "SH":
                close()
                section, depth = arg, 0
                current = {"section": section, "heading": None, "paragraphs": []}
            elif name == "TP" and depth == 0:
                close()
                current = {"section": section, "heading": None, "paragraphs": []}
                want_heading = True
            elif name == "RS":
                depth += 1
            elif name == "RE":
                depth = max(0, depth - 1)
            elif name == "nf":
                preformatted = True
            elif name == "fi":
                preformatted = False
            elif current is not None and name in _BREAKS | {"TP"}:
                current["paragraphs"].append({"pre": preformatted, "lines": []})
                if name == "IP" and arg:
                    current["paragraphs"][-1]["lines"].append(arg)
            elif current is not None and name in ("B", "I") and arg:
                paragraph().append(arg)
            continue
        if current is None:
            continue
        text = clean(raw)
        if want_heading:
            current["heading"], want_heading = text.strip(), False
            continue
        target = paragraph()
        target.append(text if preformatted else text.strip())
    close()
    return [e for e in entries if e["section"] not in SKIPPED_SECTIONS]


def _body(entry: dict) -> str:
    """Filled paragraphs joined into running text, preformatted ones kept line by line."""
    out = []
    for p in entry["paragraphs"]:
        lines = [line for line in p["lines"] if line.strip()]
        if lines:
            out.append("\n".join(lines) if p["pre"] else re.sub(r"\s+", " ", " ".join(lines)))
    return "\n\n".join(out).strip()


def unconverted(entries: list[dict]) -> list[str]:
    """Headings of the entries still carrying a roff escape after `clean`."""
    return [
        e["heading"] or e["section"]
        for e in entries
        if _LEFTOVER.search(f"{e['heading'] or ''}\n{_body(e)}")
    ]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _entry_id(entry: dict, index: int) -> str:
    heading = entry["heading"] or ""
    if option := re.search(r"--([a-z0-9][a-z0-9-]*)", heading):
        return f"sbatch_{_slug(option.group(1))}"
    if variable := re.match(r"([A-Z][A-Z0-9_]+)", heading):
        return f"sbatch_env_{variable.group(1)}"
    return f"sbatch_{_slug(entry['section'])}_{index}"


def _splitter(size: int):
    """LangChain's recursive splitter, imported only when an entry is too long to keep
    whole: the parsing and every entry that fits need nothing outside the standard library.
    Paragraph, then sentence, then word boundaries, with the separator kept at the end of
    its piece so that a sentence keeps its full stop, and the default overlap of 200
    characters turned off."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            'the splitter comes with the dense extra: pip install -e ".[dense]", '
            "then run this with .venv/bin/python"
        ) from exc
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=0, separators=["\n\n", ". ", " "], keep_separator="end"
    )


def chunk(entries: list[dict], size: int = CHUNK_CHARS) -> list[dict]:
    """Documents in the shape `tasks/retrieval_corpus.jsonl` uses. An entry that fits is one
    document; one that does not is split on paragraph, then sentence, then word boundaries,
    each piece headed by the entry's heading, with no overlap: overlapping pieces of one
    entry would let k=2 attach the same sentence twice."""
    documents: list[dict] = []
    seen: dict[str, int] = {}
    for index, entry in enumerate(entries):
        heading = entry["heading"]
        body = _body(entry)
        if not body:
            continue
        prefix = f"{heading}\n" if heading else ""
        room = max(size - len(prefix), 100)
        if len(prefix) + len(body) <= size:
            pieces = [body]
        else:
            pieces = [p.strip() for p in _splitter(room).split_text(body) if p.strip()]
        base = _entry_id(entry, index)
        for number, piece in enumerate(pieces, start=1):
            doc_id = base if len(pieces) == 1 else f"{base}_{number}"
            if doc_id in seen:
                seen[doc_id] += 1
                doc_id = f"{doc_id}_{seen[doc_id]}"
            else:
                seen[doc_id] = 1
            text = (prefix + piece).replace(_LITERAL, "\\")
            documents.append({"id": doc_id, "text": text, "tags": []})
    return documents


def fetch(image: str, runtime: str) -> tuple[str, str]:
    """(version, roff source) of `sbatch.1` for the `slurm-client` installed in `image`."""
    script = r"""
set -e
v=$(dpkg-query -W -f='${Version}' slurm-client)
apt-get update -qq >/dev/null
cd /tmp && apt-get download -qq "slurm-client=$v" >/dev/null
dpkg-deb -x /tmp/slurm-client_*.deb /tmp/x
echo "$v"
zcat /tmp/x/usr/share/man/man1/sbatch.1.gz
"""
    out = subprocess.run([runtime, "run", "--rm", "--entrypoint", "bash", image, "-c", script],
                         capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"could not extract sbatch.1 from {image}:\n{out.stderr.strip()}")
    version, _, roff = out.stdout.partition("\n")
    return version.strip(), roff


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default="anvil:sched")
    parser.add_argument("--from-roff", metavar="PATH",
                        help="cut a roff file already on disk instead of the image's page")
    parser.add_argument("--out", default="results/corpus/sbatch_man.jsonl")
    args = parser.parse_args(argv)

    if args.from_roff:
        roff = Path(args.from_roff).read_text(encoding="utf-8", errors="replace")
        source = f"sbatch.1 read from {args.from_roff}"
    else:
        version, roff = fetch(args.image, os.environ.get("RUNTIME", "docker"))
        source = f"sbatch.1 from slurm-client {version} in {args.image}"

    entries = parse(roff)
    leftovers = unconverted(entries)
    if leftovers:
        # A roff escape this cleaner does not know would reach the prompt as noise, and the
        # model would be reading markup the curated corpus never showed it.
        print(f"unconverted roff escapes in {leftovers[:5]}: extend _ESCAPES", file=sys.stderr)
        return 1
    documents = chunk(entries)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(f"// {source}; GPL-2.0 text, generated and not committed\n")
        for document in documents:
            fh.write(json.dumps(document, ensure_ascii=False) + "\n")
    sizes = sorted(len(d["text"]) for d in documents)
    print(f"{out}: {len(documents)} documents, {sizes[0]} to {sizes[-1]} characters "
          f"(median {sizes[len(sizes) // 2]}), from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

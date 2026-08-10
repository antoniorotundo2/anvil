"""Every internal link in the documentation, followed.

The documents cross-reference each other heavily and sections get renamed as measurements
replace each other, which is how `DESIGN.md` came to point at
`#real-submission-moves-functional-and-barely-touches-the-verdict` after the section by
that name had become `### Real submission was almost redundant until a model wrote srun`.
The link still rendered, and still landed at the top of a 1000-line file instead of at the
numbers the sentence promised.

External URLs are not fetched: a test that reaches the network fails for reasons that have
nothing to do with this repository.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _slugs(text: str) -> set[str]:
    """GitHub's rule, as far as these documents use it: lowercase, drop anything that is
    not alphanumeric, space or hyphen, then spaces to hyphens. Backticks and colons appear
    in headings here, and both simply vanish."""
    out = set()
    for line in text.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip().lower()
            out.add(re.sub(r"[^a-z0-9 -]", "", title).replace(" ", "-"))
    return out


def _documents() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def test_every_internal_link_reaches_a_file():
    dead = []
    for doc in _documents():
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path = target.partition("#")[0]
            if path and not (doc.parent / path).resolve().exists():
                dead.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not dead, dead


def test_every_anchor_reaches_a_heading():
    missing = []
    for doc in _documents():
        body = doc.read_text(encoding="utf-8")
        for target in LINK.findall(body):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, fragment = target.partition("#")
            if not fragment:
                continue
            dest = (doc.parent / path).resolve() if path else doc
            if dest.suffix != ".md" or not dest.exists():
                continue
            if fragment not in _slugs(dest.read_text(encoding="utf-8")):
                missing.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not missing, missing


def test_the_check_covers_the_documents_that_exist():
    """A regex that stopped matching would make both tests above pass by finding nothing,
    which is the failure mode of every check written as an empty list."""
    docs = _documents()
    assert len(docs) >= 8, [d.name for d in docs]
    found = sum(len(LINK.findall(d.read_text(encoding="utf-8"))) for d in docs)
    assert found >= 60, found

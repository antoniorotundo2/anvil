"""Retrieval ablation: does giving a model reference material about SLURM
semantics change how correctly it writes a script?

Three conditions to compare, all operating on the same small corpus of
reference documents:

  * zero-shot   - the current baseline: no extra context, the task prompt
                  alone (this is what T1/T2/T3 already do without this module).
  * vector      - TF-IDF cosine similarity between the task prompt and each
                  document, the classic "vector retrieval" approach. Pure
                  Python (stdlib `Counter`/`math`): the core has no ML
                  dependencies, and the corpus is small enough that a real
                  embedding model would be overkill for what this ablation
                  measures.
  * vectorless  - exact tag overlap between the task and a document, no
                  similarity scoring at all. Structure-based, not
                  similarity-based: it retrieves a document because it is
                  *about* the same topic (same declared tag), not because its
                  text happens to look similar to the prompt.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .schema import Task

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Document:
    id: str
    text: str
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def load_jsonl(path: str | Path) -> list[Document]:
        docs: list[Document] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                docs.append(Document(**json.loads(line)))
        return docs


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def retrieve_zero_shot(task: Task, corpus: list[Document], k: int = 2) -> list[Document]:
    """The baseline: no retrieval at all."""
    return []


def retrieve_vectorless(task: Task, corpus: list[Document], k: int = 2) -> list[Document]:
    """Structure-based: a document matches because it is tagged with the same
    topic as the task, not because its text resembles the prompt. A document
    tagged "general" applies to every task (universal SLURM facts, such as
    silent defaults or directive placement) but is only surfaced after the
    topic-specific matches: otherwise a handful of broadly-tagged documents
    would crowd out the one document that is actually about the task's
    topic. Deterministic, no scoring: ties are broken by corpus order."""
    specific = [d for d in corpus if set(d.tags) & set(task.tags)]
    general = [d for d in corpus if "general" in d.tags and d not in specific]
    return (specific + general)[:k]


def retrieve_vector(task: Task, corpus: list[Document], k: int = 2) -> list[Document]:
    """TF-IDF cosine similarity between the task prompt and each document.
    Documents with zero overlap (similarity 0) are never returned: a
    similarity-based retriever that always returns k documents regardless of
    relevance is not actually measuring similarity."""
    if not corpus:
        return []

    doc_tokens = [_tokenize(d.text) for d in corpus]
    query_tokens = _tokenize(task.prompt)

    n_docs = len(corpus) + 1  # the query counts as a "document" for IDF
    df: Counter[str] = Counter()
    for tokens in [*doc_tokens, query_tokens]:
        df.update(set(tokens))
    idf = {term: math.log(n_docs / count) + 1.0 for term, count in df.items()}

    def _tfidf_vector(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        return {term: (count / total) * idf.get(term, 0.0) for term, count in tf.items()}

    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return num / (norm_a * norm_b)

    query_vec = _tfidf_vector(query_tokens)
    scored = [
        (_cosine(query_vec, _tfidf_vector(tokens)), d)
        for d, tokens in zip(corpus, doc_tokens, strict=True)
    ]
    scored = [(score, d) for score, d in scored if score > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [d for _, d in scored[:k]]


STRATEGIES = {
    "zero-shot": retrieve_zero_shot,
    "vector": retrieve_vector,
    "vectorless": retrieve_vectorless,
}


POSITIONS = ("append", "prepend")


def build_prompt_with_context(
    task_prompt: str, docs: list[Document], position: str = "append"
) -> str:
    """Put the retrieved documents after the task, or before it.

    `append` is the default and the position every published arm was measured at.
    `prepend` exists because where the context sits is a variable in its own right:
    the same documents read as background when they come first and as an afterthought
    when they come last, and the arms cannot tell those apart while only one is
    available.

    Appending used to be a requirement rather than a default. Oracle and broken model
    prompt-matching checked `prompt.startswith(task.prompt)`, which prepending breaks,
    so `models.OracleModel` now matches the task prompt anywhere in what it receives.
    """
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
    if not docs:
        return task_prompt
    context = "\n\n".join(f"[{d.id}]\n{d.text}" for d in docs)
    if position == "prepend":
        return f"Reference material:\n{context}\n\n{task_prompt}"
    return f"{task_prompt}\n\nReference material:\n{context}"

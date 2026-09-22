"""Retrieval ablation: does giving a model reference material about SLURM
semantics change how correctly it writes a script?

Four conditions to compare, all operating on the same small corpus of
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
  * dense       - a LangChain retrieval pipeline: the corpus goes into an
                  in-memory LangChain vector store over sentence embeddings,
                  and the task prompt is the query. Only retrieval runs in
                  LangChain; the prompt assembly and the generation are the
                  ones every arm shares, since they are what the comparison
                  holds fixed. Added after the three above were published,
                  to reopen the one question `vector` leaves open on
                  purpose: the reason given for skipping an embedding model
                  was that lexical overlap is enough for a corpus this
                  small, which says nothing about whether a retriever judged
                  by meaning would change the finding. The only arm with
                  dependencies, so it lives behind the `dense` extra and
                  imports them lazily.
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .errors import UnsupportedRequest
from .resources import resolve
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
        with open(resolve(path), encoding="utf-8") as fh:
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


# Pinned by revision as well as by name: a model id on the Hub is a moving reference, and
# an arm whose retriever can change underneath it between two runs is not one arm.
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_DENSE_EXTRA = '--retrieval dense needs the dense extra: pip install -e ".[dense]"'


@functools.lru_cache(maxsize=1)
def dense_embedder():
    """The LangChain embeddings object `retrieve_dense` uses, loaded once per process.

    The import is here and not at the top of the module because the other three arms, and
    everything else in the package, run on the standard library alone; importing anvil must
    not start requiring LangChain because one arm of one ablation uses it. A missing extra
    is a request this install cannot honour rather than a defect, hence `UnsupportedRequest`,
    which the CLI turns into one line instead of a traceback.

    Pinned to the CPU: the model being evaluated already holds the GPU, and a retriever
    whose scores depend on which device it landed on would make the arm less repeatable
    than the lexical one it is compared with.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
    except ImportError as exc:
        raise UnsupportedRequest(_DENSE_EXTRA) from exc
    return HuggingFaceEmbeddings(
        model_name=DENSE_MODEL,
        model_kwargs={"device": "cpu", "revision": DENSE_REVISION},
    )


def retrieve_dense(task: Task, corpus: list[Document], k: int = 2) -> list[Document]:
    """Cosine similarity between sentence embeddings, through LangChain: the documents are
    loaded into an `InMemoryVectorStore` over `dense_embedder()` and searched with the task
    prompt. The contract is `retrieve_vector`'s, so that the retriever is the only thing
    that differs between the two arms: descending similarity, ties left in corpus order,
    and a document with similarity 0 or below never returned.

    The store is asked for every document and the order is then fixed here, because its own
    ranking is `argsort()[::-1]`, which is not stable: equal scores come back in whatever
    order the sort leaves them, and an arm whose retrieval can reorder between two runs of
    the same input is not deterministic. The zero rule is applied here too, since the store
    has no notion of a similarity too low to be worth attaching.

    That rule rarely fires, which is a property of dense embeddings rather than a looser
    rule: unrelated English sentences still share a direction, and on the published corpus
    the lowest cosine between any T1 prompt and any document is about 0.1. `vector` returns
    fewer than k documents when nothing overlaps; this arm almost never does.

    A new store per call rather than one per corpus: eight short documents embed in
    milliseconds on the CPU, and a cache keyed on the corpus would be state for tests to
    reset and for a changed corpus to go stale in.
    """
    if not corpus:
        return []
    try:
        from langchain_core.documents import Document as LangChainDocument  # noqa: PLC0415
        from langchain_core.vectorstores import InMemoryVectorStore  # noqa: PLC0415
    except ImportError as exc:
        raise UnsupportedRequest(_DENSE_EXTRA) from exc

    embedder = dense_embedder()
    query = embedder.embed_query(task.prompt)
    # A zero vector on either side makes every cosine 0/0. The store raises on that instead
    # of scoring it, and 0 is what the rule above excludes, so the answer is known without
    # searching. Unreachable with a sentence-embedding model, reached by the first fake
    # embedding the tests used, and cheaper to state than to leave to the store's message.
    if not any(query) or not any(any(v) for v in embedder.embed_documents(
            [d.text for d in corpus])):
        return []

    store = InMemoryVectorStore(embedding=embedder)
    store.add_documents(
        [LangChainDocument(page_content=d.text, metadata={"position": i})
         for i, d in enumerate(corpus)]
    )
    hits = store.similarity_search_with_score_by_vector(query, k=len(corpus))
    ranked = sorted(
        ((score, hit.metadata["position"]) for hit, score in hits if score > 0),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return [corpus[position] for _, position in ranked[:k]]


STRATEGIES = {
    "zero-shot": retrieve_zero_shot,
    "vector": retrieve_vector,
    "vectorless": retrieve_vectorless,
    "dense": retrieve_dense,
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

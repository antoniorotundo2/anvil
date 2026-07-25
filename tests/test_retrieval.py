"""Retrieval ablation tests: pure Python, no GPU, no model needed.

Guiding principle: vectorless (structure) and vector (surface similarity) are
deliberately different retrieval strategies, and the tests check that they
actually differ where it matters, not just that they run without crashing.
"""

from __future__ import annotations

from pathlib import Path

from anvil.retrieval import (
    Document,
    build_prompt_with_context,
    retrieve_vector,
    retrieve_vectorless,
    retrieve_zero_shot,
)
from anvil.schema import Task

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tasks" / "retrieval_corpus.jsonl"


def _task(**kw) -> Task:
    kw.setdefault("id", "x")
    kw.setdefault("prompt", "p")
    return Task(**kw)


# ---------------------------------------------------------------- corpus
def test_corpus_loads_and_is_non_empty():
    corpus = Document.load_jsonl(CORPUS)
    assert corpus
    assert all(d.id and d.text for d in corpus)


def test_every_task_topic_tag_has_a_specific_document():
    """If a T1 task tag has no dedicated document, the vectorless arm can
    never retrieve anything topic-specific for it."""
    tasks = Task.load_jsonl(ROOT / "tasks" / "t1_slurm.jsonl")
    corpus = Document.load_jsonl(CORPUS)
    all_doc_tags = {t for d in corpus for t in d.tags if t != "general"}
    difficulty_tags = ("easy", "medium", "hard")
    task_topic_tags = {t for task in tasks for t in task.tags if t not in difficulty_tags}
    missing = task_topic_tags - all_doc_tags
    assert not missing, f"no document tagged for: {missing}"


# ---------------------------------------------------------------- zero-shot
def test_zero_shot_retrieves_nothing():
    corpus = Document.load_jsonl(CORPUS)
    assert retrieve_zero_shot(_task(tags=["gpu"]), corpus) == []


# ---------------------------------------------------------------- vectorless
def test_vectorless_prioritises_topic_specific_over_general():
    corpus = [
        Document(id="general_doc", text="applies to everything", tags=["general"]),
        Document(id="gpu_doc", text="about gpus specifically", tags=["gpu"]),
    ]
    docs = retrieve_vectorless(_task(tags=["gpu", "medium"]), corpus, k=1)
    assert [d.id for d in docs] == ["gpu_doc"]


def test_vectorless_falls_back_to_general_when_no_topic_match():
    corpus = [
        Document(id="general_doc", text="applies to everything", tags=["general"]),
        Document(id="gpu_doc", text="about gpus specifically", tags=["gpu"]),
    ]
    docs = retrieve_vectorless(_task(tags=["array"]), corpus, k=2)
    assert [d.id for d in docs] == ["general_doc"]


def test_vectorless_on_real_corpus_matches_every_t1_task_topic():
    tasks = Task.load_jsonl(ROOT / "tasks" / "t1_slurm.jsonl")
    corpus = Document.load_jsonl(CORPUS)
    for task in tasks:
        docs = retrieve_vectorless(task, corpus, k=2)
        assert docs, f"no document retrieved for {task.id}"
        topic_tags = {t for t in task.tags if t not in ("easy", "medium", "hard")}
        assert set(docs[0].tags) & topic_tags, (
            f"{task.id}: first retrieved doc {docs[0].id!r} is not topic-specific"
        )


# ---------------------------------------------------------------- vector
def test_vector_returns_nothing_below_zero_similarity():
    corpus = [Document(id="d", text="completely unrelated vocabulary here", tags=[])]
    docs = retrieve_vector(_task(prompt="xyzxyz qqqqq zzzzz"), corpus, k=2)
    assert docs == []


def test_vector_ranks_the_more_similar_document_first():
    corpus = [
        Document(id="gpu_doc", text="request a gpu with the gpus directive", tags=[]),
        Document(id="array_doc", text="job array index task", tags=[]),
    ]
    docs = retrieve_vector(_task(prompt="please request a gpu for this job"), corpus, k=1)
    assert [d.id for d in docs] == ["gpu_doc"]


def test_vector_respects_k():
    corpus = Document.load_jsonl(CORPUS)
    task = _task(prompt="request a gpu and set the walltime and memory")
    assert len(retrieve_vector(task, corpus, k=1)) <= 1
    assert len(retrieve_vector(task, corpus, k=100)) <= len(corpus)


# ---------------------------------------------------------------- prompt augmentation
def test_build_prompt_with_context_no_docs_returns_prompt_unchanged():
    assert build_prompt_with_context("write a script", []) == "write a script"


def test_build_prompt_with_context_starts_with_original_prompt():
    """Oracle/broken prompt-matching relies on this: the augmented prompt
    must always start with the exact original task prompt."""
    docs = [Document(id="d1", text="some reference text", tags=[])]
    augmented = build_prompt_with_context("write a script", docs)
    assert augmented.startswith("write a script")
    assert "some reference text" in augmented
    assert "d1" in augmented

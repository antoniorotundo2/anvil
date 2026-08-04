"""Retrieval ablation tests: pure Python, no GPU, no model needed.

Guiding principle: vectorless (structure) and vector (surface similarity) are
deliberately different retrieval strategies, and the tests check that they
actually differ where it matters, not just that they run without crashing.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------- intervention corpora
# The seven-condition series in DESIGN.md rests on one precondition: the document a
# variant promotes must actually reach every task. It is a property of corpus order and
# of nothing else, so it breaks silently if a document is ever added to the corpus above
# the general block, and the table would then describe an experiment nobody ran.
def _variants() -> dict[str, list[dict]]:
    from scripts.corpus_variants import build_variants  # noqa: PLC0415

    return build_variants()


def _attached(rows: list[dict]) -> set[str]:
    corpus = [Document(**row) for row in rows]
    tasks = Task.load_jsonl(ROOT / "tasks" / "t1_slurm.jsonl")
    return {docs[-1].id for t in tasks if (docs := retrieve_vectorless(t, corpus))}


def test_each_variant_puts_its_document_in_front_of_every_task():
    expected = {
        "timemem_first": "doc_time_mem",
        "control_offtopic": "doc_control_offtopic",
        "control_offtopic2": "doc_control_offtopic2",
    }
    variants = _variants()
    assert set(variants) == set(expected)
    for name, doc_id in expected.items():
        assert _attached(variants[name]) == {doc_id}, name


def test_default_corpus_attaches_the_incumbent_instead():
    """The baseline the variants are measured against: without the intervention the
    fallback slot goes to the document that comes first in corpus order."""
    rows = [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert _attached(rows) == {"doc_directive_placement"}


def test_controls_match_the_length_of_the_document_they_displace():
    """Relevance is the variable under test, so the text either side of it is the same
    size. The id also reaches the prompt and is not length-matched, which is recorded in
    the script rather than fixed."""
    variants = _variants()
    subject = next(
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["id"] == "doc_time_mem"
    )
    for name in ("control_offtopic", "control_offtopic2"):
        control = variants[name][0]
        assert len(control["text"]) == len(subject["text"]), name


def test_variants_never_drop_a_document():
    """A control displaces `doc_time_mem` from the fallback slot without removing it: a
    corpus that is also smaller would confound the comparison."""
    ids = {
        json.loads(line)["id"]
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for name, rows in _variants().items():
        assert ids <= {row["id"] for row in rows}, name

"""Retrieval ablation tests: pure Python, no GPU, no model needed.

Guiding principle: vectorless (structure) and vector (surface similarity) are
deliberately different retrieval strategies, and the tests check that they
actually differ where it matters, not just that they run without crashing.
The same goes for dense (meaning) against vector: its tests include the case
only an embedding can retrieve.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from anvil import retrieval
from anvil.retrieval import (
    RANKERS,
    STRATEGIES,
    Document,
    build_prompt_with_context,
    provenance,
    rank_vector,
    retrieve_dense,
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


# ---------------------------------------------------------------- dense
class _SynonymEmbeddings:
    """Stands in for the sentence-embedding model, which the suite never downloads.

    Each axis is a concept with more than one spelling, so two texts can be close with no
    word in common. That is the property separating `dense` from `vector`, reproduced here
    without the model, and deterministic by construction. What it cannot stand in for is
    which documents the real model prefers; that is measured, not tested.
    """

    AXES = (
        ("gpu", "gpus", "accelerator", "cuda"),
        ("array", "index"),
        ("memory", "mem", "ram"),
    )

    def __init__(self) -> None:
        self.calls = 0

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z]+", text.lower())
        return [float(sum(t in axis for t in tokens)) for axis in self.AXES]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._embed(t) for t in texts]


def _needs_the_pipeline() -> None:
    """The pipeline is LangChain's vector store, whose cosine needs numpy. Both come with the
    `dev` extra, so CI runs these; the verification image installs neither and skips them
    with this reason rather than failing on an import."""
    pytest.importorskip("langchain_core.vectorstores", reason="dense pipeline: langchain-core")
    pytest.importorskip("numpy", reason="dense pipeline: numpy")


@pytest.fixture
def fake_embedder(monkeypatch):
    _needs_the_pipeline()
    fake = _SynonymEmbeddings()
    monkeypatch.setattr(retrieval, "dense_embedder", lambda: fake)
    return fake


def test_dense_ranks_the_more_similar_document_first(fake_embedder):
    corpus = [
        Document(id="array_doc", text="job array index", tags=[]),
        Document(id="gpu_doc", text="request a gpu with the gpus directive", tags=[]),
    ]
    docs = retrieve_dense(_task(prompt="please request a gpu for this job"), corpus, k=1)
    assert [d.id for d in docs] == ["gpu_doc"]


def test_dense_retrieves_a_paraphrase_that_vector_cannot(fake_embedder):
    """The reason the arm exists. No word of the prompt occurs in the document, so TF-IDF
    has nothing to score and returns nothing; an embedding places both near the same
    concept."""
    corpus = [Document(id="gpu_doc", text="request gpus with the gres directive", tags=[])]
    task = _task(prompt="run this on an accelerator")
    assert retrieve_vector(task, corpus) == []
    assert [d.id for d in retrieve_dense(task, corpus)] == ["gpu_doc"]


def test_dense_returns_nothing_at_zero_similarity(fake_embedder):
    """The rule `vector` states, kept identical: orthogonal and empty embeddings score 0,
    and a document scoring 0 is never attached just to fill k."""
    corpus = [
        Document(id="orthogonal", text="job array index", tags=[]),
        Document(id="empty", text="completely unrelated vocabulary", tags=[]),
    ]
    assert retrieve_dense(_task(prompt="request a gpu"), corpus, k=2) == []


def test_dense_respects_k(fake_embedder):
    corpus = Document.load_jsonl(CORPUS)
    task = _task(prompt="request a gpu, an array index and some memory")
    assert len(retrieve_dense(task, corpus, k=1)) <= 1
    assert len(retrieve_dense(task, corpus, k=100)) <= len(corpus)


def test_dense_with_a_zero_query_embedding_returns_nothing(fake_embedder):
    """Every cosine is 0/0, which the store raises on and the rule reads as 0. Found by the
    end-to-end test below, whose fake embedding maps most T1 prompts to the zero vector."""
    corpus = [Document(id="gpu_doc", text="request a gpu", tags=[])]
    assert retrieve_dense(_task(prompt="hello world"), corpus) == []
    assert retrieve_dense(_task(prompt="request a gpu"), [
        Document(id="blank", text="hello world", tags=[])]) == []


def test_dense_on_an_empty_corpus_never_loads_the_model(monkeypatch):
    """Nothing to rank means nothing to embed, and no reason to need the extra."""
    def _fail():
        raise AssertionError("the embedder was loaded for an empty corpus")

    monkeypatch.setattr(retrieval, "dense_embedder", _fail)
    assert retrieve_dense(_task(prompt="request a gpu"), []) == []


def test_dense_is_deterministic_and_breaks_ties_by_corpus_order(fake_embedder):
    """Three documents with the same embedding. The vector store alone returns them in the
    order a reversed, unstable argsort leaves them, which is the reverse of the corpus here;
    the order is fixed after the search, and this is what would catch that step going."""
    corpus = [
        Document(id="first", text="gpu", tags=[]),
        Document(id="second", text="accelerator", tags=[]),
        Document(id="third", text="cuda", tags=[]),
    ]
    task = _task(prompt="gpus")
    runs = [[d.id for d in retrieve_dense(task, corpus, k=3)] for _ in range(3)]
    assert runs == [["first", "second", "third"]] * 3


def test_dense_retrieves_through_the_langchain_vector_store(fake_embedder, monkeypatch):
    """The arm is defined as a LangChain retrieval pipeline, not as an embedding model with
    anvil's own ranking around it. Every query goes through the store's scored search."""
    from langchain_core.vectorstores import InMemoryVectorStore  # noqa: PLC0415

    calls = []
    original = InMemoryVectorStore.similarity_search_with_score_by_vector

    def _spy(self, embedding, *args, **kwargs):
        calls.append(embedding)
        return original(self, embedding, *args, **kwargs)

    monkeypatch.setattr(InMemoryVectorStore, "similarity_search_with_score_by_vector", _spy)
    corpus = [Document(id="gpu_doc", text="request a gpu", tags=[])]
    assert [d.id for d in retrieve_dense(_task(prompt="one gpu please"), corpus)] == ["gpu_doc"]
    assert calls == [fake_embedder.embed_query("one gpu please")]


def test_dense_without_langchain_is_a_refusal_not_an_import_error(monkeypatch):
    """The pipeline half of the extra missing, with the embedding half faked: still the one
    line the CLI prints, never a bare ImportError from inside a retrieval."""
    from anvil.errors import UnsupportedRequest  # noqa: PLC0415

    monkeypatch.setattr(retrieval, "dense_embedder", _SynonymEmbeddings)
    monkeypatch.setitem(sys.modules, "langchain_core.vectorstores", None)
    corpus = [Document(id="gpu_doc", text="request a gpu", tags=[])]
    with pytest.raises(UnsupportedRequest, match=r"\[dense\]"):
        retrieve_dense(_task(prompt="one gpu please"), corpus)


def test_every_strategy_is_named_in_the_cli_help(capsys):
    """`choices` is read from STRATEGIES, the help string is typed by hand, and only the
    second can fall behind: `dense` was a valid value before this test existed while the
    help still listed three arms."""
    from anvil.cli import main  # noqa: PLC0415

    with pytest.raises(SystemExit):
        main(["run", "--help"])
    flat = re.sub(r"\s+", "", capsys.readouterr().out)
    for name in STRATEGIES:
        assert name in flat, name


def test_dense_without_the_extra_is_one_line_not_a_traceback(monkeypatch, capsys, tmp_path):
    """Checked before the model loads, so the answer costs a second, and raised as a
    request this install cannot honour, so the CLI prints it instead of a stack."""
    from anvil.cli import main  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)
    retrieval.dense_embedder.cache_clear()
    try:
        code = main(["run", "--model", "oracle", "--retrieval", "dense",
                     "--out", str(tmp_path / "out.json")])
    finally:
        retrieval.dense_embedder.cache_clear()
    err = capsys.readouterr().err
    assert code == 2
    assert 'pip install -e ".[dense]"' in err
    assert "Traceback" not in err
    assert not (tmp_path / "out.json").exists()


def test_dense_runs_end_to_end_and_leaves_the_oracle_at_its_bound(monkeypatch, tmp_path):
    """An arm that changed what the upper bound scores would measure the harness, not the
    model. Every task gets its documents recorded beside its script, which is what the
    copying analysis reads."""
    from anvil import cli  # noqa: PLC0415

    _needs_the_pipeline()
    fake = _SynonymEmbeddings()
    monkeypatch.setattr(retrieval, "dense_embedder", lambda: fake)
    monkeypatch.setattr(cli, "dense_embedder", lambda: fake)
    out, gens = tmp_path / "out.json", tmp_path / "gens.jsonl"
    code = cli.main(["run", "--model", "oracle", "--retrieval", "dense", "--no-exec",
                     "--out", str(out), "--save-generations", str(gens)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["retrieval"] == "dense"
    # `submittability` needs sbatch and `functional` is off, and a skipped level is never a
    # passed one; the three below are decided by the script alone, wherever this runs.
    for level in ("syntax", "resource_fit", "safety"):
        assert report["summary"][level]["pass@1"] == 1.0, level
    ids = {d.id for d in Document.load_jsonl(CORPUS)}
    rows = [json.loads(line) for line in gens.read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["retrieval"] == "dense" for row in rows)
    assert all(set(row["retrieved_docs"]) <= ids for row in rows)
    assert fake.calls > 0
    # The record names the retriever, so two revisions of the embedding model can be told
    # apart after the fact, and carries the score each attached document was chosen by.
    assert report["retrieval_model"] == provenance("dense")
    for row in rows:
        assert row["retrieval_model"] == provenance("dense")
        scores = row["retrieved_scores"]
        assert len(scores) == len(row["retrieved_docs"])
        assert scores == sorted(scores, reverse=True) and all(x > 0 for x in scores)


# ---------------------------------------------------------------- provenance and scores
def test_a_scoring_retriever_is_its_ranking_cut_at_k():
    """The CLI records scores from the ranking and attaches documents from it, so the arm
    as published and the arm as recorded are one list. Checked on the real corpus for
    every T1 task, where the published `vector` numbers came from."""
    tasks = Task.load_jsonl(ROOT / "tasks" / "t1_slurm.jsonl")
    corpus = Document.load_jsonl(CORPUS)
    for task in tasks:
        ranked = rank_vector(task, corpus)
        scores = [score for score, _ in ranked]
        assert scores == sorted(scores, reverse=True) and all(x > 0 for x in scores)
        for k in (1, 2, 3):
            assert retrieve_vector(task, corpus, k) == [d for _, d in ranked[:k]], (task.id, k)


def test_dense_is_its_ranking_cut_at_k(fake_embedder):
    corpus = Document.load_jsonl(CORPUS)
    task = _task(prompt="request a gpu, an array index and some memory")
    for k in (1, 2, 3):
        assert retrieve_dense(task, corpus, k) == [
            d for _, d in RANKERS["dense"](task, corpus)[:k]
        ]


def test_arms_record_scores_only_when_they_score(tmp_path):
    """`vector` scores and depends on nothing outside the package; `zero-shot` and
    `vectorless` do not score. A key present with nothing behind it would read as a
    measurement."""
    from anvil.cli import main  # noqa: PLC0415

    for arm, scored in (("zero-shot", False), ("vectorless", False), ("vector", True)):
        gens = tmp_path / f"{arm}.jsonl"
        assert main(["run", "--model", "oracle", "--retrieval", arm, "--no-exec",
                     "--save-generations", str(gens)]) == 0
        rows = [json.loads(line) for line in gens.read_text(encoding="utf-8").splitlines()]
        assert rows
        assert all(("retrieved_scores" in row) is scored for row in rows), arm
        assert all("retrieval_model" not in row for row in rows), arm


# ---------------------------------------------------------------- prompt augmentation
def test_build_prompt_with_context_no_docs_returns_prompt_unchanged():
    assert build_prompt_with_context("write a script", []) == "write a script"


def test_build_prompt_with_context_starts_with_original_prompt():
    """The default position, the one every published arm was measured at. The oracle
    used to rely on it and no longer does: it matches the task prompt anywhere, which
    `test_oracle_resolves_the_task_under_either_position` holds."""
    docs = [Document(id="d1", text="some reference text", tags=[])]
    augmented = build_prompt_with_context("write a script", docs)
    assert augmented.startswith("write a script")
    assert "some reference text" in augmented
    assert "d1" in augmented


# ---------------------------------------------------------------- context position
def test_append_is_unchanged():
    """Every published arm was measured at `append`, so this is a regression test on the
    exact bytes and not merely on the ordering."""
    docs = [Document(id="d1", text="reference text", tags=[])]
    assert build_prompt_with_context("write a script", docs) == (
        "write a script\n\nReference material:\n[d1]\nreference text"
    )


def test_prepend_puts_the_context_first_and_the_task_last():
    docs = [Document(id="d1", text="reference text", tags=[])]
    augmented = build_prompt_with_context("write a script", docs, "prepend")
    assert augmented.startswith("Reference material:")
    assert augmented.endswith("write a script")
    assert "reference text" in augmented


def test_position_is_validated():
    docs = [Document(id="d1", text="reference text", tags=[])]
    with pytest.raises(ValueError):
        build_prompt_with_context("write a script", docs, "middle")


def test_oracle_resolves_the_task_under_either_position():
    """The oracle indexes tasks by prompt. Appending left the prompt at the front, which
    `startswith` was enough for; prepending does not, and a benchmark whose upper bound
    silently returns empty scripts under one arm measures nothing."""
    from anvil.models import OracleModel  # noqa: PLC0415

    tasks = Task.load_jsonl(ROOT / "tasks" / "t1_slurm.jsonl")
    oracle = OracleModel(ROOT / "tasks" / "t1_reference.jsonl", ROOT / "tasks" / "t1_slurm.jsonl")
    corpus = Document.load_jsonl(CORPUS)
    for task in tasks:
        docs = retrieve_vectorless(task, corpus)
        plain = oracle.generate(task.prompt)[0]
        assert "ANVIL_OK" in plain or plain.strip() != "```bash\n```", task.id
        for position in ("append", "prepend"):
            augmented = build_prompt_with_context(task.prompt, docs, position)
            assert oracle.generate(augmented)[0] == plain, (task.id, position)


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

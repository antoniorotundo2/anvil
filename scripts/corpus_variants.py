#!/usr/bin/env python3
"""Build the corpus variants the retrieval intervention series is measured on.

`docs/DESIGN.md` publishes seven conditions under Retrieval ablation, and five of them
differ from the default only in which document ends up attached to every task. Those
conditions were first produced by hand, which left a table in the documentation whose
inputs existed nowhere anybody else could reach. This script is the inputs.

Three variants, each written beside the default corpus rather than over it, so
`tasks/retrieval_corpus.jsonl` keeps the digest every published arm was measured with:

  timemem_first      the two `general` documents swapped, so `doc_time_mem` takes the
                     fallback slot instead of `doc_directive_placement`
  control_offtopic   an off-topic passage on coastal tides takes that slot
  control_offtopic2  a second one, on sourdough, takes it

`retrieve_vectorless` fills its remaining slots from the documents tagged `general` in
corpus order, so the first of them is attached to every task and the rest are never
seen. Position in the file is the whole mechanism, which is why a variant is a
reordering and not an edit.

The controls exist to separate relevance from volume. Each is cut to exactly the length
of the document it displaces, so the prompt grows by the same number of characters and
the only thing that changes is whether those characters concern the domain. Neither
passage shares a term with SLURM, batch scheduling or the shell.

`build_prompt_with_context` writes each document as `[id]` and then its text, so the id
reaches the model too. The control ids are therefore the ones the published runs used
and are not renamed for readability: a different id is a different prompt. It also means
length parity holds for the text and not quite for the whole document, the control ids
being eight characters longer than `doc_time_mem`, which is eight characters in about
three hundred and thirty and carries no domain vocabulary either way.

    ./scripts/corpus_variants.py                    # writes results/corpus_variants/
    ./scripts/corpus_variants.py /tmp/variants      # or wherever

Each variant is printed with the document `vectorless` actually attaches to each T1
task, because that is the experimental precondition and it is worth reading before
spending GPU time rather than after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anvil.retrieval import Document, retrieve_vectorless  # noqa: E402
from anvil.schema import Task  # noqa: E402

CORPUS = ROOT / "tasks" / "retrieval_corpus.jsonl"
TASKS = ROOT / "tasks" / "t1_slurm.jsonl"

# The document whose slot every variant contests, and the one that takes it by default.
SUBJECT = "doc_time_mem"
INCUMBENT = "doc_directive_placement"

# Off-topic filler, keyed by the document id it is published under. Repeated and then cut
# to the subject's length, so the two controls and the document they displace are the same
# size to the character. Written out here rather than generated, because a control that
# changes between runs is not a control.
CONTROLS: dict[str, str] = {
    "doc_control_offtopic": (
        "Tidal patterns along a coastline follow the gravitational pull of the moon and "
        "the sun, with two high waters and two low waters in most places each lunar day. "
        "The range between them varies with the phase of the moon: spring tides near new "
        "and full moon, neap tides near the quarters. Local geography matters more than "
        "either, since a narrowing estuary amplifies the range while a broad shelf damps "
        "it, which is why two ports on the same coast can differ by several metres on the "
        "same afternoon. "
    ),
    "doc_control_offtopic2": (
        "Sourdough leavening depends on a culture of wild yeasts and lactic acid bacteria "
        "that the baker keeps alive by feeding it flour and water on a regular rhythm. "
        "Temperature sets the pace: a kitchen at eighteen degrees stretches a rise into "
        "the night, while twenty six degrees can halve it. Bakers judge readiness by "
        "volume and smell rather than the clock, since the same culture behaves "
        "differently from one week to the next. "
    ),
}


def _load_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_variants() -> dict[str, list[dict]]:
    """Variant name to its corpus, as rows ready to serialise.

    Every variant keeps all the original documents. A control adds one, it never
    removes: displacing `doc_time_mem` from the fallback slot is enough to keep it out
    of what `vectorless` attaches, and deleting it would confound the comparison with
    a corpus that is also smaller.
    """
    rows = _load_rows()
    subject = next(d for d in rows if d["id"] == SUBJECT)

    variants: dict[str, list[dict]] = {}

    swapped = list(rows)
    i = next(k for k, d in enumerate(swapped) if d["id"] == INCUMBENT)
    j = next(k for k, d in enumerate(swapped) if d["id"] == SUBJECT)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    variants["timemem_first"] = swapped

    for doc_id, filler in CONTROLS.items():
        text = (filler * 4)[: len(subject["text"])]
        control = {"id": doc_id, "text": text, "tags": ["general"]}
        variants[doc_id.removeprefix("doc_")] = [control] + rows

    return variants


def attachments(rows: list[dict]) -> dict[str, list[str]]:
    """What `vectorless` attaches to each T1 task under this corpus."""
    corpus = [Document(**row) for row in rows]
    return {
        task.id: [d.id for d in retrieve_vectorless(task, corpus)]
        for task in Task.load_jsonl(TASKS)
    }


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "corpus_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in build_variants().items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        general = [d["id"] for d in rows if "general" in d.get("tags", [])]
        print(f"{path}")
        print(f"  general documents, in the order the fallback takes them: {', '.join(general)}")
        for task_id, docs in attachments(rows).items():
            print(f"    {task_id:<24} {', '.join(docs)}")
        print()

    print("Use one with the ablation, which passes it through to `anvil run`:")
    print(f"  STRATEGIES=vectorless CORPUS={out_dir}/control_offtopic.jsonl \\")
    print("    ./scripts/retrieval_ablation.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

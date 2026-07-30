"""Anvil command-line interface.

    python -m anvil.cli doctor
    python -m anvil.cli run --model oracle --tasks tasks/t1_slurm.jsonl
    python -m anvil.cli run --model broken --tasks tasks/t1_slurm.jsonl -n 6
    python -m anvil.cli run --model Qwen/Qwen2.5-Coder-1.5B-Instruct --tasks tasks/t1_slurm.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .inducer import FAULT_CATEGORIES
from .metrics import aggregate, aggregate_by_category, aggregate_recipes
from .models import build_model, build_recipe_model, reference_path_for
from .parse import extract_script
from .recipe_parse import extract_recipe
from .recipe_verifier import apptainer_available, verify_recipe
from .repair import (
    build_repair_model,
    build_repair_prompt,
    induce_t2_tasks,
    verify_repair,
)
from .retrieval import STRATEGIES, Document, build_prompt_with_context
from .schema import Level, RecipeLevel, RecipeTask, RepairTask, Task
from .verifier import (
    FUNCTIONAL_EXECUTORS,
    sbatch_execution_healthy,
    set_functional_executor,
    slurm_healthy,
    verify,
)


def _fmt(v: object, w: int) -> str:
    return str(v).ljust(w)


def _add_executor_flag(parser: argparse.ArgumentParser) -> None:
    """The default is None, not "bash": an absent flag must leave
    ANVIL_FUNCTIONAL_EXECUTOR alone instead of quietly overriding it."""
    parser.add_argument(
        "--executor", choices=list(FUNCTIONAL_EXECUTORS), default=None,
        help="how the 'functional' level executes: bash (default, sandbox) | sbatch "
        "(real submission; needs a scheduler that actually runs jobs). Also settable "
        "with ANVIL_FUNCTIONAL_EXECUTOR",
    )


def _file_sha(path: str | Path) -> str:
    """Short digest of the task file.

    Generations are produced *for a specific task set*. Verifying them against a
    different one is a silent scientific error: the scripts answer questions that
    were never asked. The digest travels with the generations so `verify` can refuse.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what this machine can do. Also feeds the paper's setup section:
    hardware and toolchain must be declared precisely."""
    from .device import environment_report

    rep = environment_report()
    notes = rep.pop("notes")

    print("Anvil - environment")
    print("-" * 62)
    for key, val in rep.items():
        print(f"{_fmt(key, 20)}{val}")
    print("-" * 62)

    healthy, why = slurm_healthy()
    print("\nVerification levels:")
    print("  syntax          always active")
    print(f"  submittability  {'ACTIVE' if healthy else 'SKIPPED - ' + why}")
    # Which executor runs `functional` has to be stated precisely, not left to the reader:
    # the string ends up in environment.json and therefore in the paper's setup section.
    # `bash` is the default and simulates the scheduler; `sbatch` submits for real, and
    # then a scheduler that accepts jobs without running them skips the level.
    if rep["functional_executor"] == "sbatch":
        runs, why_exec = sbatch_execution_healthy()
        status = "ACTIVE via sbatch (real submission)" if runs else f"SKIPPED - {why_exec}"
        print(f"  functional      {status}")
    else:
        print("  functional      active (bash sandbox; NOT via sbatch)")
    print("  resource_fit    always active")
    print("  safety          always active")

    if not rep["gnu_faithful"]:
        print(
            f"\n[warning] coreutils detected: {rep['coreutils']}\n"
            "          The `functional` level executes generated scripts: on a non-GNU\n"
            "          implementation the outcome may diverge from the cluster.\n"
            "          For valid results run inside the container:\n"
            "            docker build -t anvil docker/\n"
            '            docker run --rm -v "$PWD":/work -w /work anvil pytest -q'
        )

    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")

    if args.json:
        rep["notes"] = notes
        print("\n" + json.dumps(rep, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    tasks = Task.load_jsonl(args.tasks)
    model_kw: dict = {}
    if args.model not in ("oracle", "broken"):
        model_kw = {
            "load_in_4bit": args.load_in_4bit,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }
    model = build_model(args.model, args.tasks, **model_kw)
    tasks_sha = _file_sha(args.tasks)

    corpus = Document.load_jsonl(args.retrieval_corpus) if args.retrieval != "zero-shot" else []
    retrieve = STRATEGIES[args.retrieval]

    healthy, why = slurm_healthy()
    if not healthy:
        print(
            f"[warning] level 'submittability' SKIPPED (not counted as passed): {why}\n",
            file=sys.stderr,
        )

    results = []
    generations: list[dict] = []
    t0 = time.time()
    for task in tasks:
        docs = retrieve(task, corpus, k=args.retrieval_k)
        prompt = build_prompt_with_context(task.prompt, docs)
        raw_outputs = model.generate(prompt, n=args.n, seed=args.seed)
        first = len(results)
        for sample_idx, raw in enumerate(raw_outputs):
            script = extract_script(raw)
            generations.append({
                "task_id": task.id,
                "sample": sample_idx,
                "model": model.name,
                "seed": args.seed,
                "tasks_sha": tasks_sha,
                "retrieval": args.retrieval,
                "retrieved_docs": [d.id for d in docs],
                "script": script,
            })
            results.append(verify(script, task, run_functional=not args.no_exec))
        if args.verbose:
            _print_task_detail(task, results[first:])
    elapsed = time.time() - t0

    if args.save_generations:
        with open(args.save_generations, "w", encoding="utf-8") as fh:
            for g in generations:
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"Generations written to {args.save_generations} "
              f"({len(generations)} scripts) — verify them elsewhere with `anvil verify`.")

    _report(model.name, args.tasks, tasks, results, args, elapsed)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify previously generated scripts. No model, no GPU.

    Generation needs the machine with the accelerator; faithful verification needs
    the machine with the scheduler and GNU coreutils. Decoupling them lets you
    generate once and verify anywhere — including across several base images, to
    test whether artifact correctness is portable between distributions.
    """
    tasks = {t.id: t for t in Task.load_jsonl(args.tasks)}

    healthy, why = slurm_healthy()
    if not healthy:
        print(
            f"[warning] level 'submittability' SKIPPED (not counted as passed): {why}\n",
            file=sys.stderr,
        )

    expected_sha = _file_sha(args.tasks)

    results = []
    models = set()
    unknown: set[str] = set()
    shas: set[str] = set()
    t0 = time.time()
    with open(args.generations, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            g = json.loads(line)
            task = tasks.get(g["task_id"])
            if task is None:
                unknown.add(g["task_id"])
                continue
            models.add(g.get("model", "?"))
            shas.add(g.get("tasks_sha", "unknown"))
            results.append(verify(g["script"], task, run_functional=not args.no_exec))
            if args.verbose:
                _print_task_detail(task, results[-1:])
    elapsed = time.time() - t0

    if unknown:
        print(f"[warning] {len(unknown)} generations reference unknown task ids: "
              f"{sorted(unknown)[:3]}", file=sys.stderr)

    stale = shas - {expected_sha}
    if stale:
        print(
            f"[ERROR] these generations were produced against a different task file "
            f"(theirs: {sorted(stale)}, current: {expected_sha}).\n"
            f"        Verifying them would score answers to questions that were never "
            f"asked. Re-run `make generate`.",
            file=sys.stderr,
        )
        return 2

    if not results:
        print("No generations verified.", file=sys.stderr)
        return 1

    name = models.pop() if len(models) == 1 else f"{len(models)} models"
    _report(name, args.tasks, list(tasks.values()), results, args, elapsed)
    return 0


def cmd_induce(args: argparse.Namespace) -> int:
    """Build the T2 task set: mechanically break every T1 reference solution
    (see anvil/inducer.py) and keep only the variants that actually fail
    verification. An inducer that produces an accidentally-valid script is a
    bug in the inducer, not a fault worth teaching a model to repair."""
    # The induced task set is part of the benchmark definition, so it must not depend on
    # which executor happens to be selected: a fault that survives bash but is caught by a
    # real submission would silently drop out of t2_repair.jsonl. Pinned, not offered.
    set_functional_executor("bash")

    tasks = Task.load_jsonl(args.tasks)
    reference: dict[str, str] = {}
    with open(args.reference, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                reference[rec["id"]] = rec["script"]

    repair_tasks, warnings = induce_t2_tasks(tasks, reference, run_functional=not args.no_exec)
    for w in warnings:
        print(f"[induce] {w}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fh:
        for rt in repair_tasks:
            fh.write(json.dumps(asdict(rt), ensure_ascii=False) + "\n")

    by_category: dict[str, int] = {}
    for rt in repair_tasks:
        by_category[rt.fault_category] = by_category.get(rt.fault_category, 0) + 1

    print(f"Induced {len(repair_tasks)} T2 tasks from {len(tasks)} T1 tasks -> {args.out}")
    for cat in sorted(by_category):
        print(f"  {cat} ({FAULT_CATEGORIES[cat]}): {by_category[cat]}")
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    t1_tasks = Task.load_jsonl(args.tasks)
    tasks_by_id = {t.id: t for t in t1_tasks}
    repair_tasks = RepairTask.load_jsonl(args.repair_tasks)

    unknown = [rt.id for rt in repair_tasks if rt.base_task_id not in tasks_by_id]
    if unknown:
        print(
            f"[warning] {len(unknown)} repair tasks reference unknown base task ids: "
            f"{unknown[:3]}",
            file=sys.stderr,
        )
        repair_tasks = [rt for rt in repair_tasks if rt.base_task_id in tasks_by_id]

    model_kw: dict = {}
    if args.model not in ("oracle", "broken"):
        model_kw = {
            "load_in_4bit": args.load_in_4bit,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }
    model = build_repair_model(
        args.model, reference_path_for(args.tasks), t1_tasks, **model_kw
    )
    repair_tasks_sha = _file_sha(args.repair_tasks)

    healthy, why = slurm_healthy()
    if not healthy:
        print(
            f"[warning] level 'submittability' SKIPPED (not counted as passed): {why}\n",
            file=sys.stderr,
        )

    results = []
    generations: list[dict] = []
    t0 = time.time()
    for rt in repair_tasks:
        base_task = tasks_by_id[rt.base_task_id]
        prompt = build_repair_prompt(base_task, rt.broken_script)
        raw_outputs = model.generate(prompt, n=args.n, seed=args.seed)
        first = len(results)
        for sample_idx, raw in enumerate(raw_outputs):
            script = extract_script(raw)
            generations.append({
                "repair_task_id": rt.id,
                "base_task_id": rt.base_task_id,
                "fault_category": rt.fault_category,
                "sample": sample_idx,
                "model": model.name,
                "seed": args.seed,
                "repair_tasks_sha": repair_tasks_sha,
                "script": script,
            })
            results.append(
                verify_repair(script, rt, base_task, run_functional=not args.no_exec)
            )
        if args.verbose:
            _print_repair_detail(rt, results[first:])
    elapsed = time.time() - t0

    if args.save_generations:
        with open(args.save_generations, "w", encoding="utf-8") as fh:
            for g in generations:
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(
            f"Generations written to {args.save_generations} ({len(generations)} scripts) "
            "— verify them elsewhere with `anvil verify-repair`."
        )

    categories = {rt.id: rt.fault_category for rt in repair_tasks}
    _report(model.name, args.repair_tasks, repair_tasks, results, args, elapsed, categories)
    return 0


def cmd_verify_repair(args: argparse.Namespace) -> int:
    """Verify previously generated repairs. No model, no GPU."""
    t1_tasks = {t.id: t for t in Task.load_jsonl(args.tasks)}
    repair_tasks = {rt.id: rt for rt in RepairTask.load_jsonl(args.repair_tasks)}

    healthy, why = slurm_healthy()
    if not healthy:
        print(
            f"[warning] level 'submittability' SKIPPED (not counted as passed): {why}\n",
            file=sys.stderr,
        )

    expected_sha = _file_sha(args.repair_tasks)

    results = []
    models = set()
    unknown: set[str] = set()
    shas: set[str] = set()
    t0 = time.time()
    with open(args.generations, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            g = json.loads(line)
            rt = repair_tasks.get(g.get("repair_task_id"))
            base_task = t1_tasks.get(g.get("base_task_id"))
            if rt is None or base_task is None:
                unknown.add(g.get("repair_task_id", "?"))
                continue
            models.add(g.get("model", "?"))
            shas.add(g.get("repair_tasks_sha", "unknown"))
            results.append(
                verify_repair(g["script"], rt, base_task, run_functional=not args.no_exec)
            )
            if args.verbose:
                _print_repair_detail(rt, results[-1:])
    elapsed = time.time() - t0

    if unknown:
        print(
            f"[warning] {len(unknown)} generations reference unknown repair task ids: "
            f"{sorted(unknown)[:3]}",
            file=sys.stderr,
        )

    stale = shas - {expected_sha}
    if stale:
        print(
            f"[ERROR] these generations were produced against a different repair task file "
            f"(theirs: {sorted(stale)}, current: {expected_sha}).\n"
            f"        Re-run `anvil induce` / `anvil repair`.",
            file=sys.stderr,
        )
        return 2

    if not results:
        print("No generations verified.", file=sys.stderr)
        return 1

    name = models.pop() if len(models) == 1 else f"{len(models)} models"
    categories = {rid: rt.fault_category for rid, rt in repair_tasks.items()}
    _report(
        name, args.repair_tasks, list(repair_tasks.values()), results, args, elapsed, categories
    )
    return 0


def cmd_recipe(args: argparse.Namespace) -> int:
    """T3: write an Apptainer definition file from scratch."""
    tasks = RecipeTask.load_jsonl(args.tasks)
    model_kw: dict = {}
    if args.model not in ("oracle", "broken"):
        model_kw = {
            "load_in_4bit": args.load_in_4bit,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }
    model = build_recipe_model(args.model, args.tasks, **model_kw)
    tasks_sha = _file_sha(args.tasks)

    if not apptainer_available():
        print(
            "[warning] levels 'buildable'/'functional' SKIPPED (not counted as passed): "
            "apptainer not available\n",
            file=sys.stderr,
        )

    results = []
    generations: list[dict] = []
    t0 = time.time()
    for task in tasks:
        raw_outputs = model.generate(task.prompt, n=args.n, seed=args.seed)
        first = len(results)
        for sample_idx, raw in enumerate(raw_outputs):
            recipe = extract_recipe(raw)
            generations.append({
                "task_id": task.id,
                "sample": sample_idx,
                "model": model.name,
                "seed": args.seed,
                "tasks_sha": tasks_sha,
                "recipe": recipe,
            })
            results.append(verify_recipe(recipe, task, run_functional=not args.no_exec))
        if args.verbose:
            _print_recipe_detail(task, results[first:])
    elapsed = time.time() - t0

    if args.save_generations:
        with open(args.save_generations, "w", encoding="utf-8") as fh:
            for g in generations:
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"Generations written to {args.save_generations} "
              f"({len(generations)} recipes) — verify them elsewhere with `anvil verify-recipe`.")

    _report_recipe(model.name, args.tasks, tasks, results, args, elapsed)
    return 0


def cmd_verify_recipe(args: argparse.Namespace) -> int:
    """Verify previously generated recipes. No model, no GPU, no apptainer
    needed unless you want the 'buildable'/'functional' levels active."""
    tasks = {t.id: t for t in RecipeTask.load_jsonl(args.tasks)}

    if not apptainer_available():
        print(
            "[warning] levels 'buildable'/'functional' SKIPPED (not counted as passed): "
            "apptainer not available\n",
            file=sys.stderr,
        )

    expected_sha = _file_sha(args.tasks)

    results = []
    models = set()
    unknown: set[str] = set()
    shas: set[str] = set()
    t0 = time.time()
    with open(args.generations, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            g = json.loads(line)
            task = tasks.get(g["task_id"])
            if task is None:
                unknown.add(g["task_id"])
                continue
            models.add(g.get("model", "?"))
            shas.add(g.get("tasks_sha", "unknown"))
            results.append(verify_recipe(g["recipe"], task, run_functional=not args.no_exec))
            if args.verbose:
                _print_recipe_detail(task, results[-1:])
    elapsed = time.time() - t0

    if unknown:
        print(f"[warning] {len(unknown)} generations reference unknown task ids: "
              f"{sorted(unknown)[:3]}", file=sys.stderr)

    stale = shas - {expected_sha}
    if stale:
        print(
            f"[ERROR] these generations were produced against a different task file "
            f"(theirs: {sorted(stale)}, current: {expected_sha}).\n"
            f"        Re-run `anvil recipe`.",
            file=sys.stderr,
        )
        return 2

    if not results:
        print("No generations verified.", file=sys.stderr)
        return 1

    name = models.pop() if len(models) == 1 else f"{len(models)} models"
    _report_recipe(name, args.tasks, list(tasks.values()), results, args, elapsed)
    return 0


def _sample_status(samples: Sequence) -> str:
    """Outcome across every sample drawn for one task.

    These lines used to report only the last sample, so a task whose final draw happened to
    pass read as clean even when the earlier draws had failed. pass@k is computed over all n
    samples, so the line watched while a run is in flight has to be too.
    """
    passed = sum(1 for res in samples if res.all_passed)
    if len(samples) == 1:
        return "PASS" if passed else "FAIL"
    return f"{passed}/{len(samples)} PASS"


def _one_line(detail: str, limit: int = 200) -> str:
    """Flatten a detail onto the single line this report gives it.

    A build failure carries the tail of the builder's output, several lines of it. The
    file written by `--out` keeps all of them; here only the head is shown, which is where
    the command that actually failed reports itself, ahead of the tool's own summary.
    """
    flat = " ".join(detail.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _failed_level_lines(samples: Sequence) -> list[str]:
    """One line per level that failed in at least one sample, and in how many of them."""
    counts: dict[str, int] = {}
    details: dict[str, str] = {}
    for res in samples:
        for lr in res.levels:
            if not lr.passed and not lr.skipped:
                counts[lr.level.value] = counts.get(lr.level.value, 0) + 1
                details.setdefault(lr.level.value, lr.detail)
    lines = []
    for level, count in counts.items():
        scope = "" if len(samples) == 1 else f" ({count}/{len(samples)} samples)"
        lines.append(f"      - {level}: {_one_line(details[level])}{scope}")
    return lines


def _print_repair_detail(rt: RepairTask, samples: Sequence) -> None:
    print(f"  {_fmt(rt.id, 30)} [{rt.fault_category}] {_sample_status(samples)}")
    for line in _failed_level_lines(samples):
        print(line)


def _print_task_detail(task: Task, samples: Sequence) -> None:
    print(f"  {_fmt(task.id, 26)} {_sample_status(samples)}")
    for line in _failed_level_lines(samples):
        print(line)


def _print_recipe_detail(task: RecipeTask, samples: Sequence) -> None:
    print(f"  {_fmt(task.id, 26)} {_sample_status(samples)}")
    for line in _failed_level_lines(samples):
        print(line)


def _print_summary_table(summary: dict, k: int, levels=Level) -> None:
    print("-" * 62)
    print(f"{_fmt('level', 22)}{_fmt(f'pass@{k}', 12)}{_fmt('skipped', 10)}")
    print("-" * 62)
    for level in [*[lv.value for lv in levels], "strict_all_levels"]:
        row = summary[level]
        print(
            f"{_fmt(level, 22)}"
            f"{_fmt(row[f'pass@{k}'], 12)}"
            f"{_fmt(row['n_skipped_samples'], 10)}"
        )
    print("-" * 62)


def _report(
    model_name, tasks_file, tasks, results, args, elapsed, categories: dict[str, str] | None = None
) -> None:
    """categories: {repair_task_id: fault_category}, only for T2 (repair/verify-repair).
    When given, breaks the summary down per fault category (F1-F7) in addition
    to the overall one — a category is invisible to pass@k otherwise, and the
    whole point of inducing faults by class is to see which classes a model
    actually repairs."""
    summary = aggregate(results, k=args.k)

    n_per_task = len(results) / max(len(tasks), 1)
    print(f"\nModel: {model_name}   tasks: {len(tasks)}   samples/task: {n_per_task:g}   "
          f"time: {elapsed:.2f}s")
    _print_summary_table(summary, args.k)

    by_category = None
    if categories:
        by_category = aggregate_by_category(results, categories, k=args.k)
        for cat, cat_summary in by_category.items():
            desc = FAULT_CATEGORIES.get(cat, "")
            print(f"\n[{cat}] {desc}")
            _print_summary_table(cat_summary, args.k)

    if args.out:
        from .device import environment_report

        env = environment_report()
        payload = {
            "model": model_name,
            "tasks_file": str(tasks_file),
            "k": args.k,
            "environment": env,          # base image, bash, coreutils, device: all recorded
            "elapsed_s": round(elapsed, 2),
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        if by_category is not None:
            payload["by_category"] = by_category
        if getattr(args, "retrieval", None):
            payload["retrieval"] = args.retrieval
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Full results written to {args.out}")


def _report_recipe(model_name, tasks_file, tasks, results, args, elapsed) -> None:
    summary = aggregate_recipes(results, k=args.k)

    n_per_task = len(results) / max(len(tasks), 1)
    print(f"\nModel: {model_name}   tasks: {len(tasks)}   samples/task: {n_per_task:g}   "
          f"time: {elapsed:.2f}s")
    _print_summary_table(summary, args.k, levels=RecipeLevel)

    if args.out:
        from .device import environment_report

        env = environment_report()
        payload = {
            "model": model_name,
            "tasks_file": str(tasks_file),
            "k": args.k,
            "environment": env,
            "elapsed_s": round(elapsed, 2),
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Full results written to {args.out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="anvil", description="Executable benchmark of HPC operational artifacts"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="report this environment's capabilities")
    d.add_argument("--json", action="store_true")
    _add_executor_flag(d)
    d.set_defaults(func=cmd_doctor)

    r = sub.add_parser("run", help="generate with a model and verify")
    r.add_argument("--model", required=True, help="'oracle' | 'broken' | a HF model_id")
    r.add_argument("--tasks", default="tasks/t1_slurm.jsonl")
    r.add_argument("-n", type=int, default=1, help="samples per task")
    r.add_argument("-k", type=int, default=1, help="budget for pass@k")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-exec", action="store_true", help="skip the functional level")
    r.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantization (requires CUDA + bitsandbytes)")
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--temperature", type=float, default=0.2)
    r.add_argument("--save-generations", metavar="PATH",
                   help="write the generated scripts to JSONL for later `anvil verify`")
    r.add_argument("--out", help="write full results to JSON")
    r.add_argument("--verbose", "-v", action="store_true")
    r.add_argument("--retrieval", choices=list(STRATEGIES), default="zero-shot",
                   help="retrieval ablation: zero-shot (default, no change) | vector "
                   "(TF-IDF similarity) | vectorless (tag match)")
    r.add_argument("--retrieval-corpus", default="tasks/retrieval_corpus.jsonl")
    r.add_argument("--retrieval-k", type=int, default=2,
                   help="max documents retrieved per task (ignored for zero-shot)")
    _add_executor_flag(r)
    r.set_defaults(func=cmd_run)

    v = sub.add_parser(
        "verify",
        help="verify previously generated scripts (no model, no GPU)",
        description="Generate where the accelerator is, verify where the scheduler is.",
    )
    v.add_argument("--generations", required=True, metavar="PATH",
                   help="JSONL produced by `anvil run --save-generations`")
    v.add_argument("--tasks", default="tasks/t1_slurm.jsonl")
    v.add_argument("-k", type=int, default=1, help="budget for pass@k")
    v.add_argument("--no-exec", action="store_true", help="skip the functional level")
    v.add_argument("--out", help="write full results to JSON")
    v.add_argument("--verbose", "-v", action="store_true")
    _add_executor_flag(v)
    v.set_defaults(func=cmd_verify)

    i = sub.add_parser(
        "induce",
        help="build tasks/t2_repair.jsonl from the T1 references",
        description="Mechanically break T1 reference solutions (F1-F7), keep only variants "
        "that actually fail verification.",
    )
    i.add_argument("--tasks", default="tasks/t1_slurm.jsonl")
    i.add_argument("--reference", default="tasks/t1_reference.jsonl")
    i.add_argument("--out", default="tasks/t2_repair.jsonl")
    i.add_argument("--no-exec", action="store_true",
                    help="skip the functional level when filtering induced variants")
    i.set_defaults(func=cmd_induce)

    rp = sub.add_parser("repair", help="generate a diagnose-and-repair with a model and verify")
    rp.add_argument("--model", required=True, help="'oracle' | 'broken' | a HF model_id")
    rp.add_argument("--repair-tasks", default="tasks/t2_repair.jsonl")
    rp.add_argument("--tasks", default="tasks/t1_slurm.jsonl", help="the T1 tasks the repair "
                     "tasks were induced from")
    rp.add_argument("-n", type=int, default=1, help="samples per repair task")
    rp.add_argument("-k", type=int, default=1, help="budget for pass@k")
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--no-exec", action="store_true", help="skip the functional level")
    rp.add_argument("--load-in-4bit", action="store_true",
                     help="4-bit quantization (requires CUDA + bitsandbytes)")
    rp.add_argument("--max-new-tokens", type=int, default=512)
    rp.add_argument("--temperature", type=float, default=0.2)
    rp.add_argument("--save-generations", metavar="PATH",
                     help="write the generated repairs to JSONL for later `anvil verify-repair`")
    rp.add_argument("--out", help="write full results to JSON")
    rp.add_argument("--verbose", "-v", action="store_true")
    _add_executor_flag(rp)
    rp.set_defaults(func=cmd_repair)

    vr = sub.add_parser(
        "verify-repair",
        help="verify previously generated repairs (no model, no GPU)",
        description="Generate where the accelerator is, verify where the scheduler is.",
    )
    vr.add_argument("--generations", required=True, metavar="PATH",
                     help="JSONL produced by `anvil repair --save-generations`")
    vr.add_argument("--repair-tasks", default="tasks/t2_repair.jsonl")
    vr.add_argument("--tasks", default="tasks/t1_slurm.jsonl")
    vr.add_argument("-k", type=int, default=1, help="budget for pass@k")
    vr.add_argument("--no-exec", action="store_true", help="skip the functional level")
    vr.add_argument("--out", help="write full results to JSON")
    vr.add_argument("--verbose", "-v", action="store_true")
    _add_executor_flag(vr)
    vr.set_defaults(func=cmd_verify_repair)

    rc = sub.add_parser("recipe", help="T3: write an Apptainer recipe with a model and verify")
    rc.add_argument("--model", required=True, help="'oracle' | 'broken' | a HF model_id")
    rc.add_argument("--tasks", default="tasks/t3_apptainer.jsonl")
    rc.add_argument("-n", type=int, default=1, help="samples per task")
    rc.add_argument("-k", type=int, default=1, help="budget for pass@k")
    rc.add_argument("--seed", type=int, default=0)
    rc.add_argument("--no-exec", action="store_true", help="skip the functional level")
    rc.add_argument("--load-in-4bit", action="store_true",
                     help="4-bit quantization (requires CUDA + bitsandbytes)")
    rc.add_argument("--max-new-tokens", type=int, default=512)
    rc.add_argument("--temperature", type=float, default=0.2)
    rc.add_argument("--save-generations", metavar="PATH",
                     help="write the generated recipes to JSONL for later `anvil verify-recipe`")
    rc.add_argument("--out", help="write full results to JSON")
    rc.add_argument("--verbose", "-v", action="store_true")
    rc.set_defaults(func=cmd_recipe)

    vc = sub.add_parser(
        "verify-recipe",
        help="verify previously generated Apptainer recipes (no model, no GPU)",
        description="Generate where the accelerator is, verify where apptainer is installed "
        "(docker/Dockerfile, WITH_APPTAINER=1).",
    )
    vc.add_argument("--generations", required=True, metavar="PATH",
                     help="JSONL produced by `anvil recipe --save-generations`")
    vc.add_argument("--tasks", default="tasks/t3_apptainer.jsonl")
    vc.add_argument("-k", type=int, default=1, help="budget for pass@k")
    vc.add_argument("--no-exec", action="store_true", help="skip the functional level")
    vc.add_argument("--out", help="write full results to JSON")
    vc.add_argument("--verbose", "-v", action="store_true")
    vc.set_defaults(func=cmd_verify_recipe)

    args = p.parse_args(argv)
    # One place, so no subcommand can forget it: the executor is module state in the
    # verifier, read back by the environment report that travels with every result.
    if getattr(args, "executor", None):
        set_functional_executor(args.executor)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

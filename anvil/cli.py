"""Anvil command-line interface.

    python -m anvil.cli doctor
    python -m anvil.cli run --model oracle --tasks tasks/t1_slurm.jsonl
    python -m anvil.cli run --model broken --tasks tasks/t1_slurm.jsonl -n 6
    python -m anvil.cli run --model Qwen/Qwen2.5-Coder-1.5B-Instruct --tasks tasks/t1_slurm.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .metrics import aggregate
from .models import build_model
from .parse import extract_script
from .schema import Level, Task
from .verifier import slurm_healthy, verify


def _fmt(v: object, w: int) -> str:
    return str(v).ljust(w)


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
    # NB: check_functional ALWAYS executes with bash in a temporary sandbox; it
    # never submits to sbatch. Stating this precisely is not pedantry: the string
    # ends up in environment.json and therefore in the paper's setup section.
    # Real execution via sbatch is planned for Phase 2.
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

    healthy, why = slurm_healthy()
    if not healthy:
        print(
            f"[warning] level 'submittability' SKIPPED (not counted as passed): {why}\n",
            file=sys.stderr,
        )

    results = []
    t0 = time.time()
    for task in tasks:
        raw_outputs = model.generate(task.prompt, n=args.n, seed=args.seed)
        for raw in raw_outputs:
            script = extract_script(raw)
            results.append(verify(script, task, run_functional=not args.no_exec))
        if args.verbose:
            last = results[-1]
            status = "PASS" if last.all_passed else "FAIL"
            print(f"  {_fmt(task.id, 26)} {status}")
            if not last.all_passed:
                for lr in last.levels:
                    if not lr.passed and not lr.skipped:
                        print(f"      - {lr.level.value}: {lr.detail}")
    elapsed = time.time() - t0

    summary = aggregate(results, k=args.k)

    print(f"\nModel: {model.name}   tasks: {len(tasks)}   samples/task: {args.n}   "
          f"time: {elapsed:.1f}s")
    print("-" * 62)
    print(f"{_fmt('level', 22)}{_fmt(f'pass@{args.k}', 12)}{_fmt('skipped', 10)}")
    print("-" * 62)
    for level in [*[lv.value for lv in Level], "strict_all_levels"]:
        row = summary[level]
        print(
            f"{_fmt(level, 22)}"
            f"{_fmt(row[f'pass@{args.k}'], 12)}"
            f"{_fmt(row['n_skipped_samples'], 10)}"
        )
    print("-" * 62)

    if args.out:
        payload = {
            "model": model.name,
            "tasks_file": str(args.tasks),
            "n_samples": args.n,
            "k": args.k,
            "slurm_healthy": slurm_healthy()[0],
            "elapsed_s": round(elapsed, 2),
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Full results written to {args.out}")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="anvil", description="Executable benchmark of HPC operational artifacts"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="report this environment's capabilities")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_doctor)

    r = sub.add_parser("run", help="run the benchmark")
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
    r.add_argument("--out", help="write full results to JSON")
    r.add_argument("--verbose", "-v", action="store_true")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

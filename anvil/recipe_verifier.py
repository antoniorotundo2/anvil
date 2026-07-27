"""The T3 verifier: Apptainer recipes, correctness measured by build and
execution, mirroring the T1/T2 philosophy for a different artifact.

Requires a real `apptainer` (or `singularity`) binary. Inside Docker the package
in `docker/Dockerfile` (`WITH_APPTAINER=1`) has no setuid starter, so every build
and run goes through a user namespace, which `_unprivileged()` below selects.

That route needs exemptions from Docker's confinement, not capabilities. Each one
answers a specific failure, found one CI run at a time and listed with its error in
`docs/DESIGN.md`: seccomp for the namespace, AppArmor because `docker-default`
denies the `mount` syscall outright, `systempaths` because a fresh procfs cannot be
mounted while Docker's masked `/proc` entries cover the original, plus a
subuid/subgid range for root in the image and, on an Ubuntu 24.04 host,
`kernel.apparmor_restrict_unprivileged_userns=0`.

One configuration covers both verified environments: it passes on GitHub-hosted
runners and on WSL2 with byte-identical numbers. `--privileged` also works and
collapses the list, but grants every capability besides. On Docker Desktop for Mac
`run` failed with `exec ... failed: invalid argument`, once blamed on the nested
`linuxkit` VM; the AppArmor findings make that explanation doubtful and it has not
been retested.

Degrades gracefully: when no `apptainer` binary is reachable, `buildable` is
marked `skipped` (never "passed"), same discipline as `submittability` in
verifier.py when no scheduler is reachable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .recipe_parse import list_sections, parse_header
from .schema import LevelResult, RecipeLevel, RecipeTask, RecipeVerificationResult
from .verifier import DANGEROUS  # the same probe applies to %post/%runscript


def apptainer_available() -> bool:
    return shutil.which("apptainer") is not None or shutil.which("singularity") is not None


def _apptainer_bin() -> str:
    return shutil.which("apptainer") or shutil.which("singularity") or "apptainer"


def _unprivileged() -> bool:
    """Whether to push apptainer through a user namespace of its own.

    The container runs as root, so apptainer takes the privileged path and mounts directly.
    Docker withholds CAP_SYS_ADMIN, so that mount fails with `Failed to set mount
    propagation: Permission denied`, during the %post scriptlet and again when running the
    image. A user namespace supplies that capability inside itself, so nothing has to be
    granted from outside the container.

    The Makefile turns this on by default for containerised runs, since the image's apptainer
    is not setuid and the route is confirmed on both verified hosts. It stays a switch so a
    host apptainer that *is* setuid can take the privileged path instead.
    """
    return os.environ.get("ANVIL_APPTAINER_UNPRIVILEGED", "0") not in ("", "0")


# --------------------------------------------------------------------------
# L1 - syntax: a minimally well-formed recipe
# --------------------------------------------------------------------------
def check_recipe_syntax(recipe: str) -> LevelResult:
    problems: list[str] = []
    header = parse_header(recipe)
    if "bootstrap" not in header:
        problems.append("missing Bootstrap: header")
    if "from" not in header:
        problems.append("missing From: header")

    sections = list_sections(recipe)
    if not sections:
        problems.append("no %section found")
    elif "runscript" not in sections and "startscript" not in sections:
        problems.append("no %runscript or %startscript: nothing to execute on `apptainer run`")

    return LevelResult(
        RecipeLevel.SYNTAX, passed=not problems, detail="; ".join(problems) if problems else "ok"
    )


# --------------------------------------------------------------------------
# L2 - buildable: does `apptainer build` succeed?
# --------------------------------------------------------------------------
def check_buildable(
    recipe: str, workdir: str, timeout: int = 180
) -> tuple[LevelResult, str | None]:
    """Returns (result, sif_path). `sif_path` is None whenever the build did
    not produce a usable image: callers must not try to run or test it."""
    if not apptainer_available():
        return (
            LevelResult(
                RecipeLevel.BUILDABLE, False, skipped=True,
                detail="level skipped (NOT counted as passed): apptainer not available",
            ),
            None,
        )

    def_path = Path(workdir) / "recipe.def"
    def_path.write_text(recipe, encoding="utf-8")
    sif_path = Path(workdir) / "image.sif"
    try:
        cmd = [_apptainer_bin(), "build"]
        if _unprivileged():
            cmd.append("--fakeroot")   # root only inside the namespace, not on the host
        cmd += [str(sif_path), str(def_path)]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return LevelResult(RecipeLevel.BUILDABLE, False, f"build timed out after {timeout}s"), None

    if proc.returncode != 0 or not sif_path.exists():
        # Keep the tail, not only the final line. apptainer signs off with a summary that
        # names the section that failed but never the command that failed inside it; that
        # sits in the lines just above. Keeping one line cost a CI run: it reported
        # "while running %post section: exit status 1" for a %post holding a single echo,
        # which says nothing about why. `--out` receives all of this; the terminal shows
        # its head, where the failing command reports itself ahead of the summary.
        lines = [ln for ln in (proc.stderr or proc.stdout).strip().splitlines() if ln.strip()]
        detail = "\n".join(lines[-12:])[:1500] if lines else "build failed, no output"
        return LevelResult(RecipeLevel.BUILDABLE, False, detail), None

    return LevelResult(RecipeLevel.BUILDABLE, True, "ok"), str(sif_path)


# --------------------------------------------------------------------------
# L3 - functional: does the built container run and produce the expected output?
# --------------------------------------------------------------------------
def check_recipe_functional(sif_path: str, task: RecipeTask, timeout: int = 60) -> LevelResult:
    try:
        cmd = [_apptainer_bin(), "run"]
        if _unprivileged():
            cmd.append("--userns")
        cmd.append(sif_path)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return LevelResult(RecipeLevel.FUNCTIONAL, False, f"run timed out after {timeout}s")

    if proc.returncode != 0:
        return LevelResult(
            RecipeLevel.FUNCTIONAL, False,
            f"exit code {proc.returncode}: {proc.stderr.strip()[:200]}",
        )

    combined = proc.stdout + proc.stderr
    missing = [s for s in task.expects_in_body if s not in combined]
    if missing:
        return LevelResult(RecipeLevel.FUNCTIONAL, False, f"expected output not found: {missing}")

    return LevelResult(RecipeLevel.FUNCTIONAL, True, "exit 0, expected output present")


# --------------------------------------------------------------------------
# L4a - resource_fit: does the recipe match the header/sections asked for?
# --------------------------------------------------------------------------
def check_recipe_resource_fit(recipe: str, task: RecipeTask) -> LevelResult:
    header = parse_header(recipe)
    sections = list_sections(recipe)
    c = task.constraints
    problems: list[str] = []

    for section in task.required_sections:
        name = section.lstrip("%").lower()
        if name not in sections:
            problems.append(f"required section missing: %{name}")

    if "bootstrap" in c and header.get("bootstrap", "").lower() != c["bootstrap"].lower():
        problems.append(
            f"Bootstrap expected {c['bootstrap']!r}, found {header.get('bootstrap')!r}"
        )

    if "from_contains" in c and c["from_contains"] not in header.get("from", ""):
        problems.append(
            f"From expected to contain {c['from_contains']!r}, found {header.get('from')!r}"
        )

    return LevelResult(
        RecipeLevel.RESOURCE_FIT, passed=not problems,
        detail="; ".join(problems) if problems else "ok",
    )


# --------------------------------------------------------------------------
# L4b - safety
# --------------------------------------------------------------------------
def check_recipe_safety(recipe: str) -> LevelResult:
    hits = [why for pat, why in DANGEROUS if pat.search(recipe)]
    return LevelResult(
        RecipeLevel.SAFETY, passed=not hits,
        detail="; ".join(hits) if hits else "no dangerous pattern",
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def verify_recipe(
    recipe: str, task: RecipeTask, run_functional: bool = True
) -> RecipeVerificationResult:
    syntax_result = check_recipe_syntax(recipe)
    resource_result = check_recipe_resource_fit(recipe, task)
    safety_result = check_recipe_safety(recipe)

    res = RecipeVerificationResult(task_id=task.id, recipe=recipe)
    res.levels.append(syntax_result)

    # Never build (let alone run) an unsafe or malformed recipe: same
    # discipline as check_safety gating check_functional in verifier.py.
    if safety_result.passed and syntax_result.passed:
        workdir = tempfile.mkdtemp(prefix="anvil_recipe_")
        try:
            build_result, sif_path = check_buildable(recipe, workdir)
            res.levels.append(build_result)
            if run_functional and sif_path:
                res.levels.append(check_recipe_functional(sif_path, task))
            else:
                res.levels.append(
                    LevelResult(
                        RecipeLevel.FUNCTIONAL, False, skipped=True,
                        detail="not run (build failed or functional disabled)",
                    )
                )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    else:
        res.levels.append(
            LevelResult(
                RecipeLevel.BUILDABLE, False, skipped=True,
                detail="not built (invalid syntax or unsafe recipe)",
            )
        )
        res.levels.append(
            LevelResult(
                RecipeLevel.FUNCTIONAL, False, skipped=True,
                detail="not run (invalid syntax or unsafe recipe)",
            )
        )

    res.levels.append(resource_result)
    res.levels.append(safety_result)
    return res

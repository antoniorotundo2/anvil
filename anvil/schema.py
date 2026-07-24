"""Benchmark task schema and verification results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Level(str, Enum):
    """Verification levels, weakest to strongest (see docs/REFERENCE_CLUSTER.md)."""

    SYNTAX = "syntax"                   # L1: is the script syntactically valid?
    SUBMITTABILITY = "submittability"   # L2: would SLURM accept it?
    FUNCTIONAL = "functional"           # L3: does it run and exit 0?
    RESOURCE_FIT = "resource_fit"       # L4a: does it request what was asked?
    SAFETY = "safety"                   # L4b: does it contain dangerous commands?


@dataclass
class Task:
    """A benchmark task.

    `constraints` is deliberately partial: every key present is checked, absent
    keys are ignored. This keeps tasks cheap to author.
    """

    id: str
    prompt: str                       # the natural-language specification
    constraints: dict[str, Any] = field(default_factory=dict)
    required_directives: list[str] = field(default_factory=list)
    # substrings expected in the script's combined output (functional check)
    expects_in_body: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def load_jsonl(path: str | Path) -> list[Task]:
        tasks: list[Task] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                tasks.append(Task(**json.loads(line)))
        return tasks


@dataclass
class RepairTask:
    """A T2 task: repair a broken script back to correctness.

    `base_task_id` points at the T1 task whose prompt, constraints and
    verifier apply unchanged. Repair is deliberately not a softer notion of
    correctness: a repaired script is graded by the exact same verifier that
    grades a from-scratch solution to `base_task_id`.
    """

    id: str
    base_task_id: str
    fault_category: str
    fault_detail: str
    broken_script: str

    @staticmethod
    def load_jsonl(path: str | Path) -> list[RepairTask]:
        items: list[RepairTask] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                items.append(RepairTask(**json.loads(line)))
        return items


class RecipeLevel(str, Enum):
    """Verification levels for T3 (Apptainer recipes), mirroring Level's shape
    for a different artifact: a `.def` recipe, not a SLURM script. There is no
    scheduler to submit to, so `buildable` (does `apptainer build` succeed)
    plays the role `submittability` plays for Level."""

    SYNTAX = "syntax"                # L1: a minimally well-formed recipe
    BUILDABLE = "buildable"          # L2: does `apptainer build` succeed?
    FUNCTIONAL = "functional"        # L3: does it run and produce the expected output?
    RESOURCE_FIT = "resource_fit"    # L4a: does it match the header/sections asked for?
    SAFETY = "safety"                # L4b: does it contain dangerous commands?


@dataclass
class RecipeTask:
    """A T3 task: write an Apptainer definition file (`.def`).

    Mirrors Task's shape. `constraints` supports "bootstrap" (exact match on
    the Bootstrap: header) and "from_contains" (substring match on From:).
    """

    id: str
    prompt: str
    constraints: dict[str, Any] = field(default_factory=dict)
    required_sections: list[str] = field(default_factory=list)
    expects_in_body: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def load_jsonl(path: str | Path) -> list[RecipeTask]:
        tasks: list[RecipeTask] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                tasks.append(RecipeTask(**json.loads(line)))
        return tasks


@dataclass
class LevelResult:
    level: Level
    passed: bool
    detail: str = ""
    skipped: bool = False   # e.g. L2 when no working scheduler is reachable

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


@dataclass
class VerificationResult:
    task_id: str
    script: str
    levels: list[LevelResult] = field(default_factory=list)

    def get(self, level: Level) -> LevelResult | None:
        return next((lr for lr in self.levels if lr.level is level), None)

    def passed(self, level: Level) -> bool:
        """A skipped level never counts as passed."""
        r = self.get(level)
        return bool(r and r.passed and not r.skipped)

    @property
    def all_passed(self) -> bool:
        return all(lr.passed or lr.skipped for lr in self.levels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "script": self.script,
            "levels": [lr.to_dict() for lr in self.levels],
            "all_passed": self.all_passed,
        }


@dataclass
class RecipeVerificationResult:
    """Same shape as VerificationResult, for a RecipeLevel/RecipeTask instead
    of a Level/Task: an Apptainer recipe is not a "script", so the field is
    named accordingly."""

    task_id: str
    recipe: str
    levels: list[LevelResult] = field(default_factory=list)

    def get(self, level: RecipeLevel) -> LevelResult | None:
        return next((lr for lr in self.levels if lr.level is level), None)

    def passed(self, level: RecipeLevel) -> bool:
        r = self.get(level)
        return bool(r and r.passed and not r.skipped)

    @property
    def all_passed(self) -> bool:
        return all(lr.passed or lr.skipped for lr in self.levels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "recipe": self.recipe,
            "levels": [lr.to_dict() for lr in self.levels],
            "all_passed": self.all_passed,
        }

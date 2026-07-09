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

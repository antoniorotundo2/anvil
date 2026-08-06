"""Does this script obey the rules of the cluster it is about to be submitted to?

The five verification levels answer "is this artifact correct against a task", which is
the benchmark's question and belongs to whoever is measuring a model. A site has a
different one: an artifact arrives, nobody wrote a task for it, and the question is
whether it may be submitted at all. Between the two sits most of what a submit filter
does, and almost all of the machinery for it already existed here.

The difference from `check_resource_fit` is the direction of the comparison, and it is
worth stating because the two look alike and mean opposite things. A task says the script
must request *at least* what was asked for: too little is the failure, and asking for more
than the spec is somebody else's problem. A policy says the script must request *at most*
what the site allows: too much is the failure, and asking for less is fine. The same
`--mem` value can satisfy one and violate the other.

A policy is a JSON object, every field optional, and an absent field is not a rule:

    {
      "name": "reference cluster",
      "max_time_minutes": 1440,
      "max_mem_mb": 131072,
      "max_nodes": 4,
      "max_ntasks": 16,
      "max_cpus_per_task": 8,
      "max_gpus": 4,
      "allowed_partitions": ["normal"],
      "required_directives": ["--time", "--account"],
      "forbidden_directives": ["--exclusive"]
    }

Silence is deliberate. A site that has no GPU rule should not have to write one, and a
policy file that had to mention every directive would be abandoned after the first
scheduler upgrade.

Effective requests, not written ones: a script with no `--nodes` still asks for one node,
so it is judged as asking for one. That is the same rule `check_resource_fit` applies and
for the same reason, and it is why a policy cannot be enforced by grepping the file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .parse import directive_value, parse_directives, parse_mem_to_mb, parse_time_to_minutes
from .resources import resolve


@dataclass
class Policy:
    name: str = "site policy"
    max_time_minutes: int | None = None
    max_mem_mb: float | None = None
    max_nodes: int | None = None
    max_ntasks: int | None = None
    max_cpus_per_task: int | None = None
    max_gpus: int | None = None
    allowed_partitions: list[str] = field(default_factory=list)
    required_directives: list[str] = field(default_factory=list)
    forbidden_directives: list[str] = field(default_factory=list)

    @staticmethod
    def load(path: str | Path) -> Policy:
        raw = json.loads(resolve(path).read_text(encoding="utf-8"))
        unknown = set(raw) - {f for f in Policy.__dataclass_fields__}
        if unknown:
            # Loudly, not silently: a misspelled `max_mem_gb` in a policy file would
            # otherwise read as a site that has no memory limit, which is the failure
            # mode a policy exists to prevent.
            raise ValueError(
                f"{path}: unknown policy fields {sorted(unknown)}. "
                f"Known: {sorted(Policy.__dataclass_fields__)}"
            )
        return Policy(**raw)


@dataclass
class PolicyResult:
    policy: str
    passed: bool
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"policy": self.policy, "passed": self.passed, "problems": self.problems}


def check_policy(script: str, policy: Policy) -> PolicyResult:
    d = parse_directives(script)
    problems: list[str] = []

    for directive in policy.required_directives:
        if directive not in d:
            problems.append(f"required by policy but missing: {directive}")

    for directive in policy.forbidden_directives:
        if directive in d:
            problems.append(f"forbidden by policy: {directive}")

    if policy.allowed_partitions:
        named = directive_value(d, "--partition", "-p")
        if named is None:
            problems.append(
                f"no partition named, and this site expects one of "
                f"{', '.join(policy.allowed_partitions)}"
            )
        elif named not in policy.allowed_partitions:
            problems.append(
                f"partition {named!r} is not one of {', '.join(policy.allowed_partitions)}"
            )

    def _int(*aliases: str) -> int | None:
        v = directive_value(d, *aliases)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            problems.append(f"non-integer value for {aliases[0]}: {v!r}")
            return None

    nodes = _int("--nodes", "-N")
    effective_nodes = 1 if nodes is None else nodes
    ntasks = _int("--ntasks", "-n")
    effective_ntasks = effective_nodes if ntasks is None else ntasks
    cpus = _int("--cpus-per-task", "-c")
    effective_cpus = 1 if cpus is None else cpus

    for value, ceiling, label in (
        (effective_nodes, policy.max_nodes, "nodes"),
        (effective_ntasks, policy.max_ntasks, "ntasks"),
        (effective_cpus, policy.max_cpus_per_task, "cpus-per-task"),
    ):
        if ceiling is not None and value > ceiling:
            problems.append(f"{label} {value} exceeds the site maximum {ceiling}")

    if policy.max_gpus is not None:
        raw = directive_value(d, "--gpus", "-G", "--gres")
        if raw:
            m = re.search(r"(\d+)\s*$", raw)
            if m and int(m.group(1)) > policy.max_gpus:
                problems.append(
                    f"gpus {m.group(1)} exceeds the site maximum {policy.max_gpus}"
                )

    if policy.max_time_minutes is not None:
        raw = directive_value(d, "--time", "-t")
        if raw is None:
            # An unset walltime is not a small one: SLURM applies the partition limit,
            # which the script cannot know and the site cannot infer from the file.
            problems.append(
                f"no --time, so the request cannot be shown to fit the site maximum "
                f"of {policy.max_time_minutes} minutes"
            )
        else:
            minutes = parse_time_to_minutes(raw)
            if minutes is None:
                problems.append(f"--time unparsable: {raw!r}")
            elif minutes > policy.max_time_minutes:
                problems.append(
                    f"--time {minutes}min exceeds the site maximum "
                    f"{policy.max_time_minutes}min"
                )

    if policy.max_mem_mb is not None:
        raw = directive_value(d, "--mem")
        if raw is not None:
            mb = parse_mem_to_mb(raw)
            if mb is None:
                problems.append(f"--mem unparsable: {raw!r}")
            elif mb > policy.max_mem_mb:
                problems.append(
                    f"--mem {mb:g}MB exceeds the site maximum {policy.max_mem_mb:g}MB"
                )

    return PolicyResult(policy=policy.name, passed=not problems, problems=problems)

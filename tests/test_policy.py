"""Site policy: the ceilings a submitted job may not exceed.

The comparison runs the opposite way from `check_resource_fit`, and the two are easy to
confuse: a task fails a script that asks for too little, a policy fails one that asks for
too much. Several of these tests exist to pin that direction, because a policy that
silently allowed everything would look exactly like a policy that was being obeyed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.policy import Policy, check_policy

ROOT = Path(__file__).resolve().parents[1]

COMPLIANT = (
    "#!/bin/bash\n"
    "#SBATCH --time=00:30:00\n"
    "#SBATCH --partition=normal\n"
    "#SBATCH --nodes=2\n"
    "#SBATCH --mem=4G\n"
    "echo ANVIL_OK\n"
)


def _policy(**kw) -> Policy:
    return Policy(**kw)


def test_the_shipped_example_loads_and_accepts_a_compliant_script():
    policy = Policy.load(ROOT / "policies" / "reference_cluster.json")
    assert check_policy(COMPLIANT, policy).passed


def test_every_ceiling_catches_a_script_that_exceeds_it():
    policy = Policy.load(ROOT / "policies" / "reference_cluster.json")
    greedy = (
        "#!/bin/bash\n"
        "#SBATCH --time=48:00:00\n"
        "#SBATCH --partition=gpu\n"
        "#SBATCH --nodes=9\n"
        "#SBATCH --mem=64G\n"
        "echo ANVIL_OK\n"
    )
    problems = " ".join(check_policy(greedy, policy).problems)
    assert "partition" in problems
    assert "nodes" in problems
    assert "--time" in problems
    assert "--mem" in problems


def test_asking_for_less_than_the_ceiling_is_not_a_violation():
    """The direction that separates a policy from a task spec."""
    policy = _policy(max_nodes=4, max_mem_mb=16384, max_time_minutes=1440)
    modest = "#!/bin/bash\n#SBATCH --time=00:01:00\n#SBATCH --nodes=1\n#SBATCH --mem=1M\necho hi\n"
    assert check_policy(modest, policy).passed


def test_an_empty_policy_forbids_nothing():
    """An absent field is not a rule. A site with no GPU limit should not have to write
    one, and a policy file that had to mention every directive would go stale."""
    assert check_policy("#!/bin/bash\n#SBATCH --nodes=999\necho hi\n", Policy()).passed


def test_ceilings_apply_to_the_effective_request_not_the_written_one():
    """A script with no --nodes still asks for one node, and one with no --ntasks asks for
    one task per node. Grepping the file would miss both."""
    policy = _policy(max_ntasks=1)
    implicit = "#!/bin/bash\n#SBATCH --nodes=4\necho hi\n"
    assert not check_policy(implicit, policy).passed


def test_an_unset_walltime_cannot_be_shown_to_fit():
    """SLURM applies the partition limit, which the file does not state, so the site
    cannot conclude the job fits. Treating silence as compliance is the hole a submit
    filter exists to close."""
    result = check_policy("#!/bin/bash\n#SBATCH --nodes=1\necho hi\n", _policy(max_time_minutes=60))
    assert not result.passed
    assert any("no --time" in p for p in result.problems)


def test_a_partition_allow_list_rejects_silence_as_well_as_a_wrong_name():
    policy = _policy(allowed_partitions=["normal"])
    assert not check_policy("#!/bin/bash\necho hi\n", policy).passed
    assert not check_policy("#!/bin/bash\n#SBATCH --partition=gpu\necho hi\n", policy).passed
    assert check_policy("#!/bin/bash\n#SBATCH --partition=normal\necho hi\n", policy).passed


def test_required_and_forbidden_directives():
    policy = _policy(required_directives=["--account"], forbidden_directives=["--exclusive"])
    assert not check_policy("#!/bin/bash\necho hi\n", policy).passed
    assert not check_policy(
        "#!/bin/bash\n#SBATCH --account=x\n#SBATCH --exclusive\necho hi\n", policy
    ).passed
    assert check_policy("#!/bin/bash\n#SBATCH --account=x\necho hi\n", policy).passed


def test_an_unknown_field_is_refused_rather_than_ignored(tmp_path):
    """A misspelled `max_mem_gb` would otherwise read as a site with no memory limit."""
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"max_mem_gb": 16}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown policy fields"):
        Policy.load(path)

# `python -m pytest` (not bare `pytest`): the module form puts the working
# directory on sys.path, so the suite runs without relying on an editable install.
#
# Prefer the project venv when it exists. Forgetting to activate it silently
# runs the system interpreter, which lacks torch/transformers - and `make
# generate` then fails halfway, leaving a stale generations file behind.
PYTHON     ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

.DEFAULT_GOAL := help

IMAGE      ?= anvil
APPTAINER_IMAGE ?= anvil:apptainer
TASKS      ?= tasks/t1_slurm.jsonl
REFERENCE  ?= tasks/t1_reference.jsonl
REPAIR_TASKS ?= tasks/t2_repair.jsonl
RECIPE_TASKS ?= tasks/t3_apptainer.jsonl
# Execution-sensitive set: one task whose memory need only the payload knows, and the
# repair variants induced from it. Kept apart from the shared T1/T2 files on purpose,
# so that adding it changes no digest and no published number.
EXEC_TASKS ?= tasks/t1_exec.jsonl
COREUTILS_TASKS ?= tasks/t1_coreutils.jsonl
EXEC_REFERENCE ?= tasks/t1_exec_reference.jsonl
EXEC_REPAIR_TASKS ?= tasks/t2_exec_repair.jsonl
MODEL      ?= Qwen/Qwen2.5-Coder-1.5B-Instruct
GENERATIONS ?= results/generations.jsonl
REPAIR_GENERATIONS ?= results/repair_generations.jsonl
RECIPE_GENERATIONS ?= results/recipe_generations.jsonl
VERIFY_OUT  ?= results/verification.json
REPAIR_VERIFY_OUT ?= results/repair_verification.json
RECIPE_VERIFY_OUT ?= results/recipe_verification.json
DOCKER_RUN  = docker run --rm -v "$(PWD)":/work -w /work $(IMAGE)
# Real submission needs slurmd, and slurmd needs to create its stepd scope under
# /sys/fs/cgroup, which a plain `docker run` mounts read-only. Nothing else in the
# project needs these two flags: `sbatch --test-only` never talks to a daemon.
# SCHED_IMAGE is not $(IMAGE): the default image accepts jobs and never runs them,
# because this Ubuntu's SLURM has no accounting plugin other than slurmdbd and refuses
# every job with Reason=InvalidAccount. `docker-build-sched` adds one, see
# docs/REFERENCE_CLUSTER.md.
SCHED_IMAGE ?= anvil:sched
DOCKER_RUN_SCHED = docker run --rm --privileged --cgroupns=host \
	-v "$(PWD)":/work -w /work $(SCHED_IMAGE)
# apptainer's unprivileged build/run needs these two beyond the default image:
# seccomp=unconfined for the build's user namespace, /dev/fuse to mount the
# built .sif at run time. See docker/Dockerfile for what was tried and ruled
# out (a plain run needs neither; --privileged works but grants much more).
# The image's apptainer has no setuid starter, so the user-namespace route is the only
# one available inside it, and it is now verified on GitHub runners and on WSL2 with
# byte-identical numbers. Default on: leaving it off made the default the one
# configuration neither environment can run. Pass 0 for a host apptainer that is setuid.
APPTAINER_UNPRIVILEGED ?= 1
DOCKER_RUN_APPTAINER = docker run --rm --security-opt seccomp=unconfined \
	--security-opt apparmor=unconfined --security-opt systempaths=unconfined \
	--device /dev/fuse \
	-e ANVIL_APPTAINER_UNPRIVILEGED=$(APPTAINER_UNPRIVILEGED) \
	-v "$(PWD)":/work -w /work $(APPTAINER_IMAGE)

.PHONY: help install install-models test lint doctor run verify guards guards-sbatch docker-guards-sbatch docker-build-sched \
        induce-exec docker-guards-enforcement docker-guards-coreutils paper \
        induce-t2 repair guards-t2 generate-repair \
        docker-build docker-test docker-run docker-verify docker-repair \
        docker-verify-repair generate \
        recipe guards-t3 docker-build-apptainer docker-recipe docker-guards-t3 \
        generate-recipe docker-verify-recipe clean

help:
	@echo "Anvil - executable benchmark of HPC operational artifacts"
	@echo ""
	@echo "  make install         install the package and dev tools"
	@echo "  make install-models  add torch/transformers (needed to generate)"
	@echo "  make test            run the test suite"
	@echo "  make lint            run ruff"
	@echo "  make doctor          report what this environment can verify"
	@echo ""
	@echo "  make docker-build    build the faithful verification image"
	@echo "  make docker-test     run the suite inside the container"
	@echo "  make docker-run      run the oracle inside the container"
	@echo ""
	@echo "  make guards          T1: oracle must score 1.0, broken 0.0"
	@echo "  make guards-sbatch   the same bracket with functional submitted for real"
	@echo "                       (needs a scheduler that runs jobs, not just accepts them)"
	@echo "  make docker-guards-sbatch  the same inside the container (builds $(SCHED_IMAGE),"
	@echo "                       which adds the accounting the scheduler needs to run jobs)"
	@echo "  make docker-guards-enforcement  a memory under-request must be OOM-killed:"
	@echo "                       the bracket for cgroup enforcement ($(EXEC_TASKS))"
	@echo "  make paper           regenerate the figures' data and build the preprint"
	@echo "  make docker-guards-coreutils  one task must be judged differently by GNU"
	@echo "                       coreutils and by uutils ($(COREUTILS_TASKS))"
	@echo "  make generate        generate scripts with MODEL (needs an accelerator)"
	@echo "  make docker-verify   verify those scripts against a real scheduler"
	@echo "                       -> $(VERIFY_OUT) (summary + environment + elapsed_s)"
	@echo ""
	@echo "  make induce-t2       (re)build $(REPAIR_TASKS) from the T1 references"
	@echo "  make guards-t2       T2: oracle repair 1.0, no-op repair 0.0"
	@echo "  make repair          diagnose-and-repair with MODEL against $(REPAIR_TASKS)"
	@echo ""
	@echo "  make recipe          T3: write an Apptainer recipe with MODEL and verify"
	@echo "  make guards-t3       T3 lenient check (no apptainer needed); full bracket"
	@echo "                       needs docker-guards-t3 (apptainer active)"
	@echo "  make docker-build-apptainer  build the apptainer-enabled image (opt-in, slower)"
	@echo "  make docker-guards-t3        T3: oracle 1.0, broken 0.0 strict, apptainer active"
	@echo ""
	@echo "  make clean           remove caches and build artifacts"
	@echo ""
	@echo "Variables: MODEL=$(MODEL)  TASKS=$(TASKS)"

# --- local (development) ----------------------------------------------------
.venv/bin/python:
	python3 -m venv .venv

install: .venv/bin/python
	.venv/bin/python -m pip install -e ".[dev]"

install-models: .venv/bin/python
	.venv/bin/python -m pip install -e ".[models]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check anvil tests scripts

doctor:
	$(PYTHON) -m anvil.cli doctor

run:
	$(PYTHON) -m anvil.cli run --model oracle --tasks $(TASKS) -v

# The oracle must pass every executable level; the broken model must fail.
# If the oracle drops below 1.0, the benchmark is broken - not the model.
guards:
	$(PYTHON) -m anvil.cli run --model oracle --tasks $(TASKS) --out /tmp/anvil_oracle.json
	$(PYTHON) -m anvil.cli run --model broken --tasks $(TASKS) -n 6 --out /tmp/anvil_broken.json
	@$(PYTHON) -c "import json,sys; \
o=json.load(open('/tmp/anvil_oracle.json'))['summary']; \
b=json.load(open('/tmp/anvil_broken.json'))['summary']; \
bad=[l for l in ('syntax','functional','resource_fit','safety') if o[l]['pass@1']!=1.0]; \
sys.exit('FAIL: oracle not at 1.0 on %s' % bad) if bad else None; \
sys.exit('FAIL: verifier promotes defective artifacts') if b['strict_all_levels']['pass@1']!=0.0 else None; \
sys.exit(\"FAIL: 'safety' guard never exercised\") if b['safety']['pass@1']==1.0 else None; \
print('Guards OK: oracle 1.0, broken 0.0 strict, safety exercised')"

# The same bracket with `functional` submitted for real. Not folded into `guards`: it
# needs a scheduler that runs jobs and not merely one that accepts them, so on most
# machines every functional sample comes back skipped and the check would pass while
# proving nothing. The vacuity assertion below is what stops that.
#
# The oracle assertion is deliberately not "functional == 1.0". A task whose own
# spec cannot be satisfied by a real scheduler is skipped, not passed, which drops
# the level's pass@1 while nothing is wrong: t1_dependency_chain points at a held
# placeholder job that never completes. What must hold is that no oracle sample
# *fails* the level, and that at least one actually ran.
guards-sbatch:
	$(PYTHON) -m anvil.cli run --model oracle --tasks $(TASKS) --executor sbatch -v \
		--out /tmp/anvil_oracle_sbatch.json
	$(PYTHON) -m anvil.cli run --model broken --tasks $(TASKS) --executor sbatch -n 6 \
		--out /tmp/anvil_broken_sbatch.json
	@$(PYTHON) -c "import json,sys; \
o=json.load(open('/tmp/anvil_oracle_sbatch.json')); \
b=json.load(open('/tmp/anvil_broken_sbatch.json'))['summary']; \
lv=[l for r in o['results'] for l in r['levels'] if l['level']=='functional']; \
bad=[l['detail'] for l in lv if not l['passed'] and not l['skipped']]; \
ran=[l for l in lv if l['passed']]; \
sys.exit('FAIL: the run did not use the sbatch executor') if o['environment']['functional_executor']!='sbatch' else None; \
sys.exit('FAIL: the oracle fails functional under real submission: %s' % bad[:2]) if bad else None; \
sys.exit('FAIL: every functional sample was skipped, the bracket proved nothing') if not ran else None; \
sys.exit('FAIL: verifier promotes defective artifacts') if b['strict_all_levels']['pass@1']!=0.0 else None; \
print('sbatch guards OK: %d functional samples ran for real, %d skipped, broken 0.0 strict' % (len(ran), len(lv)-len(ran)))"

# The same bracket inside the container, where the topology is the declared one. Runs on
# the accounting-enabled image, since the default one cannot execute a job at all. Where
# a scheduler accepts jobs without running them every sample comes back skipped, and the
# check below calls that a failure rather than passing vacuously.
docker-build-sched:
	docker build -t $(SCHED_IMAGE) --build-arg WITH_SLURMDBD=1 docker/

docker-guards-sbatch: docker-build-sched
	@mkdir -p results
	$(DOCKER_RUN_SCHED) python -m anvil.cli run --model oracle --tasks $(TASKS) \
		--executor sbatch -v --out results/sbatch_oracle.json
	$(DOCKER_RUN_SCHED) python -m anvil.cli run --model broken --tasks $(TASKS) \
		--executor sbatch -n 6 --out results/sbatch_broken.json
	@$(PYTHON) -c "import json,sys; \
o=json.load(open('results/sbatch_oracle.json')); \
b=json.load(open('results/sbatch_broken.json'))['summary']; \
lv=[l for r in o['results'] for l in r['levels'] if l['level']=='functional']; \
bad=[l['detail'] for l in lv if not l['passed'] and not l['skipped']]; \
ran=[l for l in lv if l['passed']]; \
sys.exit('FAIL: the run did not use the sbatch executor') if o['environment']['functional_executor']!='sbatch' else None; \
sys.exit('FAIL: the oracle fails functional under real submission: %s' % bad[:2]) if bad else None; \
sys.exit('FAIL: every functional sample was skipped, the bracket proved nothing') if not ran else None; \
sys.exit('FAIL: verifier promotes defective artifacts') if b['strict_all_levels']['pass@1']!=0.0 else None; \
print('sbatch guards OK in the container: %d functional samples ran for real, %d skipped' % (len(ran), len(lv)-len(ran)))"

# F8 is induced under real submission, because that is the only place it fails: the value
# it writes is well formed, within spec and accepted by the scheduler. Regenerating it
# needs the accounting image, hence the container.
induce-exec: docker-build-sched
	$(DOCKER_RUN_SCHED) python -m anvil.cli induce --tasks $(EXEC_TASKS) \
		--reference $(EXEC_REFERENCE) --out $(EXEC_REPAIR_TASKS) --executor sbatch

# The bracket for toolchain sensitivity. Every other T1 task returns the same verdict on
# GNU coreutils and on uutils, which is what the cross-distribution ablation reports and
# is only as strong as the tasks behind it. This one is built to return two, and the
# target fails if it ever stops doing so. It lives here rather than in `guards` because
# the ground truth is defined in the declared environment: on a BSD userland the
# reference solution counts bytes and pads its output, and neither is a defect in it.
docker-guards-coreutils:
	./scripts/coreutils_task_check.sh

# The bracket for cgroup enforcement. Its last assertion is the one that matters: the
# no-op repair of the F8 task must FAIL `functional`, and it can only fail there by being
# OOM-killed. Without enforcement that sample passes and this target says so, which is
# what keeps the guard from certifying an environment that enforces nothing.
docker-guards-enforcement: docker-build-sched
	@mkdir -p results
	$(DOCKER_RUN_SCHED) python -m anvil.cli run --model oracle --tasks $(EXEC_TASKS) \
		--executor sbatch -v --out results/exec_oracle.json
	$(DOCKER_RUN_SCHED) python -m anvil.cli repair --model oracle \
		--repair-tasks $(EXEC_REPAIR_TASKS) --tasks $(EXEC_TASKS) --executor sbatch \
		--out results/exec_repair_oracle.json
	$(DOCKER_RUN_SCHED) python -m anvil.cli repair --model broken \
		--repair-tasks $(EXEC_REPAIR_TASKS) --tasks $(EXEC_TASKS) --executor sbatch \
		--out results/exec_repair_broken.json
	@$(PYTHON) -c "import json,sys; \
t=json.load(open('results/exec_oracle.json'))['summary']; \
o=json.load(open('results/exec_repair_oracle.json'))['summary']; \
b=json.load(open('results/exec_repair_broken.json')); \
sys.exit('FAIL: the oracle does not solve the execution task') if t['strict_all_levels']['pass@1']!=1.0 else None; \
sys.exit('FAIL: the oracle repair does not fix every induced fault') if o['strict_all_levels']['pass@1']!=1.0 else None; \
sys.exit('FAIL: the no-op repair passes induced faults') if b['summary']['strict_all_levels']['pass@1']!=0.0 else None; \
f8=[r for r in b['results'] if r['task_id'].endswith('__F8')]; \
sys.exit('FAIL: no F8 sample in the set, enforcement is untested') if not f8 else None; \
lv=[l for r in f8 for l in r['levels'] if l['level']=='functional']; \
sys.exit('FAIL: the under-requesting script was not stopped, so nothing is enforced: %s' % [l['detail'][:80] for l in lv]) if any(l['passed'] or l['skipped'] for l in lv) else None; \
print('enforcement guards OK: the memory under-request is caught only by execution')"
	@echo "note: this target runs --privileged --cgroupns=host and leaves state in the"
	@echo "      host cgroup tree. A container started afterwards can fail to allocate"
	@echo "      ('Requested node configuration is not available'), which looks like a"
	@echo "      code regression and is not one. Run 'make docker-test' before this, as"
	@echo "      .github/workflows/ci.yml does, or restart the Docker engine after it."

# --- T2: diagnose-and-repair -------------------------------------------------
# In the container, like induce-exec. A variant is kept when the verifier refuses it, and
# submittability is one of the levels doing the refusing, so a host without the reference
# cluster would build a different task set. `anvil induce` refuses outright there now, and
# this target stops depending on whatever scheduler the developer's machine happens to run.
induce-t2: docker-build
	$(DOCKER_RUN) python -m anvil.cli induce --tasks $(TASKS) --reference $(REFERENCE) \
		--out $(REPAIR_TASKS)

repair:
	$(PYTHON) -m anvil.cli repair --model $(MODEL) --repair-tasks $(REPAIR_TASKS) --tasks $(TASKS) -v

# The oracle repair (ignores the diagnosis, returns the T1 canonical solution)
# must pass every T2 task; a no-op "repair" that returns the broken script
# unchanged must fail every one. If either fails, t2_repair.jsonl or the
# repair verifier is broken - not a model.
# Not `guards-t2: induce-t2` any more. Induction moved into the container, and depending on
# it made one of the four mandatory pre-commit checks need Docker to run at all. The guard's
# job is to bracket the committed task file, not to rebuild it; that the file is in sync with
# the inducers is a test (`test_t2_repair_file_is_in_sync_with_current_inducers`), which is
# where a mismatch belongs.
guards-t2:
	$(PYTHON) -m anvil.cli repair --model oracle --repair-tasks $(REPAIR_TASKS) --tasks $(TASKS) \
		--out /tmp/anvil_repair_oracle.json
	$(PYTHON) -m anvil.cli repair --model broken --repair-tasks $(REPAIR_TASKS) --tasks $(TASKS) \
		--out /tmp/anvil_repair_broken.json
	@$(PYTHON) -c "import json,sys; \
o=json.load(open('/tmp/anvil_repair_oracle.json'))['summary']; \
b=json.load(open('/tmp/anvil_repair_broken.json'))['summary']; \
bad=[l for l in ('syntax','functional','resource_fit','safety') if o[l]['pass@1']!=1.0]; \
sys.exit('FAIL: oracle repair not at 1.0 on %s' % bad) if bad else None; \
sys.exit('FAIL: no-op repair passes induced faults') if b['strict_all_levels']['pass@1']!=0.0 else None; \
print('T2 guards OK: oracle repair 1.0, no-op repair 0.0 strict')"

# --- T3: Apptainer recipes ---------------------------------------------------
recipe:
	$(PYTHON) -m anvil.cli recipe --model $(MODEL) --tasks $(RECIPE_TASKS) -v

# Unlike T1/T2, `buildable` and `functional` both need a real `apptainer`
# binary, which is rare outside the opt-in docker-build-apptainer image. This
# lenient check only asserts what syntax/resource_fit/safety can prove without
# it; the strict oracle-1.0/broken-0.0 bracket is `docker-guards-t3`.
guards-t3:
	$(PYTHON) -m anvil.cli recipe --model oracle --tasks $(RECIPE_TASKS) -v --out /tmp/anvil_recipe_oracle.json
	$(PYTHON) -m anvil.cli recipe --model broken --tasks $(RECIPE_TASKS) -n 5 -v --out /tmp/anvil_recipe_broken.json
	@$(PYTHON) -c "import json,sys; \
o=json.load(open('/tmp/anvil_recipe_oracle.json'))['summary']; \
b=json.load(open('/tmp/anvil_recipe_broken.json'))['summary']; \
bad=[l for l in ('syntax','resource_fit','safety') if o[l]['pass@1']!=1.0]; \
sys.exit('FAIL: oracle not at 1.0 on %s' % bad) if bad else None; \
sys.exit(\"FAIL: 'safety' guard never exercised\") if b['safety']['pass@1']==1.0 else None; \
sys.exit('FAIL: static checks promote defective recipes') if all(b[l]['pass@1']==1.0 for l in ('syntax','resource_fit','safety')) else None; \
print('T3 lenient guards OK (syntax/resource_fit/safety). Run docker-guards-t3 for the strict bracket.')"

# --- container (recommended) ------------------------------------------------
docker-build:
	docker build -t $(IMAGE) docker/

docker-test: docker-build
	$(DOCKER_RUN) python -m pytest -q

docker-run: docker-build
	$(DOCKER_RUN) python -m anvil.cli run --model oracle --tasks $(TASKS) -v

docker-repair: docker-build
	$(DOCKER_RUN) python -m anvil.cli repair --model oracle \
		--repair-tasks $(REPAIR_TASKS) --tasks $(TASKS) -v

# apptainer is opt-in (see docker/Dockerfile): most anvil work never touches
# it, and its PPA install adds real build time.
docker-build-apptainer:
	docker build -t $(APPTAINER_IMAGE) --build-arg WITH_APPTAINER=1 docker/

docker-recipe: docker-build-apptainer
	$(DOCKER_RUN_APPTAINER) python -m anvil.cli recipe --model oracle --tasks $(RECIPE_TASKS) -v

# The strict T3 bracket: with apptainer actually active, `buildable` and
# `functional` can be exercised, so the oracle/broken check is as strict as
# T1's and T2's. Not observed to work on Docker Desktop for Mac (nested
# linuxkit VM): `apptainer run` fails there even with these flags. Confirmed
# on Docker Desktop for Windows.
docker-apptainer-probe: docker-build-apptainer
	@$(DOCKER_RUN_APPTAINER) sh -c 'set -e; \
		echo "user            : $$(id -un) uid=$$(id -u)"; \
		apptainer --version; \
		if test -u /usr/libexec/apptainer/bin/starter-suid; then \
			echo "starter-suid    : present and setuid"; \
		else \
			echo "starter-suid    : absent or not setuid (unprivileged path only)"; \
		fi; \
		if unshare -U true 2>/dev/null; then \
			echo "user namespaces : can be created"; \
		else \
			echo "user namespaces : REFUSED ($$(unshare -U true 2>&1))"; \
		fi; \
		echo "max_user_ns     : $$(cat /proc/sys/user/max_user_namespaces 2>&1)"; \
		echo "subuid for root : $$(grep ^root: /etc/subuid 2>&1 || echo MISSING)"; \
		echo "newuidmap       : $$(command -v newuidmap || echo absent)"; \
		echo "conf mount home : $$(grep -E ^[[:space:]]*mount[[:space:]]+home /etc/apptainer/apptainer.conf 2>&1)"; \
		echo "build --no-mount: $$(apptainer build --help 2>&1 | grep -c no-mount) occurrences in help"; \
		echo "requested mode  : ANVIL_APPTAINER_UNPRIVILEGED=$$ANVIL_APPTAINER_UNPRIVILEGED"; \
		echo "apparmor profile: $$(cat /proc/self/attr/current 2>&1)"; \
		echo "userns restrict : $$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>&1)"; \
		echo "masked proc     : $$(grep -c " /proc/" /proc/mounts) submounts"; \
		grep -E "^CapEff|^CapBnd" /proc/self/status; \
		echo "HOME            : $$HOME"; \
		echo "passwd uid 0    : $$(getent passwd 0)"; \
		echo "--- minimal fakeroot build, mount lines from --debug ---"; \
		echo "Bootstrap: docker" > /tmp/p.def; \
		echo "From: alpine:latest" >> /tmp/p.def; \
		echo "%runscript" >> /tmp/p.def; \
		echo "    echo ANVIL_OK" >> /tmp/p.def; \
		apptainer --debug build --fakeroot /tmp/p.sif /tmp/p.def 2>&1 \
			| grep -iE "mount|fatal|error|fakeroot|namespace" | tail -30 || true'

docker-guards-t3: docker-build-apptainer
	@mkdir -p results
	$(DOCKER_RUN_APPTAINER) python -m anvil.cli recipe --model oracle --tasks $(RECIPE_TASKS) -v \
		--out results/anvil_recipe_oracle.json
	$(DOCKER_RUN_APPTAINER) python -m anvil.cli recipe --model broken --tasks $(RECIPE_TASKS) -n 5 -v \
		--out results/anvil_recipe_broken.json
	@$(PYTHON) -c "import json,sys; \
o=json.load(open('results/anvil_recipe_oracle.json'))['summary']; \
b=json.load(open('results/anvil_recipe_broken.json'))['summary']; \
bad=[l for l in ('syntax','buildable','functional','resource_fit','safety') if o[l]['pass@1']!=1.0]; \
sys.exit('FAIL: oracle not at 1.0 on %s' % bad) if bad else None; \
sys.exit('FAIL: verifier promotes defective recipes') if b['strict_all_levels']['pass@1']!=0.0 else None; \
sys.exit(\"FAIL: 'safety' guard never exercised\") if b['safety']['pass@1']==1.0 else None; \
print('T3 guards OK: oracle 1.0, broken 0.0 strict, safety exercised')"

# --- generate here, verify there --------------------------------------------
generate:
	@echo "Using interpreter: $(PYTHON)"
	@$(PYTHON) -c "import transformers, torch" 2>/dev/null || { \
	  echo ""; \
	  echo "ERROR: torch/transformers not available to $(PYTHON)."; \
	  echo "       Activate the venv (source .venv/bin/activate) and run: make install-models"; \
	  exit 1; }
	@mkdir -p $(dir $(GENERATIONS))
	$(PYTHON) -m anvil.cli run --model $(MODEL) --tasks $(TASKS) \
		--save-generations $(GENERATIONS)

docker-verify: docker-build
	@test -f $(GENERATIONS) || { \
	  echo "ERROR: $(GENERATIONS) does not exist. Run: make generate"; exit 1; }
	$(DOCKER_RUN) python -m anvil.cli verify \
		--generations $(GENERATIONS) --tasks $(TASKS) -v --out $(VERIFY_OUT)

generate-repair:
	@echo "Using interpreter: $(PYTHON)"
	@$(PYTHON) -c "import transformers, torch" 2>/dev/null || { \
	  echo ""; \
	  echo "ERROR: torch/transformers not available to $(PYTHON)."; \
	  echo "       Activate the venv (source .venv/bin/activate) and run: make install-models"; \
	  exit 1; }
	@mkdir -p $(dir $(REPAIR_GENERATIONS))
	$(PYTHON) -m anvil.cli repair --model $(MODEL) --repair-tasks $(REPAIR_TASKS) --tasks $(TASKS) \
		--save-generations $(REPAIR_GENERATIONS)

docker-verify-repair: docker-build
	@test -f $(REPAIR_GENERATIONS) || { \
	  echo "ERROR: $(REPAIR_GENERATIONS) does not exist. Run: make generate-repair"; exit 1; }
	$(DOCKER_RUN) python -m anvil.cli verify-repair \
		--generations $(REPAIR_GENERATIONS) --repair-tasks $(REPAIR_TASKS) --tasks $(TASKS) \
		-v --out $(REPAIR_VERIFY_OUT)

generate-recipe:
	@echo "Using interpreter: $(PYTHON)"
	@$(PYTHON) -c "import transformers, torch" 2>/dev/null || { \
	  echo ""; \
	  echo "ERROR: torch/transformers not available to $(PYTHON)."; \
	  echo "       Activate the venv (source .venv/bin/activate) and run: make install-models"; \
	  exit 1; }
	@mkdir -p $(dir $(RECIPE_GENERATIONS))
	$(PYTHON) -m anvil.cli recipe --model $(MODEL) --tasks $(RECIPE_TASKS) \
		--save-generations $(RECIPE_GENERATIONS)

docker-verify-recipe: docker-build-apptainer
	@test -f $(RECIPE_GENERATIONS) || { \
	  echo "ERROR: $(RECIPE_GENERATIONS) does not exist. Run: make generate-recipe"; exit 1; }
	$(DOCKER_RUN_APPTAINER) python -m anvil.cli verify-recipe \
		--generations $(RECIPE_GENERATIONS) --tasks $(RECIPE_TASKS) -v --out $(RECIPE_VERIFY_OUT)

# The tables and the plotted data come from leaderboard/entries/, so the manuscript cannot
# quote a run that has since been re-imported. latexmk is not a dependency of this project:
# the target says so plainly rather than failing with a shell error nobody can read.
#
# SOURCE_DATE_EPOCH is the commit time of the manuscript's own sources, which does two
# things. The build becomes byte-reproducible, so `git status` after `make paper` answers
# whether the paper changed instead of always reporting a modified binary: two consecutive
# compiles of identical sources used to differ by 64 bytes of embedded timestamp. And the
# date on the title page becomes the date the manuscript last changed rather than the date
# somebody happened to rebuild it, which for a preprint is the more honest of the two.
paper:
	./scripts/paper_data.py
	@cd paper && if command -v tectonic >/dev/null 2>&1; then \
		SOURCE_DATE_EPOCH=$$(git log -1 --format=%ct -- anvil.tex anvil.bib data) \
		tectonic -X compile anvil.tex; \
	elif command -v latexmk >/dev/null 2>&1; then latexmk -pdf anvil.tex; \
	else \
		echo "no TeX engine found: install tectonic (a single binary that fetches what it"; \
		echo "needs) or a TeX distribution with latexmk."; \
		echo "The generated data in paper/data/ is up to date regardless."; exit 1; \
	fi

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

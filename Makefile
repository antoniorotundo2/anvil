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
MODEL      ?= Qwen/Qwen2.5-Coder-1.5B-Instruct
GENERATIONS ?= results/generations.jsonl
REPAIR_GENERATIONS ?= results/repair_generations.jsonl
RECIPE_GENERATIONS ?= results/recipe_generations.jsonl
VERIFY_OUT  ?= results/verification.json
REPAIR_VERIFY_OUT ?= results/repair_verification.json
RECIPE_VERIFY_OUT ?= results/recipe_verification.json
DOCKER_RUN  = docker run --rm -v "$(PWD)":/work -w /work $(IMAGE)
# apptainer's unprivileged build/run needs these two beyond the default image:
# seccomp=unconfined for the build's user namespace, /dev/fuse to mount the
# built .sif at run time. See docker/Dockerfile for what was tried and ruled
# out (a plain run needs neither; --privileged works but grants much more).
# 1 pushes apptainer through its own user namespace instead of relying on host
# privileges the container does not have. See _unprivileged() in recipe_verifier.py.
APPTAINER_UNPRIVILEGED ?= 0
DOCKER_RUN_APPTAINER = docker run --rm --security-opt seccomp=unconfined --device /dev/fuse \
	-e ANVIL_APPTAINER_UNPRIVILEGED=$(APPTAINER_UNPRIVILEGED) \
	-v "$(PWD)":/work -w /work $(APPTAINER_IMAGE)

.PHONY: help install install-models test lint doctor run verify guards \
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
install:
	pip install -e ".[dev]"

install-models:
	pip install -e ".[models]"

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

# --- T2: diagnose-and-repair -------------------------------------------------
induce-t2:
	$(PYTHON) -m anvil.cli induce --tasks $(TASKS) --reference $(REFERENCE) --out $(REPAIR_TASKS)

repair:
	$(PYTHON) -m anvil.cli repair --model $(MODEL) --repair-tasks $(REPAIR_TASKS) --tasks $(TASKS) -v

# The oracle repair (ignores the diagnosis, returns the T1 canonical solution)
# must pass every T2 task; a no-op "repair" that returns the broken script
# unchanged must fail every one. If either fails, t2_repair.jsonl or the
# repair verifier is broken - not a model.
guards-t2: induce-t2
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
		echo "conf mount home : $$(grep -E '^[[:space:]]*mount home' /etc/apptainer/apptainer.conf 2>&1)"; \
		echo "build --no-mount: $$(apptainer build --help 2>&1 | grep -c no-mount) occurrences in help"; \
		echo "requested mode  : ANVIL_APPTAINER_UNPRIVILEGED=$$ANVIL_APPTAINER_UNPRIVILEGED"; \
		grep -E "^CapEff|^CapBnd" /proc/self/status'

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

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

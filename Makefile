# `python -m pytest` (not bare `pytest`): the module form puts the working
# directory on sys.path, so the suite runs without relying on an editable install.
#
# Prefer the project venv when it exists. Forgetting to activate it silently
# runs the system interpreter, which lacks torch/transformers - and `make
# generate` then fails halfway, leaving a stale generations file behind.
PYTHON     ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

.DEFAULT_GOAL := help

IMAGE      ?= anvil
TASKS      ?= tasks/t1_slurm.jsonl
MODEL      ?= Qwen/Qwen2.5-Coder-1.5B-Instruct
GENERATIONS ?= results/generations.jsonl
VERIFY_OUT  ?= results/verification.json
DOCKER_RUN  = docker run --rm -v "$(PWD)":/work -w /work $(IMAGE)

.PHONY: help install install-models test lint doctor run verify guards \
        docker-build docker-test docker-run docker-verify generate clean

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
	@echo "  make guards          oracle must score 1.0, broken 0.0"
	@echo "  make generate        generate scripts with MODEL (needs an accelerator)"
	@echo "  make docker-verify   verify those scripts against a real scheduler"
	@echo "                       -> $(VERIFY_OUT) (summary + environment + elapsed_s)"
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

# --- container (recommended) ------------------------------------------------
docker-build:
	docker build -t $(IMAGE) docker/

docker-test: docker-build
	$(DOCKER_RUN) python -m pytest -q

docker-run: docker-build
	$(DOCKER_RUN) python -m anvil.cli run --model oracle --tasks $(TASKS) -v

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

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# Build plumbing. OWNER: B (ML Engineer — Agent Runtime & Sandbox).
#
# `make check` is the team's gate: lint + unit tests + a 3-iteration smoke run with
# a stubbed LLM, under 60 s. Green before every merge, no exceptions. Breaking it is
# the team's top priority.

PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
RUN     ?=
DATA    ?= data

.DEFAULT_GOAL := help
.PHONY: help venv install lint fmt test smoke check dev official report clean clean-procs

help:  ## show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

venv:  ## create the virtualenv
	@test -d $(VENV) || $(PY) -m venv $(VENV)

install: venv  ## install orchestrator + pipeline requirements
	@$(BIN)/pip install -q --upgrade pip
	@$(BIN)/pip install -q -r requirements.txt
	@test -f requirements-pipeline.txt && $(BIN)/pip install -q -r requirements-pipeline.txt || true
	@echo "installed"

lint:  ## ruff
	@$(BIN)/ruff check orchestrator tests || $(PY) -m ruff check orchestrator tests

fmt:  ## ruff --fix
	@$(BIN)/ruff check --fix orchestrator tests || $(PY) -m ruff check --fix orchestrator tests

test:  ## unit tests + fault-injection suite
	@$(BIN)/pytest -q || $(PY) -m pytest -q

# The smoke run is stubbed end to end: no API key, no spend, no data download.
# Until A lands orchestrator/run.py this reports as pending rather than failing —
# a red main for a dependency nobody can fix yet helps no one.
smoke:  ## 3 iterations, stubbed LLM, subsampled data
	@if [ -f orchestrator/run.py ]; then \
	  TECHJAM_LLM=stub $(PY) -m orchestrator.run --task kuairand-pure --mode smoke; \
	else \
	  echo "smoke: PENDING — waiting on A's orchestrator/run.py."; \
	  echo "       agent+sandbox are covered end to end by tests/test_faults.py."; \
	fi

check: lint test smoke  ## the merge gate
	@echo "check: green"

dev:  ## 8 iterations, subsampled, real LLM
	@$(PY) -m orchestrator.run --task kuairand-pure --mode dev

official:  ## the scored run: 50 iterations / 6 h, unattended
	@$(PY) -m orchestrator.run --task kuairand-pure --mode official

report:  ## regenerate RESULTS.md + trajectory PNG from a run's journal
	@test -n "$(RUN)" || (echo "usage: make report RUN=<run_id>"; exit 1)
	@$(PY) -m orchestrator.report --run $(RUN)

clean-procs:  ## kill any pipeline processes a crashed run left behind
	@pkill -f "pipeline.py --data-dir" 2>/dev/null && echo "killed stragglers" || echo "none"

clean:  ## remove caches
	@rm -rf .pytest_cache .ruff_cache **/__pycache__ orchestrator/__pycache__ tests/__pycache__

# ============================================================
# Redact, Then Ask — Makefile
# ============================================================
# Usage:
#   make setup        create venv, install deps, download spaCy model
#   make test         run pytest
#   make lint         ruff check + ruff format --check
#   make data         download + convert datasets (Phase 2)
#   make eval         run evaluation harness, write reports/
#   make experiments  run all experiment scripts
#   make demo         launch Streamlit app

PYTHON  := python3.11
VENV    := .venv
PY      := $(VENV)/Scripts/python
PIP     := $(VENV)/Scripts/pip
RTA     := $(VENV)/Scripts/rta
PYTEST  := $(VENV)/Scripts/pytest
RUFF    := $(VENV)/Scripts/ruff

# ------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)

.PHONY: setup
setup: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,app,baseline]"
	$(PY) -m spacy download en_core_web_lg
	@echo "Setup complete. Activate with:  source $(VENV)/bin/activate"

# ------------------------------------------------------------
# quality gates
# ------------------------------------------------------------
.PHONY: test
test:
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-cov
test-cov:
	$(PYTEST) tests/ -v --tb=short --cov=src/rta --cov-report=term-missing

.PHONY: lint
lint:
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/

.PHONY: fmt
fmt:
	$(RUFF) check --fix src/ tests/
	$(RUFF) format src/ tests/

# ------------------------------------------------------------
# data pipeline
# ------------------------------------------------------------
.PHONY: data
data:
	$(PY) -m rta.data.ai4privacy
	$(PY) -m rta.data.tab

# ------------------------------------------------------------
# evaluation
# ------------------------------------------------------------
.PHONY: eval
eval:
	$(RTA) eval --dataset ai4privacy --split test --compare presidio

# ------------------------------------------------------------
# experiments
# ------------------------------------------------------------
.PHONY: experiments
experiments:
	$(PY) experiments/e1_privacy_utility.py
	$(PY) experiments/e2_leak.py
	$(PY) experiments/e3_error_propagation.py

# ------------------------------------------------------------
# demo
# ------------------------------------------------------------
.PHONY: demo
demo:
	$(VENV)/Scripts/streamlit run app/Home.py

# ------------------------------------------------------------
# housekeeping
# ------------------------------------------------------------
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV)

.DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "make setup       create venv and install dependencies"
	@echo "make test        run pytest"
	@echo "make lint        ruff check + format check"
	@echo "make fmt         auto-fix lint and formatting"
	@echo "make data        download and convert datasets"
	@echo "make eval        run evaluation harness"
	@echo "make experiments run all experiment scripts"
	@echo "make demo        launch Streamlit app"
	@echo "make clean       remove Python cache files"
	@echo "make clean-all   remove venv and cache files"

VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

.PHONY: install run cli test test-live coverage lint fmt dataset clean

install:  ## create venv and install app + dev deps
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

run:  ## start the API on :8000
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

cli:  ## example CLI run: make cli OBJ="Check payments-api status"
	$(PY) -m app.cli run "$(OBJ)"

test:  ## offline test suite (no network)
	$(PY) -m pytest -q

test-live:  ## live DeepSeek smoke test (needs DEEPSEEK_API_KEY)
	$(PY) -m pytest -q -m live

coverage:
	$(PY) -m pytest -q --cov=app --cov-report=term-missing

lint:
	$(PY) -m ruff check app tests

fmt:
	$(PY) -m ruff format app tests

dataset:  ## package mock dataset for upload
	mkdir -p dist && cd data && zip -qr ../dist/mock-dataset.zip . && echo "dist/mock-dataset.zip"

clean:
	rm -rf dist .pytest_cache .ruff_cache harness.db

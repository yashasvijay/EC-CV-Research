PYTHON ?= python

.PHONY: install test lint typecheck check
install:
	$(PYTHON) -m pip install -e '.[dev]'
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m mypy src
typecheck:
	$(PYTHON) -m mypy src
check: lint test

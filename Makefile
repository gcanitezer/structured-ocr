.PHONY: test lint format dev-install typecheck check

SHELL = /bin/bash
PYTHON = python3
SRC = src
TESTS = tests

test:
	$(PYTHON) -m pytest $(TESTS) -v

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

format:
	$(PYTHON) -m ruff format $(SRC) $(TESTS)

dev-install:
	$(PYTHON) -m pip install -e ".[train,dev]"

typecheck:
	$(PYTHON) -m mypy $(SRC)

check: lint typecheck test
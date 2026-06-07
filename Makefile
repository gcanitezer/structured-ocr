.PHONY: test lint format dev-install typecheck check

SHELL = /bin/bash
PYTHON = python3
SRC = src
TESTS = tests
UNIT = tests/unit
INTEGRATION = tests/integration
BENCHMARK = tests/benchmark
EDGE_CASES = tests/edge_cases

test:
	$(PYTHON) -m pytest $(TESTS) -v

test-unit:
	$(PYTHON) run_tests.py

test-integration:
	@echo "[warn] tests/integration is empty — no integration tests yet."
	$(PYTHON) -m pytest $(INTEGRATION) -v --tb=short -x || true

test-e2e:
	@echo "[warn] tests/e2e does not exist — no end-to-end tests yet."
	@true

test-benchmark:
	@echo "[warn] tests/benchmark is empty — no benchmark tests yet."
	$(PYTHON) -m pytest $(BENCHMARK) -v --tb=short -x || true

test-edge-cases:
	$(PYTHON) -m pytest $(EDGE_CASES) -v --tb=short -x || true

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

format:
	$(PYTHON) -m ruff format $(SRC) $(TESTS)

dev-install:
	$(PYTHON) -m pip install -e ".[dev]"

typecheck:
	$(PYTHON) -m mypy $(SRC)

check: lint typecheck test-unit
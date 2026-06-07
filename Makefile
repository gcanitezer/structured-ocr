.PHONY: help dev-install install train test lint format clean run-api run-cli

PYTHON ?= python3
SRC := src
TESTS := tests
PKG := structured-ocr

help:
	@echo "Common targets:"
	@echo "  dev-install    - install the package with dev + train extras"
	@echo "  install        - install the package (runtime deps only)"
	@echo "  test           - run the test suite"
	@echo "  lint           - run linters (ruff/black --check)"
	@echo "  format         - auto-format with black + isort"
	@echo "  clean          - remove build artifacts"
	@echo "  train-sft      - run SFT with configs/training_sft.yaml"
	@echo "  train-grpo     - run GRPO with configs/training_grpo.yaml"
	@echo "  run-cli        - run the latexocr CLI"
	@echo "  run-api        - run the FastAPI server"

install:
	$(PYTHON) -m pip install -e .

dev-install:
	$(PYTHON) -m pip install -e ".[dev,train]"

test:
	PYTHONPATH=$(SRC) $(PYTHON) -m pytest $(TESTS) -v

lint:
	$(PYTHON) -m black --check $(SRC) $(TESTS) || true
	$(PYTHON) -m isort --check-only $(SRC) $(TESTS) || true

format:
	$(PYTHON) -m black $(SRC) $(TESTS) || true
	$(PYTHON) -m isort $(SRC) $(TESTS) || true

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type d -name .pytest_cache -exec rm -rf {} + || true

train-sft:
	PYTHONPATH=$(SRC) $(PYTHON) scripts/train_sft.py --config configs/training_sft.yaml

train-grpo:
	PYTHONPATH=$(SRC) $(PYTHON) scripts/train_grpo.py --config configs/training_grpo.yaml

run-cli:
	PYTHONPATH=$(SRC) $(PYTHON) -m structured_ocr.cli

run-api:
	$(PYTHON) -m uvicorn structured_ocr.api:app --reload

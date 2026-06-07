# Structured OCR

LaTeX OCR System for Full Document Reconstruction

Extract LaTeX from images and PDFs, verify compilability, train custom
OCR models via supervised fine-tuning or GRPO reinforcement learning,
and evaluate output against ground-truth references.

## Table of Contents

- [Architecture](#architecture)
- [Quick Start / Installation](#quick-start--installation)
- [CLI Reference](#cli-reference)
- [REST API Reference](#rest-api-reference)
- [Inference Configuration](#inference-configuration)
- [Training Pipeline](#training-pipeline)
- [Evaluation Framework](#evaluation-framework)
- [Corpus Generation](#corpus-generation)
- [Verification](#verification)
- [Development](#development)
- [Docker & Docker Compose](#docker--docker-compose)
- [Project Structure](#project-structure)

## Architecture

The system is organised into these modules under `src/structured_ocr/`:

| Module | Purpose |
|---|---|
| `inference/` | OCR inference engine with pluggable backends (Pix2Text, HuggingFace) |
| `api/` | FastAPI REST API exposing inference, verification, evaluation, and training endpoints |
| `training/` | SFT and GRPO/RLVR training pipeline with nine reward functions |
| `eval/` | TexOCR-Bench evaluation framework with multiple metrics |
| `corpus/` | LaTeX corpus generator (textbooks, newspapers, leaflets) |
| `verification/` | LaTeX compilation verification (pdflatex/xelatex/lualatex) |
| `data/` | Core data types (OCRResult, DocumentStructure, BoundingBox, etc.) |
| `evaluation/` | Formula-level and structural metrics |

## Quick Start / Installation

```bash
# Base install (CLI + inference)
pip install -e .

# Enable REST API
pip install -e ".[api]"

# Enable evaluation framework
pip install -e ".[eval]"

# Enable full training pipeline (TRL, PEFT, bitsandbytes, accelerate)
pip install -e ".[train]"

# All extras
pip install -e ".[eval,api,train]"
```

**Dependencies by extra:**

| Extra | Packages |
|---|---|
| `api` | `python-multipart`, `uvicorn` |
| `eval` | `pdf2image`, `scipy` |
| `train` | `trl`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `deepspeed` |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy` |

## CLI Reference

The CLI is accessed via the `latexocr` command.

### `latexocr infer`

Perform OCR on an image and extract LaTeX.

```bash
latexocr infer image.png -o output.tex
latexocr infer image.png --backend huggingface --device cuda --model Qwen/Qwen2.5-VL-7B-Instruct
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `IMAGE_PATH` | path (positional) | required | Input image file |
| `--output, -o` | path | None | Output LaTeX file path |
| `--backend, -b` | `pix2text` \| `huggingface` | None | Inference backend |
| `--device, -d` | str | None | Device (`cpu`, `cuda`, `auto`) |
| `--model, -m` | str | None | Override model name |

### `latexocr infer pdf`

Extract and OCR all pages from a PDF document.

```bash
latexocr infer pdf document.pdf -o results.json
latexocr infer pdf document.pdf --dpi 200 --backend pix2text
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `PDF_PATH` | path (positional) | required | Input PDF file |
| `--output, -o` | path | None | Output JSON file path |
| `--dpi` | int | 150 | PDF rendering DPI |
| `--backend, -b` | `pix2text` \| `huggingface` | None | Inference backend |
| `--device, -d` | str | None | Device (`cpu`, `cuda`, `auto`) |
| `--model, -m` | str | None | Override model name |

### `latexocr verify`

Verify LaTeX compilability and structure of a `.tex` file.

```bash
latexocr verify output.tex
latexocr verify output.tex --engine xelatex --timeout 60 --passes 3 --report report.json
latexocr verify output.tex --reference ground_truth.tex
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `LATEX_FILE` | path (positional) | required | `.tex` file to verify |
| `--reference, -r` | path | None | Reference `.tex` file for similarity checks |
| `--engine` | `pdflatex` \| `xelatex` \| `lualatex` | `pdflatex` | LaTeX engine |
| `--timeout` | float | `30.0` | Per-pass timeout (seconds) |
| `--passes` | int | `2` | Number of compilation passes |
| `--report` | path | None | Path for JSON verification report |

### `latexocr train sft`

Run supervised fine-tuning.

```bash
latexocr train sft --config configs/training_sft.yaml
latexocr train sft --model-name Qwen/Qwen2-VL-2B-Instruct --train-data ./data/train.jsonl --epochs 5 --lora
```

**Options:**

| Flag | Type | Description |
|---|---|---|
| `--config, -c` | path | Training YAML/JSON config |
| `--model-name` | str | Base model name or path |
| `--train-data` | path | Training data file |
| `--eval-data` | path | Evaluation data file |
| `--output-dir` | path | Output directory for checkpoints |
| `--epochs` | int | Override number of training epochs |
| `--batch-size` | int | Override per-device batch size |
| `--grad-accum` | int | Override gradient accumulation steps |
| `--learning-rate` | float | Override learning rate |
| `--lora / --no-lora` | flag | Enable LoRA (default: disabled) |
| `--qlora / --no-qlora` | flag | Enable QLoRA (default: disabled) |
| `--deepspeed` | path | Path to DeepSpeed JSON config |
| `--fsdp / --no-fsdp` | flag | Enable FSDP (default: disabled) |
| `--log-level` | str | Logging level (default: `INFO`) |

### `latexocr train grpo`

Run GRPO / RLVR reinforcement learning with nine reward functions.

```bash
latexocr train grpo --config configs/training_grpo.yaml
latexocr train grpo --model-name ./outputs/sft --rlvr-num-generations 8 --rlvr-kl-coef 0.04
```

**Options (all from `train sft` plus):**

| Flag | Type | Description |
|---|---|---|
| `--rlvr-num-generations` | int | Generations per prompt |
| `--rlvr-kl-coef` | float | KL penalty coefficient |
| `--rlvr-clip-ratio` | float | GRPO clip ratio |
| `--rlvr-max-new-tokens` | int | Max tokens per completion |
| `--rlvr-temperature` | float | Sampling temperature |
| `--rlvr-max-steps` | int | Hard cap on GRPO optimizer steps |

### `latexocr eval evaluate`

Run evaluation on OCR predictions against references (TexOCR-Bench).

```bash
latexocr eval evaluate --predictions preds.json --references refs.json
latexocr eval evaluate -p preds.json -r refs.json -i images.json --baseline baselines.json --model-name "MyModel"
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--predictions, -p` | path (required) | | JSON file: `sample_id -> latex` |
| `--references, -r` | path (required) | | JSON file: `sample_id -> latex` |
| `--images, -i` | path | None | JSON file: `sample_id -> image path` |
| `--output, -o` | path | `eval_report.json` | Output report path |
| `--model-name, -m` | str | `unknown` | Model name for the report |
| `--baseline, -b` | path | None | Baseline scores JSON for comparison |
| `--no-compilability` | flag | | Skip compilability checks |
| `--no-references` | flag | | Skip reference integrity checks |

### `latexocr eval check-compilable`

Quickly check if a LaTeX string is compilable.

```bash
latexocr eval check-compilable --latex "\documentclass{article}\begin{document}Hello\end{document}"
```

**Options:**

| Flag | Type | Description |
|---|---|---|
| `--latex, -l` | str (required) | LaTeX string to check |

### `latexocr eval init-baselines`

Initialize default baseline scores file (GPT-4V and olmOCR defaults).

```bash
latexocr eval init-baselines -o my_baselines.json
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output, -o` | path | `default_baselines.json` | Output path |

### `latexocr corpus textbooks`

Generate textbook LaTeX corpus.

```bash
latexocr corpus textbooks -o ./data/corpus -n 100 -s math
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output, -o` | path | `./data/corpus` | Output directory |
| `--count, -n` | int | `100` | Number of documents |
| `--subject, -s` | `math` \| `biology` \| `chemistry` | `math` | Textbook subject |

### `latexocr corpus newspapers`

Generate newspaper LaTeX corpus.

```bash
latexocr corpus newspapers -o ./data/corpus -n 50
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output, -o` | path | `./data/corpus` | Output directory |
| `--count, -n` | int | `50` | Number of documents |

### `latexocr corpus leaflets`

Generate leaflet/brochure LaTeX corpus.

```bash
latexocr corpus leaflets -o ./data/corpus -n 50
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output, -o` | path | `./data/corpus` | Output directory |
| `--count, -n` | int | `50` | Number of documents |

### `latexocr corpus index`

Index the generated corpus for HuggingFace format.

```bash
latexocr corpus index -o ./data/corpus_index.json -c ./data/corpus
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output, -o` | path | `./data/corpus_index.json` | Output index file |
| `--corpus-dir, -c` | path | `./data/corpus` | Corpus directory to index |

## REST API Reference

Start the API:

```bash
# With docker-compose (recommended)
docker-compose up

# Directly with uvicorn
uvicorn structured_ocr.api.app:create_app --host 0.0.0.0 --port 8000 --factory
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### `GET /health`

Health check endpoint.

**Response:**
```json
{"status": "ok", "version": "0.1.0"}
```

### `POST /ocr`

Run OCR on a single image.

**Request:** multipart form with:
- `file` (UploadFile) — the image file
- `backend` (str, optional) — `pix2text` or `huggingface`
- `device` (str, optional) — `cpu`, `cuda`, `auto`
- `max_new_tokens` (int, optional)

**Response:**
```json
{
  "latex": "\\frac{a}{b}",
  "confidence": 0.95,
  "processing_time_ms": 1234.5,
  "model_name": "Pix2Text",
  "detected_elements": {}
}
```

### `POST /ocr/batch`

Run OCR on multiple images.

**Request:** multipart form with multiple `file` fields + same query params as `/ocr`.

**Response:** Array of `OCRResponse` objects.

### `POST /verify`

Verify LaTeX document.

**Request:**
```json
{
  "latex": "\\documentclass{article}..."
}
```

**Response:**
```json
{
  "passed": true,
  "total_score": 0.85,
  "compilation": {
    "outcome": "success",
    "elapsed_seconds": 1.23,
    "log_summary": ""
  },
  "components": [
    {"name": "equation_accuracy", "passed": true, "score": 1.0, "details": "..."}
  ],
  "errors": []
}
```

### `POST /evaluate`

Evaluate predictions against references.

**Request:**
```json
{
  "predictions": {"sample1": "\\frac{a}{b}", ...},
  "references": {"sample1": "\\frac{a}{b}", ...},
  "model_name": "MyModel"
}
```

**Response:**
```json
{
  "model_name": "MyModel",
  "total_samples": 100,
  "avg_edit_distance": 0.05,
  "avg_similarity_ratio": 0.95,
  "avg_bleu": 0.88,
  "avg_section_f1": 0.92,
  "avg_table_f1": 0.85,
  "avg_equation_f1": 0.90,
  "avg_citation_f1": 0.87,
  "compilability_rate": 0.95,
  "avg_compilation_time": 2.1,
  "avg_image_similarity": 0.89,
  "avg_reference_integrity": 0.94
}
```

### `POST /train`

Queue a training job.

**Request:**
```json
{
  "config": {"model_name": "Qwen/Qwen2-VL-2B-Instruct", ...}
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

### `GET /train/status/{job_id}`

Check training job status.

**Response:**
```json
{
  "job_id": "550e8400-...",
  "status": "queued",
  "message": ""
}
```

## Inference Configuration

The `InferConfig` dataclass controls inference behaviour:

| Field | Type | Default | Description |
|---|---|---|---|
| `backend` | str | `"pix2text"` | `"pix2text"` or `"huggingface"` |
| `model_name` | str \| None | None | Model name/path (None = backend default) |
| `device` | str | `"auto"` | `"cpu"`, `"cuda"`, or `"auto"` |
| `max_new_tokens` | int | `2048` | Max tokens to generate (HF backend) |
| `temperature` | float | `0.1` | Sampling temperature (HF backend) |
| `batch_size` | int | `1` | Batch size for inference |
| `timeout` | int | `120` | Inference timeout in seconds |

**Backend details:**

- **Pix2TextBackend** (`backend="pix2text"`): Uses the `pix2text` library for layout detection, formula OCR, and table OCR. Default model: `"Pix2Text"`. Falls back to HuggingFace if pix2text is not installed.
- **HFBackend** (`backend="huggingface"`): Uses HuggingFace Transformers with a VLM (default: `"Qwen/Qwen2.5-VL-7B-Instruct"`). Supports `device`, `max_new_tokens`, `temperature`.

**PDF processing** converts each page to a PIL Image (default 150 DPI) and runs batch inference. The `extract_images_from_pdf` function uses `pdf2image` under the hood.

## Training Pipeline

The training pipeline has three modes specified in `TrainingMode`:

| Mode | Description |
|---|---|
| `sft` | Supervised fine-tuning only |
| `grpo` | GRPO / RLVR reinforcement learning only |
| `sft_then_grpo` | Run SFT, then continue with GRPO from the checkpoint |

### SFT Stage

Supervised fine-tuning on generated LaTeX corpora. Uses `transformers.Trainer` or TRL's `SFTTrainer` depending on availability, with support for:

- LoRA (`r=16`, `lora_alpha=32`, `lora_dropout=0.05`)
- QLoRA (4-bit NF4 quantization with double quant)
- DeepSpeed ZeRO-2 / ZeRO-3
- FSDP via accelerate

**Default hyperparameters:**

| Parameter | SFT Default |
|---|---|
| Model | `Qwen/Qwen2-VL-2B-Instruct` |
| Learning rate | `2e-4` |
| Epochs | `3` |
| Batch size | `2` |
| Gradient accumulation | `8` |
| Max sequence length | `4096` |
| Warmup ratio | `0.03` |

### GRPO / RLVR Stage

Group Relative Policy Optimization with nine reward functions. Typically run on an SFT checkpoint.

**Default hyperparameters:**

| Parameter | GRPO Default |
|---|---|
| Learning rate | `5e-6` |
| Epochs | `1` |
| Batch size | `1` |
| Gradient accumulation | `16` |

### Nine Reward Functions

| Reward Component | Default Weight | Description |
|---|---|---|
| `equation_accuracy` | `0.15` | Semantic match of extracted equations |
| `equation_syntax` | `0.15` | Well-formed equation environments |
| `table_structure` | `0.10` | Column/row count match for tabular |
| `section_hierarchy` | `0.10` | Section heading presence & ordering |
| `citation_label_integrity` | `0.10` | All `\cite` keys have matching `\label` |
| `cross_reference_validity` | `0.10` | All `\ref` / `\eqref` have targets |
| `compilation_success` | `0.20` | Compiles with pdflatex (score: 1.0 success, 0.5 compiler missing, 0.0 fail) |
| `visual_similarity` | `0.05` | Rendered image similarity (requires reference image) |
| `semantic_coherence` | `0.05` | Textual similarity via difflib |

Reward shaping uses `tanh(score * 3.0)` mapped to `[0, 1]`. Weights must sum to ~1.0.

### Distributed Training

```bash
# Single-node, multi-GPU SFT via torchrun
./scripts/distributed_launch.sh sft --config configs/training_sft.yaml

# Multi-node GRPO via accelerate FSDP
./scripts/distributed_launch.sh grpo --launcher accelerate \
    --accelerate-config configs/accelerate_fsdp.yaml \
    --config configs/training_grpo.yaml
```

### Configuration Files

All files are in `configs/`:

| File | Purpose |
|---|---|
| `training_sft.yaml` | SFT configuration (model, optimizer, LoRA, DeepSpeed) |
| `training_grpo.yaml` | GRPO configuration (lower LR, higher grad accum) |
| `training_sft_then_grpo.yaml` | Two-stage SFT -> GRPO pipeline |
| `deepspeed_zero2.json` | DeepSpeed ZeRO-2 preset |
| `deepspeed_zero3.json` | DeepSpeed ZeRO-3 preset |
| `accelerate_fsdp.yaml` | Accelerate FSDP launcher config |

## Evaluation Framework

The TexOCR-Bench evaluation framework measures OCR output quality across multiple dimensions.

### Metrics

| Metric | Description |
|---|---|
| **Edit distance** | Character-level Levenshtein distance between prediction and reference |
| **Similarity ratio** | `1 - edit_distance` (SequenceMatcher ratio) |
| **BLEU** | n-gram precision (unigram to 4-gram) with brevity penalty |
| **Structural F1** | Precision/recall/F1 for sections, tables, equations, citations |
| **Compilability rate** | Fraction of predictions that compile successfully |
| **Image similarity** | MSE-based similarity between rendered PDF and reference image |
| **Reference integrity** | Label/citation integrity score (undefined labels, missing bib entries) |

### Baseline Comparison

Built-in baselines for GPT-4V and olmOCR are defined in `BaselineScores`. Use `init-baselines` to generate the default file:

```bash
latexocr eval init-baselines -o baselines.json
```

Use `--baseline baselines.json` with `eval evaluate` to compare results.

### Report Output

The evaluation produces a JSON report with:
- Summary statistics (edit distance, BLEU, structural F1, compilability rate)
- Per-sample analysis (min/max/median, high-error samples, low-compilability samples)
- Baseline comparison showing whether your model beats GPT-4V and olmOCR score

## Corpus Generation

Generate synthetic LaTeX documents for training and evaluation.

### Templates

| Command | Template | Description |
|---|---|---|
| `corpus textbooks` | `TextbookTemplate` | Multi-page textbooks with sections, equations, tables (math/biology/chemistry) |
| `corpus newspapers` | `NewspaperTemplate` | Newspaper-style documents with columns |
| `corpus leaflets` | `LeafletTemplate` | Leaflet/brochure documents |

Each generated document includes:
- The `.tex` source file
- A compiled `.pdf`
- Per-page `.png` renders

Use `corpus index` to create a HuggingFace-compatible index from the generated corpus.

## Verification

The `LaTeXVerifier` orchestrates compilation and unit tests for LaTeX documents.

### Supported Engines

| Engine | Binary | Use Case |
|---|---|---|
| `pdflatex` | `pdflatex` | Standard LaTeX (default) |
| `xelatex` | `xelatex` | Unicode / OTF fonts |
| `lualatex` | `lualatex` | LuaTeX features |

The compiler (`LaTeXCompiler`) supports:
- Configurable number of passes (default: 2 for cross-reference resolution)
- Wall-clock timeout per pass
- Early termination when aux files stabilise
- Structured error/warning parsing from `.log` files
- Thread-safe operation

### Verification Result Components

The verifier produces a `VerificationResult` containing:

- `total_score`: weighted sum of all component scores (usable as a reward signal)
- `passed` / `pass_rate`: whether the document meets the fail threshold (default: 0.5)
- Per-component breakdown with scores, weights, and pass/fail
- Compilation details (outcome, engine, elapsed time, errors)

## Development

```bash
make dev-install    # pip install -e ".[dev]"
make test           # Run all tests
make test-unit      # Run unit tests (via run_tests.py)
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy
make check          # lint + typecheck + unit tests
```

## Docker & Docker Compose

### Docker

```bash
docker build -t structured-ocr .
docker run -it structured-ocr
```

The Dockerfile installs TeX Live (pdflatex, xelatex, lualatex) and all Python dependencies.

### Docker Compose

```bash
docker-compose up
```

Starts the API on port `8000` with:
- Automatic restart and health checks (`GET /health`)
- Volume mounts for `./data`, `./models`, `./output`
- Environment variable passthrough for `CUDA_VISIBLE_DEVICES`

## Project Structure

```
structured-ocr/
├── configs/                      # Training & distributed configs
│   ├── training_sft.yaml
│   ├── training_grpo.yaml
│   ├── training_sft_then_grpo.yaml
│   ├── deepspeed_zero2.json
│   ├── deepspeed_zero3.json
│   └── accelerate_fsdp.yaml
├── data/                         # Training/evaluation data
├── scripts/
│   ├── distributed_launch.sh     # Multi-GPU launcher
│   ├── train_sft.py
│   └── train_grpo.py
├── src/structured_ocr/
│   ├── cli.py                    # Click CLI entry point
│   ├── inference/                # OCR inference engine
│   │   ├── config.py             # InferConfig dataclass
│   │   ├── backend.py            # Abstract Backend base class
│   │   ├── pix2text_backend.py   # Pix2Text backend
│   │   ├── hf_backend.py         # HuggingFace VLM backend
│   │   ├── engine.py             # InferenceEngine orchestrator
│   │   └── pdf.py                # PDF batch processing
│   ├── api/                      # FastAPI REST API
│   │   ├── app.py                # create_app() factory
│   │   ├── dependencies.py       # FastAPI dependency injection
│   │   ├── routers/              # Endpoint routers
│   │   │   ├── ocr.py            # POST /ocr, /ocr/batch
│   │   │   ├── verify.py         # POST /verify
│   │   │   ├── evaluate.py       # POST /evaluate
│   │   │   └── train.py          # POST /train, GET /train/status
│   │   └── models/               # Pydantic schemas
│   │       ├── requests.py       # Request models
│   │       └── responses.py      # Response models
│   ├── training/                 # Training pipeline
│   │   ├── types.py              # TrainingConfig, LoRAConfig, RewardConfig
│   │   ├── dataset_utils.py      # Dataset preparation
│   │   ├── sft_trainer.py        # SFT trainer
│   │   ├── grpo_trainer.py       # GRPO/RLVR trainer
│   │   ├── reward_functions.py   # 9 reward functions
│   │   └── pipeline.py           # TrainingPipeline orchestrator
│   ├── eval/                     # Evaluation framework
│   │   ├── cli.py                # Standalone eval CLI
│   │   ├── benchmark.py          # BenchmarkRunner, BaselineScores
│   │   ├── metrics.py            # Edit distance, BLEU, structural F1
│   │   ├── compilability.py      # CompilabilityChecker
│   │   ├── references.py         # ReferenceIntegrityChecker
│   │   └── report.py             # Report generation
│   ├── corpus/                   # Corpus generation
│   │   ├── corpus_cli.py         # Corpus CLI commands
│   │   ├── generator.py          # CorpusGenerator
│   │   └── templates.py          # Textbook, Newspaper, Leaflet templates
│   ├── verification/             # LaTeX verification
│   │   ├── compiler.py           # LaTeXCompiler (pdflatex/xelatex/lualatex)
│   │   ├── verifier.py           # LaTeXVerifier orchestrator
│   │   └── result.py             # VerificationResult types
│   ├── data/                     # Core data types
│   │   └── types.py              # OCRResult, DocumentStructure, BoundingBox, etc.
│   └── evaluation/               # Formula-level metrics
│       ├── base.py
│       ├── formula_metrics.py
│       └── structural.py
├── tests/
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```
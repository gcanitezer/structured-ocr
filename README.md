# Structured OCR

LaTeX OCR System for Full Document Reconstruction

## Features

- OCR pipeline for LaTeX documents
- Compilability verification
- Training and evaluation framework
  - Supervised fine-tuning (SFT) via TRL/PEFT or transformers.Trainer
  - GRPO / RLVR reinforcement learning with nine reward functions
  - LoRA / QLoRA support for efficient fine-tuning
  - Distributed training via torchrun, accelerate, or DeepSpeed
- REST API and CLI interface

## Installation

```bash
pip install -e .
```

To enable the full training pipeline (TRL/PEFT/bitsandbytes/accelerate):

```bash
pip install -e ".[train]"
```

## Usage

```bash
latexocr infer image.png -o output.tex
latexocr verify output.tex
latexocr train sft --config configs/training_sft.yaml
latexocr train grpo --config configs/training_grpo.yaml
latexocr evaluate
```

## Training

The training pipeline has two stages:

1. **SFT** — supervised fine-tuning on the generated LaTeX corpus
   (`latexocr train sft --config configs/training_sft.yaml`).
2. **GRPO/RLVR** — reinforcement learning with the nine reward
   functions in `structured_ocr/training/reward_functions.py`
   (`latexocr train grpo --config configs/training_grpo.yaml`).

Both stages can be launched with the bundled launcher script:

```bash
# Single-node, multi-GPU SFT via torchrun
./scripts/distributed_launch.sh sft --config configs/training_sft.yaml

# Multi-node GRPO via accelerate FSDP
./scripts/distributed_launch.sh grpo --launcher accelerate \
    --accelerate-config configs/accelerate_fsdp.yaml \
    --config configs/training_grpo.yaml
```

See `configs/` for example configurations and DeepSpeed presets.

## Development

```bash
make dev-install
make test
make lint
make format
```

## Docker

```bash
docker build -t structured-ocr .
docker run -it structured-ocr
```
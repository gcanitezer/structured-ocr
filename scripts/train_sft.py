#!/usr/bin/env python3
"""CLI entry point for supervised fine-tuning.

Usage
-----

    python scripts/train_sft.py --config configs/training_sft.yaml
    python scripts/train_sft.py --train-data data/train.jsonl \
        --model-name Qwen/Qwen2-VL-2B-Instruct --output-dir ./outputs/sft

For multi-GPU / multi-node training, launch with ``torchrun``::

    torchrun --nproc_per_node=4 scripts/train_sft.py --config configs/training_sft.yaml

or with ``accelerate``::

    accelerate launch --config_file configs/accelerate_fsdp.yaml scripts/train_sft.py \
        --config configs/training_sft.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from structured_ocr.training import (  # noqa: E402
    SFTTrainer,
    TrainingConfig,
    TrainingMode,
    TrainingPipeline,
)
from structured_ocr.training.dataset_utils import DatasetUtils  # noqa: E402


def _build_config(args: argparse.Namespace) -> TrainingConfig:
    if args.config:
        cfg = TrainingConfig.from_yaml(args.config)
    else:
        cfg = TrainingConfig()
    cfg.mode = TrainingMode.SFT
    cfg.model_name = args.model_name or cfg.model_name
    cfg.output_dir = Path(args.output_dir or cfg.output_dir)
    if args.train_data:
        cfg.train_dataset = Path(args.train_data)
    if args.eval_data:
        cfg.eval_dataset = Path(args.eval_data)
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.gradient_accumulation_steps = args.grad_accum
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.lora:
        cfg.lora.enabled = True
    if args.qlora:
        cfg.lora.enabled = True
        cfg.lora.use_qlora = True
        cfg.quantization.enabled = True
    if args.deepspeed:
        cfg.deepspeed = Path(args.deepspeed)
    if args.fsdp:
        cfg.fsdp = True
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="Path to YAML/JSON training config")
    parser.add_argument("--model-name", type=str, help="Base model name or path")
    parser.add_argument("--train-data", type=str, help="Path to training data (json/jsonl/csv)")
    parser.add_argument("--eval-data", type=str, help="Path to evaluation data")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Override per-device batch size")
    parser.add_argument("--grad-accum", type=int, help="Override gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--qlora", action="store_true", help="Enable QLoRA (4-bit base + LoRA)")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config JSON")
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP via Accelerate")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = _build_config(args)
    logger = logging.getLogger("train_sft")
    logger.info("Starting SFT with config: %s", json.dumps(cfg.to_dict(), default=str))
    pipeline = TrainingPipeline(config=cfg)
    samples, eval_samples = pipeline.load_samples()
    logger.info("Loaded %d training and %d eval samples", len(samples), len(eval_samples))
    trainer = SFTTrainer(cfg)
    result = trainer.train(samples, eval_samples)
    logger.info(
        "SFT complete: train_loss=%.4f eval_loss=%.4f steps=%d elapsed=%.1fs",
        result.train_loss or 0.0,
        result.eval_loss or 0.0,
        result.num_steps,
        result.elapsed_seconds,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

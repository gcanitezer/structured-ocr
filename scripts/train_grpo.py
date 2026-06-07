#!/usr/bin/env python3
"""CLI entry point for GRPO / RLVR training.

Usage
-----

    # Start GRPO from a base model
    python scripts/train_grpo.py --config configs/training_grpo.yaml

    # Continue from an SFT checkpoint
    python scripts/train_grpo.py --config configs/training_grpo.yaml \
        --model-name ./outputs/sft --rlvr-num-generations 4

For multi-GPU / multi-node training, launch with ``torchrun``::

    torchrun --nproc_per_node=4 scripts/train_grpo.py --config configs/training_grpo.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from structured_ocr.training import (  # noqa: E402
    GRPOTrainer,
    RLVRConfig,
    TrainingConfig,
    TrainingMode,
    TrainingPipeline,
)
from structured_ocr.training.reward_functions import RewardFunction  # noqa: E402


def _build_config(args: argparse.Namespace) -> TrainingConfig:
    if args.config:
        cfg = TrainingConfig.from_yaml(args.config)
    else:
        cfg = TrainingConfig()
    cfg.mode = TrainingMode.GRPO
    if args.model_name:
        cfg.model_name = args.model_name
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.train_data:
        cfg.train_dataset = Path(args.train_data)
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
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _build_rlvr(args: argparse.Namespace) -> RLVRConfig:
    rlvr = RLVRConfig()
    if args.rlvr_num_generations is not None:
        rlvr.num_generations = args.rlvr_num_generations
    if args.rlvr_kl_coef is not None:
        rlvr.kl_coef = args.rlvr_kl_coef
    if args.rlvr_clip_ratio is not None:
        rlvr.clip_ratio = args.rlvr_clip_ratio
    if args.rlvr_max_new_tokens is not None:
        rlvr.max_new_tokens = args.rlvr_max_new_tokens
    if args.rlvr_temperature is not None:
        rlvr.temperature = args.rlvr_temperature
    if args.rlvr_max_steps is not None:
        rlvr.max_steps = args.rlvr_max_steps
    if args.rlvr_log_every is not None:
        rlvr.log_rewards_every = args.rlvr_log_every
    return rlvr


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="Path to YAML/JSON training config")
    parser.add_argument("--model-name", type=str, help="Base model (or SFT checkpoint) to start from")
    parser.add_argument("--train-data", type=str, help="Path to training data")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Override per-device batch size")
    parser.add_argument("--grad-accum", type=int, help="Override gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--qlora", action="store_true", help="Enable QLoRA")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config JSON")
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP")
    parser.add_argument("--rlvr-num-generations", type=int, help="Generations per prompt")
    parser.add_argument("--rlvr-kl-coef", type=float, help="KL penalty coefficient")
    parser.add_argument("--rlvr-clip-ratio", type=float, help="GRPO clip ratio (epsilon)")
    parser.add_argument("--rlvr-max-new-tokens", type=int, help="Max tokens to generate per completion")
    parser.add_argument("--rlvr-temperature", type=float, help="Sampling temperature")
    parser.add_argument("--rlvr-max-steps", type=int, help="Hard cap on GRPO optimizer steps")
    parser.add_argument("--rlvr-log-every", type=int, help="Reward log frequency")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = _build_config(args)
    rlvr = _build_rlvr(args)
    logger = logging.getLogger("train_grpo")
    logger.info("Starting GRPO with config: %s", json.dumps(cfg.to_dict(), default=str))
    logger.info("RLVR config: %s", json.dumps(rlvr.to_dict(), default=str))
    pipeline = TrainingPipeline(config=cfg, rlvr=rlvr)
    samples, _ = pipeline.load_samples()
    logger.info("Loaded %d training prompts", len(samples))
    trainer = GRPOTrainer(
        cfg,
        rlvr=rlvr,
        reward_function=RewardFunction(weights=cfg.reward_weights),
    )
    result = trainer.train(samples)
    logger.info(
        "GRPO complete: final_reward=%.4f steps=%d elapsed=%.1fs",
        result.final_reward or 0.0,
        result.num_steps,
        result.elapsed_seconds,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

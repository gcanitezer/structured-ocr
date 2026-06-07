"""Command-line interface for Structured OCR."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click


def _run_script(script_name: str, argv: list[str]) -> None:
    """Execute a sibling scripts/ entry point with the given argv."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / script_name
    if not script.exists():
        click.echo(f"Cannot find {script_name} at {script}", err=True)
        sys.exit(1)
    import runpy

    runpy.run_path(str(script), run_name="__main__")


@click.group()
def main():
    """Structured OCR - LaTeX OCR System for Full Document Reconstruction."""
    pass


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output LaTeX file path")
def infer(image_path: str, output: str | None):
    """Perform OCR on an image and extract LaTeX."""
    click.echo(f"Processing {image_path}...")
    # Implementation will be added
    click.echo("OCR complete")


@main.group()
def train():
    """Training subcommands (SFT, GRPO)."""
    pass


@train.command("sft")
@click.option("--config", "-c", type=click.Path(exists=True), help="Training YAML/JSON config")
@click.option("--model-name", type=str, help="Base model name or path")
@click.option("--train-data", type=click.Path(exists=True), help="Training data file")
@click.option("--eval-data", type=click.Path(exists=True), help="Evaluation data file")
@click.option("--output-dir", type=click.Path(), help="Output directory for checkpoints")
@click.option("--epochs", type=int, help="Override number of training epochs")
@click.option("--batch-size", type=int, help="Override per-device batch size")
@click.option("--grad-accum", type=int, help="Override gradient accumulation steps")
@click.option("--learning-rate", type=float, help="Override learning rate")
@click.option("--lora/--no-lora", default=False, help="Enable LoRA")
@click.option("--qlora/--no-qlora", default=False, help="Enable QLoRA")
@click.option("--deepspeed", type=click.Path(exists=True), help="Path to DeepSpeed JSON config")
@click.option("--fsdp/--no-fsdp", default=False, help="Enable FSDP")
@click.option("--log-level", default="INFO", help="Logging level")
def train_sft(
    config: str | None,
    model_name: str | None,
    train_data: str | None,
    eval_data: str | None,
    output_dir: str | None,
    epochs: int | None,
    batch_size: int | None,
    grad_accum: int | None,
    learning_rate: float | None,
    lora: bool,
    qlora: bool,
    deepspeed: str | None,
    fsdp: bool,
    log_level: str,
):
    """Run supervised fine-tuning."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    from structured_ocr.training import SFTTrainer, TrainingConfig, TrainingMode

    cfg = TrainingConfig.from_yaml(config) if config else TrainingConfig()
    cfg.mode = TrainingMode.SFT
    if model_name:
        cfg.model_name = model_name
    if train_data:
        cfg.train_dataset = train_data
    if eval_data:
        cfg.eval_dataset = eval_data
    if output_dir:
        cfg.output_dir = Path(output_dir)
    if epochs is not None:
        cfg.num_train_epochs = epochs
    if batch_size is not None:
        cfg.per_device_train_batch_size = batch_size
    if grad_accum is not None:
        cfg.gradient_accumulation_steps = grad_accum
    if learning_rate is not None:
        cfg.learning_rate = learning_rate
    if lora:
        cfg.lora.enabled = True
    if qlora:
        cfg.lora.enabled = True
        cfg.lora.use_qlora = True
        cfg.quantization.enabled = True
    if deepspeed:
        cfg.deepspeed = Path(deepspeed)
    if fsdp:
        cfg.fsdp = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    from structured_ocr.training import TrainingPipeline

    pipeline = TrainingPipeline(config=cfg)
    samples, eval_samples = pipeline.load_samples()
    trainer = SFTTrainer(cfg)
    result = trainer.train(samples, eval_samples)
    click.echo(json.dumps(result.to_dict(), indent=2, default=str))


@train.command("grpo")
@click.option("--config", "-c", type=click.Path(exists=True), help="Training YAML/JSON config")
@click.option("--model-name", type=str, help="Base model or SFT checkpoint path")
@click.option("--train-data", type=click.Path(exists=True), help="Training data file")
@click.option("--output-dir", type=click.Path(), help="Output directory for checkpoints")
@click.option("--epochs", type=int, help="Override number of training epochs")
@click.option("--batch-size", type=int, help="Override per-device batch size")
@click.option("--grad-accum", type=int, help="Override gradient accumulation steps")
@click.option("--learning-rate", type=float, help="Override learning rate")
@click.option("--lora/--no-lora", default=False, help="Enable LoRA")
@click.option("--qlora/--no-qlora", default=False, help="Enable QLoRA")
@click.option("--deepspeed", type=click.Path(exists=True), help="Path to DeepSpeed JSON config")
@click.option("--fsdp/--no-fsdp", default=False, help="Enable FSDP")
@click.option("--rlvr-num-generations", type=int, help="Generations per prompt")
@click.option("--rlvr-kl-coef", type=float, help="KL penalty coefficient")
@click.option("--rlvr-clip-ratio", type=float, help="GRPO clip ratio")
@click.option("--rlvr-max-new-tokens", type=int, help="Max tokens to generate per completion")
@click.option("--rlvr-temperature", type=float, help="Sampling temperature")
@click.option("--rlvr-max-steps", type=int, help="Hard cap on GRPO optimizer steps")
@click.option("--log-level", default="INFO", help="Logging level")
def train_grpo(
    config: str | None,
    model_name: str | None,
    train_data: str | None,
    output_dir: str | None,
    epochs: int | None,
    batch_size: int | None,
    grad_accum: int | None,
    learning_rate: float | None,
    lora: bool,
    qlora: bool,
    deepspeed: str | None,
    fsdp: bool,
    rlvr_num_generations: int | None,
    rlvr_kl_coef: float | None,
    rlvr_clip_ratio: float | None,
    rlvr_max_new_tokens: int | None,
    rlvr_temperature: float | None,
    rlvr_max_steps: int | None,
    log_level: str,
):
    """Run GRPO / RLVR fine-tuning with the 9 reward functions."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    from structured_ocr.training import (
        GRPOTrainer,
        RLVRConfig,
        TrainingConfig,
        TrainingMode,
    )
    from structured_ocr.training.reward_functions import RewardFunction

    cfg = TrainingConfig.from_yaml(config) if config else TrainingConfig()
    cfg.mode = TrainingMode.GRPO
    if model_name:
        cfg.model_name = model_name
    if train_data:
        cfg.train_dataset = train_data
    if output_dir:
        cfg.output_dir = Path(output_dir)
    if epochs is not None:
        cfg.num_train_epochs = epochs
    if batch_size is not None:
        cfg.per_device_train_batch_size = batch_size
    if grad_accum is not None:
        cfg.gradient_accumulation_steps = grad_accum
    if learning_rate is not None:
        cfg.learning_rate = learning_rate
    if lora:
        cfg.lora.enabled = True
    if qlora:
        cfg.lora.enabled = True
        cfg.lora.use_qlora = True
        cfg.quantization.enabled = True
    if deepspeed:
        cfg.deepspeed = Path(deepspeed)
    if fsdp:
        cfg.fsdp = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rlvr = RLVRConfig()
    if rlvr_num_generations is not None:
        rlvr.num_generations = rlvr_num_generations
    if rlvr_kl_coef is not None:
        rlvr.kl_coef = rlvr_kl_coef
    if rlvr_clip_ratio is not None:
        rlvr.clip_ratio = rlvr_clip_ratio
    if rlvr_max_new_tokens is not None:
        rlvr.max_new_tokens = rlvr_max_new_tokens
    if rlvr_temperature is not None:
        rlvr.temperature = rlvr_temperature
    if rlvr_max_steps is not None:
        rlvr.max_steps = rlvr_max_steps
    from structured_ocr.training import TrainingPipeline

    pipeline = TrainingPipeline(config=cfg, rlvr=rlvr)
    samples, _ = pipeline.load_samples()
    trainer = GRPOTrainer(
        cfg,
        rlvr=rlvr,
        reward_function=RewardFunction(weights=cfg.reward_weights),
    )
    result = trainer.train(samples)
    click.echo(json.dumps(result.to_dict(), indent=2, default=str))


@main.command()
@click.argument("latex_file", type=click.Path(exists=True))
def verify(latex_file: str):
    """Verify LaTeX compilability."""
    click.echo(f"Verifying {latex_file}...")
    # Implementation will be added


@main.command()
def evaluate():
    """Run evaluation benchmarks."""
    click.echo("Running evaluation...")
    # Implementation will be added


if __name__ == "__main__":
    main()

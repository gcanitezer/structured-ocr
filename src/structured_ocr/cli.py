"""Command-line interface for Structured OCR."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from structured_ocr.corpus import corpus


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


main.add_command(corpus)


@main.group(invoke_without_command=True)
@click.argument("image_path", type=click.Path(exists=True), required=False)
@click.option("--output", "-o", type=click.Path(), help="Output LaTeX file path")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["pix2text", "huggingface"]),
    default=None,
    help="Inference backend",
)
@click.option("--device", "-d", type=str, default=None, help="Device (cpu, cuda, auto)")
@click.option("--model", "-m", type=str, default=None, help="Override model name")
@click.pass_context
def infer(
    ctx: click.Context,
    image_path: str | None,
    output: str | None,
    backend: str | None,
    device: str | None,
    model: str | None,
):
    """Perform OCR on an image and extract LaTeX."""
    if ctx.invoked_subcommand is not None:
        return
    if image_path is None:
        click.echo("Missing argument 'IMAGE_PATH'.", err=True)
        ctx.exit(1)
        return
    _run_infer(image_path, output=output, backend=backend, device=device, model=model)


def _run_infer(
    image_path: str,
    output: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    model: str | None = None,
) -> None:
    """Run single-image inference and handle output."""
    from structured_ocr.inference import InferConfig, InferenceEngine

    cfg = InferConfig()
    if backend:
        cfg.backend = backend
    if device:
        cfg.device = device
    if model:
        cfg.model_name = model

    engine = InferenceEngine(config=cfg)
    click.echo(f"Processing {image_path}...")
    result = engine.infer(image_path)
    if output:
        Path(output).write_text(result.latex)
        click.echo(f"Output written to {output}")
    else:
        click.echo(result.latex)


@infer.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
@click.option("--dpi", type=int, default=150, help="PDF rendering DPI")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["pix2text", "huggingface"]),
    default=None,
    help="Inference backend",
)
@click.option("--device", "-d", type=str, default=None, help="Device (cpu, cuda, auto)")
@click.option("--model", "-m", type=str, default=None, help="Override model name")
def pdf(
    pdf_path: str,
    output: str | None,
    dpi: int,
    backend: str | None,
    device: str | None,
    model: str | None,
):
    """Extract and OCR all pages from a PDF document."""
    from structured_ocr.inference import InferConfig, InferenceEngine
    from structured_ocr.inference.pdf import batch_infer, extract_images_from_pdf

    cfg = InferConfig()
    if backend:
        cfg.backend = backend
    if device:
        cfg.device = device
    if model:
        cfg.model_name = model

    engine = InferenceEngine(config=cfg)
    click.echo(f"Processing PDF: {pdf_path}")
    images = extract_images_from_pdf(pdf_path, dpi=dpi)
    click.echo(f"Extracted {len(images)} pages")
    results = batch_infer(images, engine)
    data = [
        {
            "page": r.page_number,
            "latex": r.latex,
            "confidence": r.confidence,
            "processing_time_ms": r.processing_time_ms,
            "model_name": r.model_name,
            "warnings": r.warnings,
        }
        for r in results
    ]
    output_json = json.dumps(data, indent=2, default=str)
    if output:
        Path(output).write_text(output_json)
        click.echo(f"Results written to {output}")
    else:
        click.echo(output_json)


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
@click.option(
    "--reference",
    "-r",
    type=click.Path(exists=True),
    help="Optional reference .tex file for similarity-based checks.",
)
@click.option(
    "--engine",
    type=click.Choice(["pdflatex", "xelatex", "lualatex"], case_sensitive=False),
    default="pdflatex",
    show_default=True,
    help="LaTeX engine used to verify compilability.",
)
@click.option(
    "--timeout", type=float, default=30.0, show_default=True, help="Per-pass timeout in seconds."
)
@click.option(
    "--passes", type=int, default=2, show_default=True, help="Number of compilation passes."
)
@click.option(
    "--report", type=click.Path(), help="Optional path to write the JSON verification report."
)
def verify(
    latex_file: str,
    reference: str | None,
    engine: str,
    timeout: float,
    passes: int,
    report: str | None,
):
    """Verify LaTeX compilability and structure of a .tex file."""
    from structured_ocr.verification import LaTeXVerifier, VerificationConfig

    cfg = VerificationConfig(
        compiler_engine=engine,
        compiler_timeout=timeout,
        compiler_passes=passes,
    )
    verifier = LaTeXVerifier(config=cfg)
    result = verifier.verify_file(latex_file, reference_path=reference)
    click.echo(json.dumps(result.to_dict(), indent=2, default=str))
    if report:
        verifier.write_report(result, report)
        click.echo(f"Wrote report to {report}")
    if not result.passed:
        ctx = click.get_current_context()
        ctx.exit(1)


@main.group()
def eval():
    """Evaluation subcommands (TexOCR-Bench)."""
    pass


@eval.command()
@click.option(
    "--predictions",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with predictions (sample_id -> latex)",
)
@click.option(
    "--references",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with references (sample_id -> latex)",
)
@click.option(
    "--images",
    "-i",
    type=click.Path(exists=True),
    default=None,
    help="JSON file with image paths (sample_id -> path)",
)
@click.option(
    "--output", "-o", type=click.Path(), default="eval_report.json", help="Output report path"
)
@click.option("--model-name", "-m", default="unknown", help="Model name for report")
@click.option(
    "--baseline",
    "-b",
    type=click.Path(exists=True),
    default=None,
    help="Baseline scores JSON for comparison",
)
@click.option("--no-compilability", is_flag=True, help="Skip compilability checks")
@click.option("--no-references", is_flag=True, help="Skip reference integrity checks")
def evaluate(
    predictions: str,
    references: str,
    images: str | None,
    output: str,
    model_name: str,
    baseline: str | None,
    no_compilability: bool,
    no_references: bool,
):
    """Run evaluation on OCR predictions against references."""
    from structured_ocr.eval.benchmark import BaselineScores, BenchmarkRunner
    from structured_ocr.eval.report import generate_report, save_report

    with open(predictions) as f:
        pred_data = json.load(f)
    with open(references) as f:
        ref_data = json.load(f)

    img_data = None
    if images:
        with open(images) as f:
            img_data = json.load(f)
        img_data = {k: Path(v) for k, v in img_data.items()}

    baseline_scores = None
    if baseline:
        baseline_scores = BaselineScores.load(Path(baseline))

    runner = BenchmarkRunner(
        check_compilability=not no_compilability,
        check_references=not no_references,
    )

    result = runner.run(pred_data, ref_data, img_data, model_name)
    report = generate_report(result, baseline_scores)
    save_report(report, Path(output))

    click.echo(f"Evaluation complete. Results saved to {output}")
    click.echo(f"  Samples: {result.total_samples}")
    click.echo(f"  Edit distance: {result.avg_edit_distance:.4f}")
    click.echo(f"  BLEU: {result.avg_bleu:.4f}")
    click.echo(f"  Compilability: {result.compilability_rate:.2%}")


@eval.command()
@click.option("--latex", "-l", required=True, help="LaTeX string to check")
def check_compilable(latex: str):
    """Check if LaTeX is compilable."""
    from structured_ocr.eval.compilability import CompilabilityChecker

    checker = CompilabilityChecker()
    result = checker.check(latex)

    if result.compilable:
        click.echo(f"Compilable with {result.compiler_used}")
        click.echo(f"  Output: {result.output_path}")
        click.echo(f"  Time: {result.elapsed_seconds}s")
    else:
        click.echo(f"Not compilable after {result.attempts} attempts")
        if result.error_log:
            click.echo(f"  Error: {result.error_log[:200]}")


@eval.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="default_baselines.json",
    help="Output path for baseline scores",
)
def init_baselines(output: str):
    """Initialize default baseline scores file."""
    import json

    from structured_ocr.eval.benchmark import BaselineScores

    baselines = BaselineScores()
    with open(output, "w") as f:
        json.dump(baselines.model_dump(), f, indent=2)
    click.echo(f"Baseline scores written to {output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from structured_ocr.eval import BenchmarkRunner, BaselineScores, generate_report, save_report
from structured_ocr.eval.compilability import CompilabilityChecker


@click.group()
def cli():
    """LaTeX OCR Evaluation CLI - TexOCR-Bench Protocol"""
    pass


@cli.command()
@click.option("--predictions", "-p", required=True, type=click.Path(exists=True),
              help="JSON file with predictions (sample_id -> latex)")
@click.option("--references", "-r", required=True, type=click.Path(exists=True),
              help="JSON file with references (sample_id -> latex)")
@click.option("--images", "-i", type=click.Path(exists=True), default=None,
              help="JSON file with image paths (sample_id -> path)")
@click.option("--output", "-o", type=click.Path(), default="eval_report.json",
              help="Output report path")
@click.option("--model-name", "-m", default="unknown", help="Model name for report")
@click.option("--baseline", "-b", type=click.Path(exists=True), default=None,
              help="Baseline scores JSON for comparison")
@click.option("--no-compilability", is_flag=True, help="Skip compilability checks")
@click.option("--no-references", is_flag=True, help="Skip reference integrity checks")
def evaluate(predictions: str, references: str, images: Optional[str], output: str,
             model_name: str, baseline: Optional[str], no_compilability: bool,
             no_references: bool):
    """Run evaluation on OCR predictions against references"""
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


@cli.command()
@click.option("--latex", "-l", required=True, help="LaTeX string to check")
def check_compilable(latex: str):
    """Check if LaTeX is compilable"""
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


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="default_baselines.json",
              help="Output path for baseline scores")
def init_baselines(output: str):
    """Initialize default baseline scores file"""
    baselines = BaselineScores()
    with open(output, "w") as f:
        json.dump(baselines.model_dump(), f, indent=2)
    click.echo(f"Baseline scores written to {output}")


if __name__ == "__main__":
    cli()
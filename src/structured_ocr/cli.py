"""Command-line interface for Structured OCR."""

import click


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


@main.command()
@click.option("--data-dir", "-d", default="./data", help="Data directory path")
def train(data_dir: str):
    """Train the OCR model."""
    click.echo(f"Training with data from {data_dir}")
    # Implementation will be added


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
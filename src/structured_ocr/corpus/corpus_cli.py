"""CLI for corpus generation."""

import json
from pathlib import Path

import click

from .generator import CorpusGenerator
from .templates import LeafletTemplate, NewspaperTemplate, TextbookTemplate


@click.group()
def corpus():
    """Corpus generation commands."""
    pass


@corpus.command()
@click.option("--output", "-o", default="./data/corpus", help="Output directory")
@click.option("--count", "-n", default=100, help="Number of documents")
@click.option(
    "--subject",
    "-s",
    type=click.Choice(["math", "biology", "chemistry"]),
    default="math",
    help="Textbook subject",
)
def textbooks(output: str, count: int, subject: str):
    """Generate textbook LaTeX corpus."""
    click.echo(f"Generating {count} {subject} textbooks to {output}")
    generator = CorpusGenerator(output_dir=output)
    template = TextbookTemplate(subject=subject)

    for i in range(count):
        result = generator.generate(
            template.generate(seed=i)["latex"],
            f"textbook_{subject}_{i:05d}",
            {"type": "textbook", "subject": subject, "index": i},
        )
        if result["success"]:
            click.echo(f"Generated {result['doc_id']} ({result['num_pages']} pages)")
        else:
            click.echo(
                f"Failed {result.get('doc_id', i)}: {result.get('error', 'unknown')}", err=True
            )


@corpus.command()
@click.option("--output", "-o", default="./data/corpus", help="Output directory")
@click.option("--count", "-n", default=50, help="Number of documents")
def newspapers(output: str, count: int):
    """Generate newspaper LaTeX corpus."""
    click.echo(f"Generating {count} newspapers to {output}")
    generator = CorpusGenerator(output_dir=output)
    template = NewspaperTemplate()

    for i in range(count):
        result = generator.generate(
            template.generate(seed=i)["latex"],
            f"newspaper_{i:05d}",
            {"type": "newspaper", "index": i},
        )
        if result["success"]:
            click.echo(f"Generated {result['doc_id']} ({result['num_pages']} pages)")


@corpus.command()
@click.option("--output", "-o", default="./data/corpus", help="Output directory")
@click.option("--count", "-n", default=50, help="Number of documents")
def leaflets(output: str, count: int):
    """Generate leaflet/brochure LaTeX corpus."""
    click.echo(f"Generating {count} leaflets to {output}")
    generator = CorpusGenerator(output_dir=output)
    template = LeafletTemplate()

    for i in range(count):
        result = generator.generate(
            template.generate(seed=i)["latex"],
            f"leaflet_{i:05d}",
            {"type": "leaflet", "index": i},
        )
        if result["success"]:
            click.echo(f"Generated {result['doc_id']} ({result['num_pages']} pages)")


@corpus.command()
@click.option("--output", "-o", default="./data/corpus_index.json", help="Output index file")
@click.option("--corpus-dir", "-c", default="./data/corpus", help="Corpus directory to index")
def index(output: str, corpus_dir: str):
    """Index generated corpus for HuggingFace format."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        click.echo(f"Corpus directory {corpus_dir} does not exist", err=True)
        return

    index_data = []
    for doc_dir in corpus_path.iterdir():
        if doc_dir.is_dir():
            tex_file = doc_dir / "document.tex"
            pdf_file = doc_dir / "document.pdf"
            images = sorted(doc_dir.glob("page-*.png"))

            if tex_file.exists() and pdf_file.exists() and images:
                index_data.append(
                    {
                        "id": doc_dir.name,
                        "latex": tex_file.read_text(),
                        "pdf": str(pdf_file),
                        "images": [str(img) for img in images],
                    }
                )

    Path(output).write_text(json.dumps(index_data, indent=2))
    click.echo(f"Indexed {len(index_data)} documents to {output}")

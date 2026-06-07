"""LaTeX corpus generator for diverse document types."""

from .corpus_cli import corpus
from .generator import CorpusGenerator
from .templates import (
    LeafletTemplate,
    NewspaperTemplate,
    TextbookTemplate,
)

__all__ = [
    "CorpusGenerator",
    "TextbookTemplate",
    "NewspaperTemplate",
    "LeafletTemplate",
    "corpus",
]

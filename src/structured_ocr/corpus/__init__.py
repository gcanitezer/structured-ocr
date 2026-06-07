"""LaTeX corpus generator for diverse document types."""

from .generator import CorpusGenerator
from .templates import (
    TextbookTemplate,
    NewspaperTemplate,
    LeafletTemplate,
)
from .corpus_cli import corpus

__all__ = [
    "CorpusGenerator",
    "TextbookTemplate",
    "NewspaperTemplate",
    "LeafletTemplate",
    "corpus",
]
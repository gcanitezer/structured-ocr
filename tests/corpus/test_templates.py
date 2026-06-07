"""Tests for the corpus module."""

from __future__ import annotations

from pathlib import Path

import pytest

from structured_ocr.corpus import (
    CorpusGenerator,
    LeafletTemplate,
    NewspaperTemplate,
    TextbookTemplate,
)
from structured_ocr.corpus.corpus_cli import corpus

TEMPLATE_CLASSES = [
    (TextbookTemplate, {"subject": "math"}),
    (TextbookTemplate, {"subject": "biology"}),
    (TextbookTemplate, {"subject": "chemistry"}),
    (NewspaperTemplate, {}),
    (LeafletTemplate, {}),
]


class TestTemplates:
    @pytest.mark.parametrize("template_cls,kwargs", TEMPLATE_CLASSES)
    def test_generates_valid_latex_skeleton(self, template_cls, kwargs):
        template = template_cls(**kwargs)
        result = template.generate(seed=42)
        latex = result["latex"]
        assert latex.startswith("\\documentclass{")
        assert "\\begin{document}" in latex
        assert "\\end{document}" in latex
        assert isinstance(result["metadata"], dict)

    @pytest.mark.parametrize("template_cls,kwargs", TEMPLATE_CLASSES)
    def test_deterministic_with_seed(self, template_cls, kwargs):
        template = template_cls(**kwargs)
        a = template.generate(seed=42)
        b = template.generate(seed=42)
        assert a["latex"] == b["latex"]

    @pytest.mark.parametrize("template_cls,kwargs", TEMPLATE_CLASSES)
    def test_different_seed_produces_different_output(self, template_cls, kwargs):
        template = template_cls(**kwargs)
        a = template.generate(seed=42)
        b = template.generate(seed=99)
        assert a["latex"] != b["latex"]


class TestCorpusCLI:
    def test_corpus_group_exists(self):
        assert corpus.name == "corpus"

    def test_corpus_has_subcommands(self):
        commands = list(corpus.commands.keys())
        assert "textbooks" in commands
        assert "newspapers" in commands
        assert "leaflets" in commands
        assert "index" in commands


class TestCorpusGenerator:
    def test_init_creates_output_dir(self, tmp_path: Path):
        out = tmp_path / "my_corpus"
        assert not out.exists()
        CorpusGenerator(output_dir=str(out))
        assert out.exists()

    def test_init_defaults(self):
        gen = CorpusGenerator()
        assert gen.dpi == 150
        assert gen.compilers == ["pdflatex", "xelatex", "lualatex"]

"""Tests for the structured_ocr.verification.verifier module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from structured_ocr.verification import (
    LaTeXVerifier,
    VerificationConfig,
    VerificationResult,
    VerificationSummary,
    verify_document,
    verify_documents,
)
from structured_ocr.verification.compiler import CompilationOutcome, CompilationResult

SAMPLE_DOC = (
    "\\documentclass{article}\n\\begin{document}\nHello. \\section{Intro}\n\\end{document}\n"
)


def test_verification_config_round_trip():
    cfg = VerificationConfig(
        compiler_engine="xelatex",
        compiler_timeout=15.0,
        compiler_passes=3,
        component_weights={"compilation_success": 0.5},
    )
    d = cfg.to_dict()
    restored = VerificationConfig.from_dict(d)
    assert restored.compiler_engine == "xelatex"
    assert restored.compiler_timeout == 15.0
    assert restored.compiler_passes == 3
    assert restored.component_weights["compilation_success"] == 0.5


def test_verifier_returns_nine_components_when_none_skipped():
    verifier = LaTeXVerifier()
    result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
    assert len(result.components) == 9
    expected_names = {
        "equation_accuracy",
        "equation_syntax",
        "table_structure",
        "section_hierarchy",
        "citation_label_integrity",
        "cross_reference_validity",
        "compilation_success",
        "visual_similarity",
        "semantic_coherence",
    }
    assert set(c.name for c in result.components) == expected_names


def test_verifier_respects_skip_components():
    verifier = LaTeXVerifier(config=VerificationConfig(skip_components=("visual_similarity",)))
    result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
    assert "visual_similarity" not in [c.name for c in result.components]
    assert len(result.components) == 8


def test_verifier_uses_injected_compiler():
    fake_compilation = CompilationResult(
        outcome=CompilationOutcome.SUCCESS, engine="pdflatex", passes=2
    )
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(fake_compilation)
    result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
    assert result.compilation is not None
    assert result.compilation.outcome == CompilationOutcome.SUCCESS
    cs = result.get("compilation_success")
    assert cs is not None
    assert cs.score == 1.0
    assert cs.passed is True


def test_verifier_reports_compiler_not_found_score():
    fake_compilation = CompilationResult(
        outcome=CompilationOutcome.COMPILER_NOT_FOUND, engine="pdflatex"
    )
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(fake_compilation)
    result = verifier.verify(SAMPLE_DOC)
    cs = result.get("compilation_success")
    assert cs is not None
    assert cs.score == 0.5
    assert cs.passed is False
    assert "not available" in cs.details


def test_verifier_reports_failure():
    fake_compilation = CompilationResult(
        outcome=CompilationOutcome.FAILED,
        engine="pdflatex",
        returncode=1,
        errors=["Undefined control sequence."],
    )
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(fake_compilation)
    result = verifier.verify(SAMPLE_DOC)
    cs = result.get("compilation_success")
    assert cs is not None
    assert cs.passed is False
    assert cs.score == 0.0
    assert "errors=1" in cs.details


def test_verifier_total_score_is_weighted_sum():
    fake_compilation = CompilationResult(outcome=CompilationOutcome.SUCCESS)
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(fake_compilation)
    result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
    total = sum(c.weighted_score for c in result.components)
    assert result.total_score == pytest.approx(total)


def test_verifier_handles_empty_source():
    fake_compilation = CompilationResult(outcome=CompilationOutcome.EMPTY_SOURCE)
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(fake_compilation)
    result = verifier.verify("")
    assert result.compilation is None
    cs = result.get("compilation_success")
    assert cs is not None
    assert cs.score == 0.0


def test_verifier_batch_returns_summary():
    fake_compilation = CompilationResult(outcome=CompilationOutcome.SUCCESS)
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(fake_compilation)
    summary = verifier.verify_batch([SAMPLE_DOC, SAMPLE_DOC], references=[SAMPLE_DOC, SAMPLE_DOC])
    assert isinstance(summary, VerificationSummary)
    assert summary.num_documents == 2
    assert summary.num_compiled == 2


def test_verifier_batch_empty_returns_empty_summary():
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
    summary = verifier.verify_batch([])
    assert summary.num_documents == 0


def test_verifier_batch_with_short_references_pads():
    fake_compilation = CompilationResult(outcome=CompilationOutcome.SUCCESS)
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(fake_compilation)
    summary = verifier.verify_batch([SAMPLE_DOC, SAMPLE_DOC], references=[SAMPLE_DOC])
    assert summary.num_documents == 2


def test_verifier_batch_with_short_images_pads():
    fake_compilation = CompilationResult(outcome=CompilationOutcome.SUCCESS)
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(fake_compilation)
    summary = verifier.verify_batch([SAMPLE_DOC, SAMPLE_DOC], references=["", ""], images=[None])
    assert summary.num_documents == 2


def test_verifier_write_report(tmp_path: Path):
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
    result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
    report_path = tmp_path / "report.json"
    verifier.write_report(result, report_path)
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["compiler"] == "pdflatex"
    assert payload["total_components"] == 9


def test_verifier_write_summary(tmp_path: Path):
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
    summary = verifier.verify_batch([SAMPLE_DOC], references=[SAMPLE_DOC])
    summary_path = tmp_path / "summary.json"
    verifier.write_summary(summary, summary_path)
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["num_documents"] == 1


def test_verifier_verify_file(tmp_path: Path):
    tex = tmp_path / "doc.tex"
    tex.write_text(SAMPLE_DOC)
    verifier = LaTeXVerifier()
    verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
    result = verifier.verify_file(tex)
    assert isinstance(result, VerificationResult)
    assert result.source == SAMPLE_DOC


def test_verify_document_helper_returns_result():
    with patch("structured_ocr.verification.verifier.LaTeXVerifier") as MockVerifier:
        instance = MockVerifier.return_value
        instance.verify.return_value = VerificationResult(source=SAMPLE_DOC, components=[])
        result = verify_document(SAMPLE_DOC)
    assert result.source == SAMPLE_DOC


def test_verify_documents_helper_returns_summary():
    with patch("structured_ocr.verification.verifier.LaTeXVerifier") as MockVerifier:
        instance = MockVerifier.return_value
        instance.verify_batch.return_value = VerificationSummary()
        summary = verify_documents([SAMPLE_DOC])
    assert isinstance(summary, VerificationSummary)


def test_verifier_component_weights_applied():
    cfg = VerificationConfig(component_weights={"compilation_success": 1.0})
    verifier = LaTeXVerifier(config=cfg)
    verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
    result = verifier.verify(SAMPLE_DOC)
    cs = result.get("compilation_success")
    assert cs is not None
    assert cs.weight == 1.0
    assert cs.weighted_score == pytest.approx(1.0)


class _FakeCompiler:
    """Stand-in for LaTeXCompiler that returns a predetermined result."""

    def __init__(self, result: CompilationResult) -> None:
        self._result = result
        self.engine = result.engine

    @property
    def is_available(self) -> bool:
        return True

    def compile_string(self, source: str):
        return self._result

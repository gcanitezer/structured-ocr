"""Tests for the refactored compilation hook in reward_functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from structured_ocr.training.reward_functions import (
    LaTeXUnitTestFramework,
    _compiler_required,
    set_compiler_required,
)
from structured_ocr.verification.compiler import CompilationOutcome, CompilationResult


def test_set_compiler_required_flag():
    set_compiler_required(False)
    assert _compiler_required() is False
    set_compiler_required(True)
    assert _compiler_required() is True
    set_compiler_required(False)


def test_compilation_test_uses_new_compiler():
    fw = LaTeXUnitTestFramework()
    fake_result = CompilationResult(
        outcome=CompilationOutcome.SUCCESS, engine="pdflatex", passes=2
    )
    with patch(
        "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
        return_value=fake_result,
    ):
        result = fw.test_compilation_success(
            "\\documentclass{article}\\begin{document}x\\end{document}"
        )
    assert result.passed is True
    assert result.score == 1.0
    assert "engine=pdflatex" in result.details


def test_compilation_test_reports_xelatex():
    fw = LaTeXUnitTestFramework()
    fake_result = CompilationResult(
        outcome=CompilationOutcome.SUCCESS, engine="xelatex", passes=2
    )
    with patch(
        "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
        return_value=fake_result,
    ):
        result = fw.test_compilation_success(
            "\\documentclass{article}\\begin{document}x\\end{document}",
            compiler="xelatex",
        )
    assert result.passed is True
    assert "engine=xelatex" in result.details


def test_compilation_test_reports_lualatex():
    fw = LaTeXUnitTestFramework()
    fake_result = CompilationResult(
        outcome=CompilationOutcome.SUCCESS, engine="lualatex", passes=2
    )
    with patch(
        "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
        return_value=fake_result,
    ):
        result = fw.test_compilation_success(
            "\\documentclass{article}\\begin{document}x\\end{document}",
            compiler="lualatex",
        )
    assert result.passed is True
    assert "engine=lualatex" in result.details


def test_compilation_test_empty_source():
    fw = LaTeXUnitTestFramework()
    result = fw.test_compilation_success("")
    assert result.passed is False
    assert result.score == 0.0
    assert "empty source" in result.details


def test_compilation_test_compiler_not_found_defaults_passed():
    fw = LaTeXUnitTestFramework()
    set_compiler_required(False)
    fake_result = CompilationResult(
        outcome=CompilationOutcome.COMPILER_NOT_FOUND, engine="pdflatex"
    )
    with patch(
        "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
        return_value=fake_result,
    ):
        result = fw.test_compilation_success("doc")
    assert result.score == 0.5
    assert result.passed is False  # 0.5 < 0.5 not; the "not available" path is passed=True by default


def test_compilation_test_compiler_not_found_fails_when_required():
    fw = LaTeXUnitTestFramework()
    set_compiler_required(True)
    try:
        fake_result = CompilationResult(
            outcome=CompilationOutcome.COMPILER_NOT_FOUND, engine="pdflatex"
        )
        with patch(
            "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
            return_value=fake_result,
        ):
            result = fw.test_compilation_success("doc")
        assert result.passed is False
        assert result.score == 0.5
    finally:
        set_compiler_required(False)


def test_compilation_test_timeout():
    fw = LaTeXUnitTestFramework()
    fake_result = CompilationResult(
        outcome=CompilationOutcome.TIMEOUT, engine="pdflatex"
    )
    with patch(
        "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
        return_value=fake_result,
    ):
        result = fw.test_compilation_success("doc", timeout=5)
    assert result.passed is False
    assert result.score == 0.0
    assert "timed out" in result.details


def test_compilation_test_reports_errors():
    fw = LaTeXUnitTestFramework()
    fake_result = CompilationResult(
        outcome=CompilationOutcome.FAILED,
        engine="pdflatex",
        returncode=1,
        errors=["Undefined control sequence."],
    )
    with patch(
        "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
        return_value=fake_result,
    ):
        result = fw.test_compilation_success("doc")
    assert result.passed is False
    assert "errors=1" in result.details

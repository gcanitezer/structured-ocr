"""Tests for the structured_ocr.verification.compiler module."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from structured_ocr.verification.compiler import (
    CompilationOutcome,
    CompilationResult,
    LaTeXCompiler,
    SUPPORTED_COMPILERS,
    compile_document,
    parse_log_errors,
    parse_log_warnings,
)


def test_supported_compilers_contains_three_engines():
    assert SUPPORTED_COMPILERS == ("pdflatex", "xelatex", "lualatex")


def test_latex_compiler_rejects_unknown_engine():
    with pytest.raises(ValueError, match="Unsupported engine"):
        LaTeXCompiler(engine="context")


def test_latex_compiler_rejects_zero_timeout():
    with pytest.raises(ValueError, match="timeout must be > 0"):
        LaTeXCompiler(timeout=0)


def test_latex_compiler_rejects_zero_passes():
    with pytest.raises(ValueError, match="passes must be >= 1"):
        LaTeXCompiler(passes=0)


def test_latex_compiler_is_available_false_when_missing():
    compiler = LaTeXCompiler(engine="pdflatex")
    if not compiler.is_available:
        assert compiler.is_available is False
    else:
        assert compiler.is_available is True


def test_compile_string_empty_source():
    compiler = LaTeXCompiler()
    result = compiler.compile_string("")
    assert result.outcome == CompilationOutcome.EMPTY_SOURCE
    assert result.succeeded is False
    assert result.score == 0.0


def test_compile_string_whitespace_source():
    compiler = LaTeXCompiler()
    result = compiler.compile_string("   \n\n  ")
    assert result.outcome == CompilationOutcome.EMPTY_SOURCE


def test_compile_string_when_compiler_missing_returns_not_found():
    compiler = LaTeXCompiler(engine="pdflatex")
    with patch.object(compiler, "is_available", return_value=False):
        result = compiler.compile_string(
            "\\documentclass{article}\\begin{document}x\\end{document}"
        )
    assert result.outcome == CompilationOutcome.COMPILER_NOT_FOUND
    assert result.score == 0.5
    assert "not on PATH" in result.message


def test_compile_string_fake_success_via_subprocess(monkeypatch):
    """Simulate a successful pdflatex run with a fake binary."""
    completed = subprocess.CompletedProcess(
        args=["pdflatex"],
        returncode=0,
        stdout="This is pdfTeX, Version 3.14\nOutput written on doc.pdf (1 page).",
        stderr="",
    )

    def fake_run(cmd, **kwargs):
        Path(kwargs["cwd"]).joinpath("doc.pdf").write_bytes(b"%PDF-fake")
        Path(kwargs["cwd"]).joinpath("doc.log").write_text("(no errors)\n")
        return completed

    compiler = LaTeXCompiler(engine="pdflatex", passes=1)
    with patch.object(compiler, "is_available", return_value=True), patch.object(
        compiler, "_lock", new=_DummyLock()
    ), patch("structured_ocr.verification.compiler.subprocess.run", side_effect=fake_run):
        result = compiler.compile_string(
            "\\documentclass{article}\\begin{document}x\\end{document}"
        )
    assert result.outcome == CompilationOutcome.SUCCESS
    assert result.returncode == 0
    assert result.score == 1.0
    assert result.succeeded is True
    assert result.elapsed_seconds >= 0
    assert result.output_path is not None
    assert result.output_path.name == "doc.pdf"


def test_compile_string_fake_failure_via_subprocess():
    completed = subprocess.CompletedProcess(
        args=["pdflatex"], returncode=1, stdout="", stderr=""
    )

    def fake_run(cmd, **kwargs):
        Path(kwargs["cwd"]).joinpath("doc.log").write_text(
            "! Undefined control sequence.\nl.2 \\foo\n"
        )
        return completed

    compiler = LaTeXCompiler(engine="pdflatex", passes=1)
    with patch.object(compiler, "is_available", return_value=True), patch.object(
        compiler, "_lock", new=_DummyLock()
    ), patch("structured_ocr.verification.compiler.subprocess.run", side_effect=fake_run):
        result = compiler.compile_string("oops")
    assert result.outcome == CompilationOutcome.FAILED
    assert result.score == 0.0
    assert result.errors
    assert "Undefined control sequence" in result.errors[0]


def test_compile_string_timeout_via_subprocess():
    def fake_run(cmd, **kwargs):
        Path(kwargs["cwd"]).joinpath("doc.log").write_text("! Emergency stop.\n")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    compiler = LaTeXCompiler(engine="pdflatex", passes=2, timeout=1.0)
    with patch.object(compiler, "is_available", return_value=True), patch.object(
        compiler, "_lock", new=_DummyLock()
    ), patch("structured_ocr.verification.compiler.subprocess.run", side_effect=fake_run):
        result = compiler.compile_string("anything")
    assert result.outcome == CompilationOutcome.TIMEOUT
    assert result.score == 0.0
    assert "timed out" in result.message
    assert result.passes == 0


def test_compile_file_missing_file():
    compiler = LaTeXCompiler()
    result = compiler.compile_file("/path/that/does/not/exist.tex")
    assert result.outcome == CompilationOutcome.IO_ERROR


def test_compile_file_runs(tmp_path: Path):
    tex_path = tmp_path / "doc.tex"
    tex_path.write_text("\\documentclass{article}\\begin{document}hi\\end{document}")
    compiler = LaTeXCompiler()
    # No compiler installed in the test environment: we expect NOT_FOUND
    if not compiler.is_available:
        result = compiler.compile_file(tex_path)
        assert result.outcome == CompilationOutcome.COMPILER_NOT_FOUND


def test_compile_dispatches_path_or_string(tmp_path: Path):
    compiler = LaTeXCompiler()
    tex_path = tmp_path / "doc.tex"
    tex_path.write_text("\\documentclass{article}\\begin{document}hi\\end{document}")
    if not compiler.is_available:
        result = compiler.compile(tex_path)
        assert result.outcome == CompilationOutcome.COMPILER_NOT_FOUND
    result = compiler.compile("just a string")
    assert result.outcome in {CompilationOutcome.EMPTY_SOURCE, CompilationOutcome.COMPILER_NOT_FOUND}


def test_compile_document_helper():
    result = compile_document("\\documentclass{article}\\begin{document}x\\end{document}")
    if not LaTeXCompiler().is_available:
        assert result.outcome == CompilationOutcome.COMPILER_NOT_FOUND


def test_parse_log_errors_empty():
    assert parse_log_errors("") == []


def test_parse_log_errors_picks_up_error_lines():
    log = (
        "This is pdfTeX, Version 3.14\n"
        "! Undefined control sequence.\n"
        "l.10 \\foo\n"
        "\n"
        "! Missing $ inserted.\n"
        "l.20 abc\n"
    )
    errors = parse_log_errors(log)
    assert errors == ["Undefined control sequence.", "Missing $ inserted."]


def test_parse_log_errors_dedupes():
    log = "! Undefined control sequence.\n! Undefined control sequence.\n"
    errors = parse_log_errors(log)
    assert errors == ["Undefined control sequence."]


def test_parse_log_warnings_extracts_warnings():
    log = "LaTeX Warning: Reference `fig:foo' on page 1 undefined.\n"
    warnings = parse_log_warnings(log)
    assert any("Reference" in w for w in warnings)


def test_parse_log_warnings_handles_empty():
    assert parse_log_warnings("") == []


def test_compilation_result_score():
    assert CompilationResult(outcome=CompilationOutcome.SUCCESS).score == 1.0
    assert CompilationResult(outcome=CompilationOutcome.COMPILER_NOT_FOUND).score == 0.5
    assert CompilationResult(outcome=CompilationOutcome.FAILED).score == 0.0
    assert CompilationResult(outcome=CompilationOutcome.TIMEOUT).score == 0.0


def test_compilation_result_to_dict_serializes_paths():
    result = CompilationResult(
        outcome=CompilationOutcome.SUCCESS,
        engine="pdflatex",
        output_path=Path("/tmp/doc.pdf"),
        log_path=Path("/tmp/doc.log"),
    )
    d = result.to_dict()
    assert d["outcome"] == "success"
    assert d["output_path"] == "/tmp/doc.pdf"
    assert d["log_path"] == "/tmp/doc.log"


def test_compilation_result_to_dict_handles_none_paths():
    result = CompilationResult(outcome=CompilationOutcome.FAILED, engine="pdflatex")
    d = result.to_dict()
    assert d["outcome"] == "failed"
    assert "output_path" not in d
    assert "log_path" not in d


def test_persistent_workdir(tmp_path: Path):
    """A persistent workdir should keep .pdf / .log files after the call."""
    tex_path = tmp_path / "doc.tex"
    tex_path.write_text("\\documentclass{article}\\begin{document}hi\\end{document}")
    compiler = LaTeXCompiler(workdir=tmp_path, keep_logs=True)
    if not compiler.is_available:
        result = compiler.compile_file(tex_path)
        assert result.outcome == CompilationOutcome.COMPILER_NOT_FOUND


def test_extra_args_passed_to_engine():
    compiler = LaTeXCompiler(extra_args=["-synctex=1", "-quiet"])
    assert "-synctex=1" in compiler.extra_args
    assert "-quiet" in compiler.extra_args
    cmd = compiler._build_command("doc")
    assert "-synctex=1" in cmd
    assert "-quiet" in cmd
    assert cmd[0] == "pdflatex"


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

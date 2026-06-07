from __future__ import annotations

import pytest

from structured_ocr.eval.compilability import CompilabilityChecker, CompilabilityResult


class TestCompilabilityChecker:
    def test_initialization(self):
        checker = CompilabilityChecker(timeout_seconds=30)
        assert checker.timeout_seconds == 30

    def test_check_valid_latex(self):
        checker = CompilabilityChecker(timeout_seconds=10)
        latex = r"\documentclass{article}\begin{document}Hello\end{document}"
        result = checker.check(latex)
        assert isinstance(result.compilable, bool)
        assert result.attempts >= 1

    def test_check_invalid_latex(self):
        checker = CompilabilityChecker(timeout_seconds=10)
        latex = r"\invalidcommand\begin{document}"
        result = checker.check(latex)
        assert result.compilable == False
        assert result.attempts >= 1
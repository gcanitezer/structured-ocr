"""LaTeX compilation verification utilities.

This package provides a thin, well-tested wrapper around the three
major LaTeX engines (pdflatex, xelatex, lualatex) plus a high-level
:class:`LaTeXVerifier` orchestrator that runs the full unit-test
battery from :mod:`structured_ocr.training.reward_functions` against
a generated LaTeX document and produces a single, structured
:class:`VerificationResult`.

The package is intentionally dependency-free beyond the Python
standard library, so it can be used both inside and outside the
training pipeline (e.g. as a post-processing step in the OCR API).
"""

from .compiler import (
    CompilationOutcome,
    CompilationResult,
    LaTeXCompiler,
    SUPPORTED_COMPILERS,
    compile_document,
    parse_log_errors,
)
from .result import (
    ComponentResult,
    VerificationResult,
    VerificationSummary,
)
from .verifier import (
    LaTeXVerifier,
    VerificationConfig,
    verify_document,
    verify_documents,
)

__all__ = [
    "CompilationOutcome",
    "CompilationResult",
    "LaTeXCompiler",
    "SUPPORTED_COMPILERS",
    "compile_document",
    "parse_log_errors",
    "ComponentResult",
    "VerificationResult",
    "VerificationSummary",
    "LaTeXVerifier",
    "VerificationConfig",
    "verify_document",
    "verify_documents",
]

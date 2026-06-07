"""Structured OCR - LaTeX OCR System for Full Document Reconstruction."""

from . import verification as verification
from .verification import (
    ComponentResult,
    CompilationOutcome,
    CompilationResult,
    LaTeXCompiler,
    LaTeXVerifier,
    SUPPORTED_COMPILERS,
    VerificationConfig,
    VerificationResult,
    VerificationSummary,
    compile_document,
    parse_log_errors,
    verify_document,
    verify_documents,
)

__version__ = "0.1.0"

__all__ = [
    "verification",
    "ComponentResult",
    "CompilationOutcome",
    "CompilationResult",
    "LaTeXCompiler",
    "LaTeXVerifier",
    "SUPPORTED_COMPILERS",
    "VerificationConfig",
    "VerificationResult",
    "VerificationSummary",
    "compile_document",
    "parse_log_errors",
    "verify_document",
    "verify_documents",
]
from __future__ import annotations

from .metrics import (
    calculate_edit_distance,
    calculate_bleu,
    calculate_structural_metrics,
    StructuralMetricsResult,
)
from .compilability import CompilabilityChecker, CompilabilityResult, compare_rendered_images
from .references import ReferenceIntegrityChecker, ReferenceReport
from .benchmark import BenchmarkRunner, BenchmarkResult, run_benchmark
from .report import generate_report, save_report

__all__ = [
    "calculate_edit_distance",
    "calculate_bleu",
    "calculate_structural_metrics",
    "StructuralMetricsResult",
    "CompilabilityChecker",
    "CompilabilityResult",
    "compare_rendered_images",
    "ReferenceIntegrityChecker",
    "ReferenceReport",
    "BenchmarkRunner",
    "BenchmarkResult",
    "run_benchmark",
    "generate_report",
    "save_report",
]
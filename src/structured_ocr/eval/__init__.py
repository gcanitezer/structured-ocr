from __future__ import annotations

from .benchmark import BenchmarkResult, BenchmarkRunner, run_benchmark
from .compilability import CompilabilityChecker, CompilabilityResult, compare_rendered_images
from .metrics import (
    StructuralMetricsResult,
    calculate_bleu,
    calculate_edit_distance,
    calculate_structural_metrics,
)
from .references import ReferenceIntegrityChecker, ReferenceReport
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

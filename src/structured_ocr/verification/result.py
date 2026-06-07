"""Structured result types returned by :class:`LaTeXVerifier`.

The verification pipeline produces three levels of detail:

* :class:`ComponentResult` — the score and metadata for a single
  unit test (e.g. compilation success, equation accuracy, ...).
* :class:`VerificationResult` — the result for a single document.
* :class:`VerificationSummary` — an aggregate report across a batch
  of documents, suitable for logging into the training pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .compiler import CompilationResult


@dataclass
class ComponentResult:
    """The score and outcome of a single verification component.

    The component name matches one of the unit-test names exposed by
    :class:`structured_ocr.training.reward_functions.LaTeXUnitTestFramework`.
    """

    name: str
    passed: bool
    score: float
    weight: float = 0.0
    weighted_score: float = 0.0
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Full verification report for a single LaTeX document."""

    source: str
    reference: str = ""
    compiler: str = "pdflatex"
    components: List[ComponentResult] = field(default_factory=list)
    compilation: Optional[CompilationResult] = None
    total_score: float = 0.0
    passed_components: int = 0
    total_components: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.passed_components == self.total_components and self.total_components > 0

    @property
    def pass_rate(self) -> float:
        if self.total_components == 0:
            return 0.0
        return self.passed_components / self.total_components

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "compiler": self.compiler,
            "components": [c.to_dict() for c in self.components],
            "compilation": self.compilation.to_dict() if self.compilation else None,
            "total_score": self.total_score,
            "passed_components": self.passed_components,
            "total_components": self.total_components,
            "pass_rate": self.pass_rate,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "reference": self.reference,
            "extra": self.extra,
        }
        return d

    def components_by_name(self) -> Dict[str, ComponentResult]:
        return {c.name: c for c in self.components}

    def get(self, name: str) -> Optional[ComponentResult]:
        return self.components_by_name().get(name)


@dataclass
class VerificationSummary:
    """Aggregate verification report across a batch of documents."""

    results: List[VerificationResult] = field(default_factory=list)
    batch_score: float = 0.0
    batch_pass_rate: float = 0.0
    component_averages: Dict[str, float] = field(default_factory=dict)
    component_pass_rates: Dict[str, float] = field(default_factory=dict)
    num_documents: int = 0
    num_compiled: int = 0
    num_failed: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_documents": self.num_documents,
            "batch_score": self.batch_score,
            "batch_pass_rate": self.batch_pass_rate,
            "component_averages": self.component_averages,
            "component_pass_rates": self.component_pass_rates,
            "num_compiled": self.num_compiled,
            "num_failed": self.num_failed,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }


__all__ = [
    "ComponentResult",
    "VerificationResult",
    "VerificationSummary",
]

"""High-level :class:`LaTeXVerifier` orchestrator.

The verifier brings together :class:`LaTeXCompiler` and the
:class:`LaTeXUnitTestFramework` into a single, configurable
pipeline. It accepts a generated LaTeX document, optionally a
reference, runs the full battery of unit tests, and produces a
:class:`VerificationResult` whose ``total_score`` is suitable for
direct use as a reward signal in GRPO/RLVR training.

Typical usage::

    verifier = LaTeXVerifier()
    result = verifier.verify(predicted_latex, reference=reference_latex)
    if not result.passed:
        log(result.compilation.errors)

    summary = verifier.verify_batch(predictions, references)
    print(summary.batch_pass_rate)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .compiler import CompilationOutcome, CompilationResult, LaTeXCompiler
from .result import ComponentResult, VerificationResult, VerificationSummary

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Settings that control :class:`LaTeXVerifier`.

    All fields have sensible defaults; the only one most callers
    override is :attr:`compiler_engine` (e.g. to ``"xelatex"`` for
    documents that require Unicode / OTF fonts).
    """

    compiler_engine: str = "pdflatex"
    compiler_timeout: float = 30.0
    compiler_passes: int = 2
    keep_logs: bool = False
    component_weights: Dict[str, float] = field(default_factory=dict)
    skip_components: tuple[str, ...] = ()
    fail_threshold: float = 0.5
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compiler_engine": self.compiler_engine,
            "compiler_timeout": self.compiler_timeout,
            "compiler_passes": self.compiler_passes,
            "keep_logs": self.keep_logs,
            "component_weights": dict(self.component_weights),
            "skip_components": list(self.skip_components),
            "fail_threshold": self.fail_threshold,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationConfig":
        return cls(
            compiler_engine=data.get("compiler_engine", "pdflatex"),
            compiler_timeout=float(data.get("compiler_timeout", 30.0)),
            compiler_passes=int(data.get("compiler_passes", 2)),
            keep_logs=bool(data.get("keep_logs", False)),
            component_weights=dict(data.get("component_weights", {})),
            skip_components=tuple(data.get("skip_components", ())),
            fail_threshold=float(data.get("fail_threshold", 0.5)),
            extra=dict(data.get("extra", {})),
        )


def _default_weights() -> Dict[str, float]:
    return {
        "equation_accuracy": 0.15,
        "equation_syntax": 0.15,
        "table_structure": 0.10,
        "section_hierarchy": 0.10,
        "citation_label_integrity": 0.10,
        "cross_reference_validity": 0.10,
        "compilation_success": 0.20,
        "visual_similarity": 0.05,
        "semantic_coherence": 0.05,
    }


class LaTeXVerifier:
    """Orchestrates compilation + unit tests for a LaTeX document.

    The verifier lazily imports
    :class:`structured_ocr.training.reward_functions.LaTeXUnitTestFramework`
    to avoid a hard import cycle: the training pipeline needs the
    reward functions, and the reward functions now want to use the
    :class:`LaTeXCompiler` for more accurate compilation scoring.

    Parameters
    ----------
    config:
        Optional :class:`VerificationConfig` overriding the defaults.
    compiler:
        Optional pre-built :class:`LaTeXCompiler` (useful for tests
        and for sharing a single configured compiler across many
        verification calls).
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        compiler: Optional[LaTeXCompiler] = None,
    ) -> None:
        self.config = config or VerificationConfig()
        weights = _default_weights()
        weights.update(self.config.component_weights)
        self._component_weights = weights
        self._skip = set(self.config.skip_components)
        if compiler is not None:
            self._compiler = compiler
        else:
            self._compiler = LaTeXCompiler(
                engine=self.config.compiler_engine,
                timeout=self.config.compiler_timeout,
                passes=self.config.compiler_passes,
                keep_logs=self.config.keep_logs,
            )

    @property
    def compiler(self) -> LaTeXCompiler:
        return self._compiler

    def verify(
        self,
        source: str,
        reference: str = "",
        extracted_image: Optional[bytes] = None,
    ) -> VerificationResult:
        """Run the full verification pipeline on a single document."""
        from structured_ocr.training.reward_functions import LaTeXUnitTestFramework

        framework = LaTeXUnitTestFramework()
        unit_results = framework.run_tests(
            predicted=source, reference=reference, extracted_image=extracted_image
        )

        compilation = self._maybe_compile(source, unit_results)
        components: List[ComponentResult] = []
        for name in self._ordered_components():
            if name in self._skip:
                continue
            if name == "compilation_success":
                if compilation is not None:
                    score = compilation.score
                    passed = compilation.outcome == CompilationOutcome.SUCCESS
                    details = (
                        f"engine={compilation.engine} "
                        f"outcome={compilation.outcome.value} "
                        f"passes={compilation.passes} "
                        f"elapsed={compilation.elapsed_seconds:.2f}s"
                    )
                    if compilation.errors:
                        details += f" errors={len(compilation.errors)}"
                else:
                    score = unit_results[name].score
                    passed = unit_results[name].passed
                    details = unit_results[name].details
            else:
                res = unit_results[name]
                score = res.score
                passed = res.passed
                details = res.details
            weight = self._component_weights.get(name, 0.0)
            components.append(
                ComponentResult(
                    name=name,
                    passed=passed,
                    score=score,
                    weight=weight,
                    weighted_score=score * weight,
                    details=details,
                )
            )
        total_score = sum(c.weighted_score for c in components)
        passed = sum(1 for c in components if c.passed)
        return VerificationResult(
            source=source,
            reference=reference,
            compiler=self.config.compiler_engine,
            components=components,
            compilation=compilation,
            total_score=total_score,
            passed_components=passed,
            total_components=len(components),
        )

    def verify_batch(
        self,
        sources: Sequence[str],
        references: Optional[Sequence[str]] = None,
        images: Optional[Sequence[Optional[bytes]]] = None,
    ) -> VerificationSummary:
        """Verify a batch of documents and return aggregate statistics."""
        if not sources:
            return VerificationSummary()
        refs: List[str] = list(references) if references is not None else [""] * len(sources)
        if len(refs) < len(sources):
            refs.extend([""] * (len(sources) - len(refs)))
        imgs: List[Optional[bytes]] = list(images) if images is not None else [None] * len(sources)
        if len(imgs) < len(sources):
            imgs.extend([None] * (len(sources) - len(imgs)))
        results: List[VerificationResult] = []
        for source, ref, img in zip(sources, refs, imgs):
            try:
                results.append(self.verify(source, ref, img))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("verification of one document failed: %s", exc)
        return self._summarize(results)

    def verify_file(
        self, path: Union[str, Path], reference_path: Optional[Union[str, Path]] = None
    ) -> VerificationResult:
        """Verify a ``.tex`` file on disk."""
        path = Path(path)
        source = path.read_text(errors="replace")
        reference = ""
        if reference_path is not None:
            reference = Path(reference_path).read_text(errors="replace")
        return self.verify(source, reference=reference)

    def write_report(
        self,
        result: VerificationResult,
        path: Union[str, Path],
    ) -> Path:
        """Persist a :class:`VerificationResult` to disk as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        return path

    def write_summary(
        self,
        summary: VerificationSummary,
        path: Union[str, Path],
    ) -> Path:
        """Persist a :class:`VerificationSummary` to disk as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        return path

    def _ordered_components(self) -> List[str]:
        return [
            "equation_accuracy",
            "equation_syntax",
            "table_structure",
            "section_hierarchy",
            "citation_label_integrity",
            "cross_reference_validity",
            "compilation_success",
            "visual_similarity",
            "semantic_coherence",
        ]

    def _maybe_compile(
        self,
        source: str,
        unit_results: Dict[str, Any],
    ) -> Optional[CompilationResult]:
        """Re-run the compilation test using the new compiler wrapper.

        If the source is empty (e.g. when only the reference is
        available) we fall back to the unit-test result without
        re-running the compiler.
        """
        if "compilation_success" in self._skip:
            return None
        if not source or not source.strip():
            return None
        return self._compiler.compile_string(source)

    @staticmethod
    def _summarize(results: Sequence[VerificationResult]) -> VerificationSummary:
        if not results:
            return VerificationSummary()
        total = len(results)
        scores = [r.total_score for r in results]
        pass_rates = [r.pass_rate for r in results]
        names = results[0].components_by_name().keys() if results else []
        averages: Dict[str, float] = {}
        pass_rate_per_component: Dict[str, float] = {}
        for name in names:
            values = [r.get(name).score for r in results if r.get(name) is not None]
            passes = [r.get(name).passed for r in results if r.get(name) is not None]
            if values:
                averages[name] = sum(values) / len(values)
            if passes:
                pass_rate_per_component[name] = sum(1 for p in passes if p) / len(passes)
        num_compiled = sum(
            1
            for r in results
            if r.compilation is not None and r.compilation.outcome == CompilationOutcome.SUCCESS
        )
        return VerificationSummary(
            results=list(results),
            batch_score=sum(scores) / len(scores),
            batch_pass_rate=sum(pass_rates) / len(pass_rates),
            component_averages=averages,
            component_pass_rates=pass_rate_per_component,
            num_documents=total,
            num_compiled=num_compiled,
            num_failed=total - num_compiled,
        )


def verify_document(
    source: str,
    reference: str = "",
    *,
    engine: str = "pdflatex",
    timeout: float = 30.0,
    passes: int = 2,
) -> VerificationResult:
    """Functional helper: verify a single LaTeX document."""
    verifier = LaTeXVerifier(
        config=VerificationConfig(
            compiler_engine=engine,
            compiler_timeout=timeout,
            compiler_passes=passes,
        )
    )
    return verifier.verify(source, reference=reference)


def verify_documents(
    sources: Sequence[str],
    references: Optional[Sequence[str]] = None,
    *,
    engine: str = "pdflatex",
    timeout: float = 30.0,
    passes: int = 2,
) -> VerificationSummary:
    """Functional helper: verify a batch of LaTeX documents."""
    verifier = LaTeXVerifier(
        config=VerificationConfig(
            compiler_engine=engine,
            compiler_timeout=timeout,
            compiler_passes=passes,
        )
    )
    return verifier.verify_batch(sources, references)


__all__ = [
    "LaTeXVerifier",
    "VerificationConfig",
    "verify_document",
    "verify_documents",
]

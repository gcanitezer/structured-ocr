"""Reward functions used by the GRPO/RLVR training stage.

There are nine reward components, each scoring a different aspect of a
generated LaTeX document. They can be combined via :class:`RewardConfig`
or used individually for analysis.

1. equation_accuracy
2. equation_syntax
3. table_structure
4. section_hierarchy
5. citation_label_integrity
6. cross_reference_validity
7. compilation_success
8. visual_similarity
9. semantic_coherence
"""

from __future__ import annotations

import difflib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .types import RewardConfig

logger = logging.getLogger(__name__)


_COMPILER_REQUIRED: bool = False
"""Global flag: when ``True``, the compilation test fails if the
engine is not on ``$PATH`` instead of returning a neutral 0.5 score."""


def set_compiler_required(required: bool) -> None:
    """Configure whether the compilation test must find a real engine.

    Defaults to ``False`` so unit tests work in minimal environments
    without a TeX distribution installed. CI / training jobs that
    need a hard signal from compilation should call
    :func:`set_compiler_required` with ``True`` at startup.
    """
    global _COMPILER_REQUIRED
    _COMPILER_REQUIRED = bool(required)


def _compiler_required() -> bool:
    return _COMPILER_REQUIRED


@dataclass
class UnitTestResult:
    """Result of a single unit test."""

    test_name: str
    passed: bool
    score: float
    details: str = ""


@dataclass
class RewardResult:
    """Aggregate result from the reward function."""

    total_reward: float
    components: Dict[str, float]
    breakdown: Dict[str, str]
    passed_tests: int
    total_tests: int


class LaTeXUnitTestFramework:
    """Per-component unit tests for evaluating LaTeX OCR output."""

    EQUATION_ENVS = ("equation", "align", "gather", "multline", "eqnarray", "displaymath")

    def run_tests(
        self,
        predicted: str,
        reference: str = "",
        extracted_image: Optional[bytes] = None,
    ) -> Dict[str, UnitTestResult]:
        return {
            "equation_accuracy": self.test_equation_accuracy(predicted, reference),
            "equation_syntax": self.test_equation_syntax(predicted),
            "table_structure": self.test_table_structure(predicted, reference),
            "section_hierarchy": self.test_section_hierarchy(predicted, reference),
            "citation_label_integrity": self.test_citation_label_integrity(predicted),
            "cross_reference_validity": self.test_cross_reference_validity(predicted),
            "compilation_success": self.test_compilation_success(predicted),
            "visual_similarity": self.test_visual_similarity(predicted, extracted_image),
            "semantic_coherence": self.test_semantic_coherence(predicted, reference),
        }

    def test_equation_accuracy(self, predicted: str, reference: str) -> UnitTestResult:
        pred_eqs = self._extract_equation_bodies(predicted)
        ref_eqs = self._extract_equation_bodies(reference)
        if not ref_eqs:
            return UnitTestResult("equation_accuracy", True, 1.0, "no reference equations")
        if not pred_eqs:
            return UnitTestResult("equation_accuracy", False, 0.0, "no predicted equations")
        score = self._sequence_similarity(pred_eqs, ref_eqs)
        passed = score >= 0.5
        return UnitTestResult(
            "equation_accuracy",
            passed,
            score,
            f"matched {int(score * len(ref_eqs))}/{len(ref_eqs)} equations",
        )

    def test_equation_syntax(self, predicted: str) -> UnitTestResult:
        errors = 0
        total = 0
        for env in self.EQUATION_ENVS:
            pattern = r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}"
            for match in re.finditer(pattern, predicted, re.DOTALL):
                total += 1
                body = match.group(0)
                if re.search(r"\\\\\s*$", body.rstrip()):
                    errors += 1
        inline = re.findall(r"\$([^$\n]+)\$", predicted)
        for math in inline:
            total += 1
            if math.count("$") % 2 == 1:
                errors += 1
        if total == 0:
            return UnitTestResult("equation_syntax", True, 1.0, "no equations to validate")
        score = 1.0 - (errors / total)
        return UnitTestResult(
            "equation_syntax",
            score >= 0.8,
            score,
            f"{errors}/{total} equation issues",
        )

    def test_table_structure(self, predicted: str, reference: str) -> UnitTestResult:
        pred_table = self._extract_table(predicted)
        ref_table = self._extract_table(reference)
        if not ref_table:
            return UnitTestResult("table_structure", True, 1.0, "no reference table")
        if not pred_table:
            return UnitTestResult("table_structure", False, 0.0, "no predicted table")
        ref_cols = ref_table["columns"]
        pred_cols = pred_table["columns"]
        col_score = min(pred_cols, ref_cols) / max(pred_cols, ref_cols)
        row_score = min(pred_table["rows"], ref_table["rows"]) / max(
            pred_table["rows"], ref_table["rows"]
        )
        score = 0.6 * col_score + 0.4 * row_score
        return UnitTestResult(
            "table_structure",
            score >= 0.6,
            score,
            f"col_score={col_score:.2f} row_score={row_score:.2f}",
        )

    def test_section_hierarchy(self, predicted: str, reference: str) -> UnitTestResult:
        pred_sections = [self._normalize(s) for s in self._extract_sections(predicted)]
        ref_sections = [self._normalize(s) for s in self._extract_sections(reference)]
        if not ref_sections:
            return UnitTestResult("section_hierarchy", True, 1.0, "no reference sections")
        if not pred_sections:
            return UnitTestResult("section_hierarchy", False, 0.0, "no predicted sections")
        matched = sum(1 for s in ref_sections if s in pred_sections)
        score = matched / len(ref_sections)
        return UnitTestResult(
            "section_hierarchy",
            score >= 0.5,
            score,
            f"matched {matched}/{len(ref_sections)} sections",
        )

    def test_citation_label_integrity(self, latex_source: str) -> UnitTestResult:
        cited = set(re.findall(r"\\cite\{([^}]+)\}", latex_source))
        cited = {c.strip() for entry in cited for c in entry.split(",")}
        labeled = set(re.findall(r"\\label\{([^}]+)\}", latex_source))
        used = cited | labeled
        if not used:
            return UnitTestResult(
                "citation_label_integrity", True, 1.0, "no citations or labels"
            )
        symmetric_diff = cited.symmetric_difference(labeled)
        score = 1.0 - (len(symmetric_diff) / max(len(used), 1))
        return UnitTestResult(
            "citation_label_integrity",
            len(symmetric_diff) == 0,
            score,
            f"unmatched: {sorted(symmetric_diff)[:5]}",
        )

    def test_cross_reference_validity(self, latex_source: str) -> UnitTestResult:
        refs = set()
        for pat in (r"\\ref\{([^}]+)\}", r"\\eqref\{([^}]+)\}", r"\\pageref\{([^}]+)\}"):
            refs.update(re.findall(pat, latex_source))
        labels = set(re.findall(r"\\label\{([^}]+)\}", latex_source))
        broken = refs - labels
        if not refs:
            return UnitTestResult("cross_reference_validity", True, 1.0, "no references used")
        score = (len(refs) - len(broken)) / len(refs)
        return UnitTestResult(
            "cross_reference_validity",
            not broken,
            score,
            f"broken refs: {sorted(broken)[:5]}",
        )

    def test_compilation_success(
        self,
        latex_source: str,
        compiler: str = "pdflatex",
        timeout: int = 30,
        passes: int = 2,
    ) -> UnitTestResult:
        """Compile the source with the requested engine and score the result.

        The actual compilation is delegated to
        :class:`structured_ocr.verification.compiler.LaTeXCompiler`
        which supports pdflatex/xelatex/lualatex, multi-pass builds,
        timeouts, and structured log error parsing. The test result is
        produced in the legacy :class:`UnitTestResult` format so it
        remains compatible with the existing
        :class:`RewardFunction` interface.

        Scoring:

        * 1.0 — the engine produced a PDF on the first try.
        * 0.5 — the engine binary was not on ``$PATH``; we cannot
          grade the source, so we mark it passed and let the
          training job continue (callers can flip this via
          :func:`set_compiler_required`).
        * 0.0 — the engine ran and failed, or the call timed out.
        """
        from structured_ocr.verification.compiler import (
            CompilationOutcome,
            LaTeXCompiler,
        )

        if not latex_source or not latex_source.strip():
            return UnitTestResult("compilation_success", False, 0.0, "empty source")
        engine = LaTeXCompiler(
            engine=compiler, timeout=float(timeout), passes=int(passes)
        )
        result = engine.compile_string(latex_source)
        if result.outcome == CompilationOutcome.SUCCESS:
            details = (
                f"engine={compiler} passes={result.passes} "
                f"elapsed={result.elapsed_seconds:.2f}s"
            )
            return UnitTestResult("compilation_success", True, 1.0, details)
        if result.outcome == CompilationOutcome.COMPILER_NOT_FOUND:
            passed = not _compiler_required()
            return UnitTestResult(
                "compilation_success", passed, 0.5, f"{compiler} not available"
            )
        if result.outcome == CompilationOutcome.TIMEOUT:
            return UnitTestResult(
                "compilation_success",
                False,
                0.0,
                f"{compiler} timed out after {timeout}s",
            )
        details = (
            f"engine={compiler} outcome={result.outcome.value} "
            f"returncode={result.returncode} "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )
        if result.errors:
            details += f" errors={len(result.errors)}"
        return UnitTestResult("compilation_success", False, 0.0, details)

    def test_visual_similarity(
        self, predicted: str, extracted_image: Optional[bytes]
    ) -> UnitTestResult:
        if extracted_image is None:
            return UnitTestResult(
                "visual_similarity", True, 0.5, "no reference image; skipping"
            )
        return UnitTestResult(
            "visual_similarity",
            False,
            0.5,
            "image comparison requires a downstream renderer; placeholder",
        )

    def test_semantic_coherence(self, predicted: str, reference: str) -> UnitTestResult:
        if not reference:
            return UnitTestResult("semantic_coherence", True, 0.5, "no reference; skipping")
        score = difflib.SequenceMatcher(
            a=self._normalize_text(predicted), b=self._normalize_text(reference)
        ).ratio()
        return UnitTestResult(
            "semantic_coherence",
            score >= 0.4,
            score,
            f"textual similarity={score:.2f}",
        )

    def _extract_equation_bodies(self, text: str) -> List[str]:
        bodies: List[str] = []
        for env in self.EQUATION_ENVS:
            for match in re.finditer(
                r"\\begin\{" + env + r"\*?\}(.*?)\\end\{" + env + r"\*?\}", text, re.DOTALL
            ):
                bodies.append(self._normalize_text(match.group(1)))
        bodies.extend(self._normalize_text(m) for m in re.findall(r"\$([^$\n]+)\$", text))
        return bodies

    def _extract_table(self, text: str) -> Optional[Dict[str, int]]:
        m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, re.DOTALL)
        if not m:
            return None
        body = m.group(0)
        col_spec = re.search(r"\\begin\{tabular\}\{([^\}]+)\}", body)
        columns = len(col_spec.group(1)) if col_spec else 0
        rows = body.count("\\\\") + (1 if body else 0)
        return {"columns": columns, "rows": rows}

    def _extract_sections(self, text: str) -> List[str]:
        return re.findall(r"\\(?:sub)*section\*?\{([^}]+)\}", text)

    def _normalize(self, s: str) -> str:
        return self._normalize_text(s)

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip().lower())
        s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", s)
        return s

    def _sequence_similarity(self, a: Sequence[str], b: Sequence[str]) -> float:
        set_a = {self._normalize_text(x) for x in a}
        set_b = {self._normalize_text(x) for x in b}
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        return intersection / max(len(set_a | set_b), 1)


class RewardFunction:
    """Compute weighted reward scores for predicted LaTeX documents."""

    def __init__(
        self,
        weights: Optional[RewardConfig] = None,
        framework: Optional[LaTeXUnitTestFramework] = None,
    ) -> None:
        self.weights = weights or RewardConfig()
        self.framework = framework or LaTeXUnitTestFramework()

    def compute(
        self,
        predicted: str,
        reference: str = "",
        extracted_image: Optional[bytes] = None,
    ) -> RewardResult:
        results = self.framework.run_tests(predicted, reference, extracted_image)
        components: Dict[str, float] = {}
        breakdown: Dict[str, str] = {}
        passed = 0
        for name, weight in self.weights.as_dict().items():
            shaped = self._shape_reward(results[name].score)
            components[name] = shaped * weight
            breakdown[name] = (
                f"score={results[name].score:.3f} "
                f"shaped={shaped:.3f} w={weight:.2f} "
                f"({results[name].details})"
            )
            if results[name].passed:
                passed += 1
        total = sum(components.values())
        return RewardResult(
            total_reward=total,
            components=components,
            breakdown=breakdown,
            passed_tests=passed,
            total_tests=len(components),
        )

    def batch_compute(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        images: Optional[Sequence[Optional[bytes]]] = None,
    ) -> List[RewardResult]:
        if images is None:
            images = [None] * len(predictions)
        return [
            self.compute(p, r, img) for p, r, img in zip(predictions, references, images)
        ]

    @staticmethod
    def _shape_reward(score: float) -> float:
        if score >= 1.0:
            return 1.0
        if score <= 0.0:
            return 0.0
        return max(0.0, min(1.0, math.tanh(score * 3.0)))

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

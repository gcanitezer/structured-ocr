"""Smoke test for the existing (refactored) reward function code paths."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from structured_ocr.training.reward_functions import (  # noqa: E402
    LaTeXUnitTestFramework,
    RewardFunction,
    RewardResult,
    UnitTestResult,
)
from structured_ocr.training.types import RewardConfig  # noqa: E402
from structured_ocr.verification.compiler import (  # noqa: E402
    CompilationOutcome,
    CompilationResult,
)


class RewardFunctionSmokeTests(unittest.TestCase):
    def test_unit_test_result_default(self):
        r = UnitTestResult(test_name="x", passed=True, score=1.0)
        self.assertEqual(r.details, "")

    def test_reward_weights_sum_to_one(self):
        w = RewardConfig()
        self.assertAlmostEqual(sum(w.as_dict().values()), 1.0, places=6)

    def test_reward_weights_validate_rejects_bad_sum(self):
        w = RewardConfig(equation_accuracy=0.5, equation_syntax=0.5)
        with self.assertRaises(ValueError):
            w.validate()

    def test_framework_run_tests_returns_nine_components(self):
        f = LaTeXUnitTestFramework()
        results = f.run_tests(predicted="\\section{x}", reference="\\section{x}")
        self.assertEqual(
            set(results.keys()),
            {
                "equation_accuracy",
                "equation_syntax",
                "table_structure",
                "section_hierarchy",
                "citation_label_integrity",
                "cross_reference_validity",
                "compilation_success",
                "visual_similarity",
                "semantic_coherence",
            },
        )

    def test_section_hierarchy_matching(self):
        f = LaTeXUnitTestFramework()
        r = f.test_section_hierarchy(
            predicted="\\section{Intro}\\section{Methods}",
            reference="\\section{Intro}\\section{Methods}",
        )
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.score, 1.0)

    def test_citation_label_integrity_clean(self):
        f = LaTeXUnitTestFramework()
        r = f.test_citation_label_integrity("no citations or labels")
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.score, 1.0)

    def test_cross_reference_validity_no_refs(self):
        f = LaTeXUnitTestFramework()
        r = f.test_cross_reference_validity("no references")
        self.assertTrue(r.passed)

    def test_visual_similarity_no_image(self):
        f = LaTeXUnitTestFramework()
        r = f.test_visual_similarity("x", None)
        self.assertAlmostEqual(r.score, 0.5)

    def test_reward_function_compute_shape(self):
        rf = RewardFunction(weights=RewardConfig())
        result = rf.compute(predicted="\\section{x}", reference="\\section{x}")
        self.assertIsInstance(result, RewardResult)
        self.assertEqual(set(result.components.keys()), set(RewardConfig().as_dict().keys()))
        self.assertGreaterEqual(result.total_reward, 0.0)
        self.assertLessEqual(result.total_reward, 1.0)

    def test_reward_function_batch_compute(self):
        rf = RewardFunction(weights=RewardConfig())
        results = rf.batch_compute(
            predictions=["\\section{x}", "no structure"],
            references=["\\section{x}", "\\section{x}"],
        )
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].total_reward, results[1].total_reward)

    def test_shape_reward_monotonic(self):
        shaper = RewardFunction._shape_reward
        self.assertAlmostEqual(shaper(0.0), 0.0)
        self.assertAlmostEqual(shaper(1.0), 1.0)
        self.assertGreater(shaper(0.5), 0.0)
        self.assertLess(shaper(0.5), 1.0)
        self.assertLess(shaper(0.2), shaper(0.8))

    def test_compilation_test_handles_compiler_not_found(self):
        """Backwards compat: compilation not found should give 0.5 score."""
        f = LaTeXUnitTestFramework()
        fake = CompilationResult(outcome=CompilationOutcome.COMPILER_NOT_FOUND)
        with patch(
            "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
            return_value=fake,
        ):
            r = f.test_compilation_success("x")
        self.assertAlmostEqual(r.score, 0.5)


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(RewardFunctionSmokeTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

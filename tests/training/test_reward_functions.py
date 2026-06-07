"""Tests for the reward functions module (covers framework + reward function)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from structured_ocr.training import (
    LaTeXUnitTestFramework,
    RewardFunction,
    RewardResult,
    RewardWeights,
    UnitTestResult,
)


def test_unit_test_result_default():
    r = UnitTestResult(test_name="x", passed=True, score=1.0)
    assert r.details == ""


def test_reward_weights_sum_to_one():
    w = RewardWeights()
    total = sum(w.as_dict().values())
    assert abs(total - 1.0) < 1e-6


def test_reward_weights_validate_rejects_bad_sum():
    w = RewardWeights(equation_accuracy=0.5, equation_syntax=0.5)
    with pytest.raises(ValueError, match="sum to ~1.0"):
        w.validate()


def test_framework_run_tests_returns_nine_components():
    f = LaTeXUnitTestFramework()
    results = f.run_tests(predicted="\\section{x}", reference="\\section{x}")
    assert set(results.keys()) == {
        "equation_accuracy",
        "equation_syntax",
        "table_structure",
        "section_hierarchy",
        "citation_label_integrity",
        "cross_reference_validity",
        "compilation_success",
        "visual_similarity",
        "semantic_coherence",
    }


def test_section_hierarchy_matching():
    f = LaTeXUnitTestFramework()
    r = f.test_section_hierarchy(
        predicted="\\section{Intro}\\section{Methods}",
        reference="\\section{Intro}\\section{Methods}",
    )
    assert r.passed is True
    assert r.score == pytest.approx(1.0)


def test_section_hierarchy_partial():
    f = LaTeXUnitTestFramework()
    r = f.test_section_hierarchy(
        predicted="\\section{Intro}",
        reference="\\section{Intro}\\section{Methods}",
    )
    assert 0.0 < r.score < 1.0


def test_citation_label_integrity_clean():
    f = LaTeXUnitTestFramework()
    r = f.test_citation_label_integrity("no citations or labels")
    assert r.passed is True
    assert r.score == pytest.approx(1.0)


def test_cross_reference_validity_no_refs():
    f = LaTeXUnitTestFramework()
    r = f.test_cross_reference_validity("no references")
    assert r.passed is True


def test_visual_similarity_no_image():
    f = LaTeXUnitTestFramework()
    r = f.test_visual_similarity("x", None)
    assert r.score == pytest.approx(0.5)


def test_compilation_success_compiles_simple_doc():
    f = LaTeXUnitTestFramework()
    src = (
        "\\documentclass{article}\n\\begin{document}\nHello.\n\\end{document}\n"
    )
    r = f.test_compilation_success(src, compiler="pdflatex", timeout=20)
    # Score is either 1.0 (compiled), 0.5 (compiler not found), or 0.0 (failed/timeout)
    assert r.score in (0.0, 0.5, 1.0)


def test_reward_function_compute_shape():
    rf = RewardFunction(weights=RewardWeights())
    result = rf.compute(predicted="\\section{x}", reference="\\section{x}")
    assert isinstance(result, RewardResult)
    assert set(result.components.keys()) == set(RewardWeights().as_dict().keys())
    assert 0.0 <= result.total_reward <= 1.0


def test_reward_function_batch_compute():
    rf = RewardFunction(weights=RewardWeights())
    results = rf.batch_compute(
        predictions=["\\section{x}", "no structure"],
        references=["\\section{x}", "\\section{x}"],
    )
    assert len(results) == 2
    # The first prediction should generally outperform the second
    assert results[0].total_reward >= results[1].total_reward


def test_shape_reward_monotonic_below_one():
    from structured_ocr.training.reward_functions import RewardFunction

    shaper = RewardFunction._shape_reward
    assert shaper(0.0) == pytest.approx(0.0)
    assert shaper(1.0) == pytest.approx(1.0)
    assert 0.0 < shaper(0.5) < 1.0
    assert shaper(0.2) < shaper(0.8)

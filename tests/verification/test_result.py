"""Tests for the structured_ocr.verification.result module."""

from __future__ import annotations

import json

import pytest

from structured_ocr.verification.compiler import CompilationOutcome, CompilationResult
from structured_ocr.verification.result import (
    ComponentResult,
    VerificationResult,
    VerificationSummary,
)


def _component(
    name: str, score: float, weight: float = 0.0, passed: bool = False
) -> ComponentResult:
    return ComponentResult(
        name=name,
        score=score,
        weight=weight,
        weighted_score=score * weight,
        passed=passed,
        details="",
    )


def test_component_result_to_dict():
    c = ComponentResult(name="x", score=0.7, weight=0.5, weighted_score=0.35, passed=True)
    d = c.to_dict()
    assert d == {
        "name": "x",
        "score": 0.7,
        "weight": 0.5,
        "weighted_score": 0.35,
        "passed": True,
        "details": "",
    }


def test_verification_result_pass_rate():
    result = VerificationResult(
        source="x",
        components=[
            _component("a", 1.0, passed=True),
            _component("b", 1.0, passed=True),
            _component("c", 0.0, passed=False),
        ],
    )
    assert result.total_components == 3
    assert result.passed_components == 2
    assert result.pass_rate == pytest.approx(2 / 3)
    assert result.passed is False


def test_verification_result_passes_only_when_all_pass():
    result = VerificationResult(
        source="x",
        components=[_component("a", 1.0, passed=True)],
    )
    assert result.passed is True


def test_verification_result_not_passed_with_no_components():
    result = VerificationResult(source="x", components=[])
    assert result.passed is False
    assert result.pass_rate == 0.0


def test_verification_result_components_by_name():
    result = VerificationResult(
        source="x",
        components=[_component("a", 1.0, passed=True), _component("b", 0.0)],
    )
    by_name = result.components_by_name()
    assert set(by_name.keys()) == {"a", "b"}
    assert result.get("a") is by_name["a"]
    assert result.get("missing") is None


def test_verification_result_to_dict_includes_compilation():
    compilation = CompilationResult(outcome=CompilationOutcome.SUCCESS, engine="pdflatex", passes=2)
    result = VerificationResult(
        source="x",
        components=[_component("compilation_success", 1.0, passed=True)],
        compilation=compilation,
        total_score=1.0,
        passed_components=1,
        total_components=1,
    )
    d = result.to_dict()
    assert d["compilation"]["outcome"] == "success"
    assert d["compilation"]["engine"] == "pdflatex"
    assert d["passed"] is True
    assert "timestamp" in d


def test_verification_result_to_dict_handles_no_compilation():
    result = VerificationResult(source="x", components=[])
    d = result.to_dict()
    assert d["compilation"] is None


def test_verification_summary_aggregates_batch(tmp_path):
    results = [
        VerificationResult(
            source=f"src-{i}",
            components=[
                _component("a", 1.0, weight=0.5, passed=True),
                _component("b", 0.0, weight=0.5, passed=False),
            ],
            total_score=0.5,
        )
        for i in range(4)
    ]
    summary = VerificationSummary(results=results)
    assert summary.num_documents == 4
    assert summary.batch_score == pytest.approx(0.5)
    assert summary.batch_pass_rate == pytest.approx(0.5)
    assert summary.component_averages["a"] == pytest.approx(1.0)
    assert summary.component_averages["b"] == pytest.approx(0.0)
    assert summary.component_pass_rates["a"] == pytest.approx(1.0)
    assert summary.component_pass_rates["b"] == pytest.approx(0.0)


def test_verification_summary_counts_compiled(tmp_path):
    results = [
        VerificationResult(
            source="x",
            compilation=CompilationResult(outcome=CompilationOutcome.SUCCESS, engine="pdflatex"),
            components=[_component("compilation_success", 1.0, passed=True)],
        ),
        VerificationResult(
            source="x",
            compilation=CompilationResult(outcome=CompilationOutcome.FAILED, engine="pdflatex"),
            components=[_component("compilation_success", 0.0)],
        ),
    ]
    summary = VerificationSummary(results=results)
    assert summary.num_compiled == 1
    assert summary.num_failed == 1


def test_verification_summary_empty():
    summary = VerificationSummary()
    assert summary.num_documents == 0
    assert summary.batch_score == 0.0
    assert summary.batch_pass_rate == 0.0


def test_verification_summary_to_dict_json_serializable():
    results = [
        VerificationResult(
            source="x",
            compilation=CompilationResult(outcome=CompilationOutcome.SUCCESS),
            components=[_component("a", 1.0, passed=True)],
        )
    ]
    summary = VerificationSummary(results=results)
    d = summary.to_dict()
    json.dumps(d)
    assert d["num_documents"] == 1
    assert d["num_compiled"] == 1

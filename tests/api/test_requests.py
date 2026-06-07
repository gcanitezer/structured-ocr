"""Tests for pydantic request/response models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from structured_ocr.api.models.requests import (
    EvaluateRequest,
    OCRRequest,
    TrainRequest,
    VerifyRequest,
)
from structured_ocr.api.models.responses import (
    EvaluateResponse,
    HealthResponse,
    OCRResponse,
    TrainResponse,
    TrainStatusResponse,
    VerifyResponse,
)


class TestOCRRequest:

    def test_defaults(self):
        req = OCRRequest()
        assert req.backend is None
        assert req.device is None
        assert req.max_new_tokens is None

    def test_custom_values(self):
        req = OCRRequest(backend="huggingface", device="cuda:0", max_new_tokens=512)
        assert req.backend == "huggingface"
        assert req.device == "cuda:0"
        assert req.max_new_tokens == 512

    def test_partial_override(self):
        req = OCRRequest(backend="pix2text")
        assert req.backend == "pix2text"
        assert req.device is None


class TestVerifyRequest:

    def test_required_latex(self):
        req = VerifyRequest(latex="E=mc^2")
        assert req.latex == "E=mc^2"

    def test_empty_latex(self):
        req = VerifyRequest(latex="")
        assert req.latex == ""

    def test_missing_latex(self):
        with pytest.raises(ValidationError):
            VerifyRequest()


class TestEvaluateRequest:

    def test_defaults(self):
        req = EvaluateRequest(
            predictions={"1": "a"}, references={"1": "b"}
        )
        assert req.predictions == {"1": "a"}
        assert req.references == {"1": "b"}
        assert req.model_name == "unknown"

    def test_custom_model_name(self):
        req = EvaluateRequest(
            predictions={"1": "a"}, references={"1": "b"}, model_name="custom"
        )
        assert req.model_name == "custom"

    def test_empty_dicts(self):
        req = EvaluateRequest(predictions={}, references={}, model_name="m")
        assert req.predictions == {}
        assert req.references == {}

    def test_serializes_to_json(self):
        req = EvaluateRequest(
            predictions={"1": "E=mc^2"}, references={"1": "E=mc^2"}, model_name="m"
        )
        data = json.loads(req.model_dump_json())
        assert data["predictions"]["1"] == "E=mc^2"
        assert data["model_name"] == "m"


class TestTrainRequest:

    def test_with_config(self):
        req = TrainRequest(config={"mode": "sft", "learning_rate": 1e-4})
        assert req.config == {"mode": "sft", "learning_rate": 1e-4}

    def test_empty_config(self):
        req = TrainRequest(config={})
        assert req.config == {}


class TestResponses:

    def test_health_response(self):
        resp = HealthResponse(status="ok", version="0.1.0")
        data = resp.model_dump()
        assert data == {"status": "ok", "version": "0.1.0"}

    def test_ocr_response(self):
        resp = OCRResponse(
            latex="x^2",
            confidence=0.95,
            processing_time_ms=10.5,
            model_name="Pix2Text",
            detected_elements={"formula": True},
        )
        data = resp.model_dump()
        assert data["latex"] == "x^2"
        assert data["confidence"] == 0.95
        assert data["processing_time_ms"] == 10.5
        assert data["model_name"] == "Pix2Text"
        assert data["detected_elements"] == {"formula": True}

    def test_ocr_response_serializes(self):
        resp = OCRResponse(
            latex="x^2", confidence=0.95, processing_time_ms=10.5,
            model_name="Pix2Text", detected_elements={},
        )
        raw = resp.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["latex"] == "x^2"

    def test_verify_response_minimal(self):
        resp = VerifyResponse(passed=True, total_score=1.0)
        data = resp.model_dump()
        assert data["passed"] is True
        assert data["total_score"] == 1.0
        assert data["compilation"] is None
        assert data["components"] is None
        assert data["errors"] is None

    def test_verify_response_full(self):
        resp = VerifyResponse(
            passed=False,
            total_score=0.0,
            compilation={"outcome": "failure", "elapsed_seconds": 0.5, "log_summary": "error"},
            components=[{"name": "syntax", "passed": False, "score": 0.0, "details": "fail"}],
            errors=["missing \\begin{document}"],
        )
        assert resp.compilation["outcome"] == "failure"
        assert len(resp.components) == 1
        assert len(resp.errors) == 1

    def test_evaluate_response(self):
        resp = EvaluateResponse(
            model_name="test",
            total_samples=10,
            avg_edit_distance=0.1,
            avg_similarity_ratio=0.9,
            avg_bleu=0.85,
            avg_section_f1=0.8,
            avg_table_f1=0.7,
            avg_equation_f1=0.9,
            avg_citation_f1=0.6,
            compilability_rate=1.0,
            avg_compilation_time=0.5,
            avg_image_similarity=0.88,
            avg_reference_integrity=0.95,
        )
        data = resp.model_dump()
        assert data["model_name"] == "test"
        assert data["total_samples"] == 10

    def test_train_response(self):
        resp = TrainResponse(job_id="abc-123", status="queued")
        assert resp.job_id == "abc-123"
        assert resp.status == "queued"

    def test_train_status_response(self):
        resp = TrainStatusResponse(job_id="abc-123", status="running", message="training started")
        data = resp.model_dump()
        assert data["job_id"] == "abc-123"
        assert data["status"] == "running"
        assert data["message"] == "training started"


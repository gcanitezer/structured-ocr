"""Tests for POST /ocr and /ocr/batch endpoints."""

from __future__ import annotations

import io

from PIL import Image
from fastapi.testclient import TestClient
from fastapi.testclient import TestClient

from structured_ocr.api.app import create_app
from structured_ocr.api.dependencies import get_inference_engine
from structured_ocr.data.types import OCRResult


def _make_test_image() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _make_non_image_bytes() -> bytes:
    return b"this is not an image"


class TestOCREndpoint:

    def test_ocr_success(self):
        app = create_app()
        client = TestClient(app)
        img_bytes = _make_test_image()

        mock_engine = OCRResult(
            latex="x^2", confidence=0.95, processing_time_ms=10.0,
            model_name="Pix2Text", detected_elements={},
        )
        class MockEngine:
            def infer(self, image, config=None):
                return mock_engine

        app.dependency_overrides[get_inference_engine] = lambda: MockEngine()  # type: ignore[assignment]
        try:
            response = client.post(
                "/ocr",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["latex"] == "x^2"
        assert data["confidence"] == 0.95
        assert data["model_name"] == "Pix2Text"

    def test_ocr_rejects_non_image(self):
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/ocr",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]

    def test_ocr_with_backend_override(self):
        app = create_app()
        client = TestClient(app)
        img_bytes = _make_test_image()

        class MockEngine:
            def infer(self, image, config=None):
                return OCRResult(
                    latex="E=mc^2", confidence=1.0, processing_time_ms=5.0,
                    model_name="HF", detected_elements={},
                )

        app.dependency_overrides[get_inference_engine] = lambda: MockEngine()
        try:
            response = client.post(
                "/ocr?backend=huggingface",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.json()["model_name"] == "HF"

    def test_ocr_with_corrupt_image(self):
        app = create_app()
        client = TestClient(app)
        corrupt = _make_non_image_bytes()
        response = client.post(
            "/ocr",
            files={"file": ("test.png", corrupt, "image/png")},
        )
        assert response.status_code == 400

    def test_ocr_no_file(self):
        app = create_app()
        client = TestClient(app)
        response = client.post("/ocr")
        assert response.status_code == 422

    def test_ocr_engine_error(self):
        app = create_app()
        client = TestClient(app)
        img_bytes = _make_test_image()

        class FailingEngine:
            def infer(self, image, config=None):
                raise RuntimeError("inference failed")

        app.dependency_overrides[get_inference_engine] = lambda: FailingEngine()
        try:
            response = client.post(
                "/ocr",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 500


class TestOCRBatchEndpoint:

    def test_ocr_batch_success(self):
        app = create_app()
        client = TestClient(app)
        img_bytes = _make_test_image()

        class BatchEngine:
            def infer_batch(self, images, config=None):
                return [
                    OCRResult(latex="a", confidence=1.0, processing_time_ms=5.0, model_name="M"),
                    OCRResult(latex="b", confidence=0.9, processing_time_ms=5.0, model_name="M"),
                ]
            def infer(self, image, config=None):
                return OCRResult(latex="", confidence=0.0, processing_time_ms=0.0, model_name="")

        app.dependency_overrides[get_inference_engine] = lambda: BatchEngine()
        try:
            response = client.post(
                "/ocr/batch",
                files=[
                    ("files", ("a.png", img_bytes, "image/png")),
                    ("files", ("b.png", img_bytes, "image/png")),
                ],
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["latex"] == "a"
        assert data[1]["latex"] == "b"

    def test_ocr_batch_rejects_non_image(self):
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/ocr/batch",
            files=[
                ("files", ("a.png", _make_test_image(), "image/png")),
                ("files", ("b.txt", b"hello", "text/plain")),
            ],
        )
        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]

    def test_ocr_batch_empty(self):
        app = create_app()
        client = TestClient(app)
        response = client.post("/ocr/batch")
        assert response.status_code == 422

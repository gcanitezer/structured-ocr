"""Tests for API app creation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from structured_ocr.api.app import create_app


def test_create_app_returns_fastapi():
    app = create_app()
    assert app.title == "Structured OCR API"
    assert app.version == "0.1.0"


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_cors_middleware_enabled():
    app = create_app()
    cors = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
    assert len(cors) == 1


def test_routers_registered():
    app = create_app()
    routes = [r.path for r in app.routes]
    assert "/ocr" in routes
    assert "/verify" in routes
    assert "/evaluate" in routes
    assert "/train" in routes
    assert "/health" in routes


def test_ocr_batch_route_registered():
    app = create_app()
    routes = [(r.path, list(r.methods)) for r in app.routes if hasattr(r, "methods")]
    ocr_batch = [p for p, m in routes if p == "/ocr/batch"]
    assert len(ocr_batch) == 1
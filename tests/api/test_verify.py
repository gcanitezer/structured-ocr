"""Tests for POST /verify endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from structured_ocr.api.app import create_app
from structured_ocr.api.dependencies import get_verifier


class TestVerifyEndpoint:

    def test_verify_success(self):
        app = create_app()
        client = TestClient(app)

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.passed = MagicMock(return_value=False)
        mock_result.passed = True
        mock_result.total_score = 0.95
        mock_result.compilation = None
        mock_result.components = []

        app.dependency_overrides[get_verifier] = lambda: mock_verifier  # type: ignore[assignment]
        mock_verifier.verify.return_value = mock_result
        try:
            response = client.post(
                "/verify",
                json={"latex": "E=mc^2"},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is True
        assert data["total_score"] == 0.95

    def test_verify_with_compilation(self):
        app = create_app()
        client = TestClient(app)

        mock_compilation = MagicMock()
        mock_compilation.outcome.value = "success"
        mock_compilation.elapsed_seconds = 1.5
        mock_compilation.errors = []
        mock_compilation.stderr = ""

        syntax = MagicMock()
        syntax.name = "syntax"
        syntax.passed = True
        syntax.score = 1.0
        syntax.details = "OK"

        structure = MagicMock()
        structure.name = "structure"
        structure.passed = True
        structure.score = 1.0
        structure.details = "OK"

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.total_score = 1.0
        mock_result.compilation = mock_compilation
        mock_result.components = [syntax, structure]

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = mock_result

        app.dependency_overrides[get_verifier] = lambda: mock_verifier  # type: ignore[assignment]
        try:
            response = client.post(
                "/verify",
                json={"latex": "\\documentclass{article}\\begin{document}\\end{document}"},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["compilation"]["outcome"] == "success"
        assert data["compilation"]["elapsed_seconds"] == 1.5
        assert len(data["components"]) == 2
        assert data["components"][0]["name"] == "syntax"

    def test_verify_empty_latex(self):
        app = create_app()
        client = TestClient(app)
        response = client.post("/verify", json={"latex": "   "})
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_verify_with_compilation_errors(self):
        app = create_app()
        client = TestClient(app)

        mock_compilation = MagicMock()
        mock_compilation.errors = ["Syntax error at line 1"]

        mock_result = MagicMock()
        mock_result.passed = False
        mock_result.total_score = 0.0
        mock_result.compilation = mock_compilation
        mock_result.components = []

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = mock_result

        app.dependency_overrides[get_verifier] = lambda: mock_verifier  # type: ignore[assignment]
        try:
            response = client.post(
                "/verify",
                json={"latex": "\\invalid"},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is False
        assert data["errors"] == ["Syntax error at line 1"]

    def test_verify_verifier_error(self):
        app = create_app()
        client = TestClient(app)

        mock_verifier = MagicMock()
        mock_verifier.verify.side_effect = RuntimeError("pdflatex not found")

        app.dependency_overrides[get_verifier] = lambda: mock_verifier  # type: ignore[assignment]
        try:
            response = client.post(
                "/verify",
                json={"latex": "E=mc^2"},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 500
        assert "pdflatex not found" in response.json()["detail"]

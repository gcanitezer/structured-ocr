"""Tests for POST /evaluate endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from structured_ocr.api.app import create_app


class TestEvaluateEndpoint:

    def test_evaluate_success(self):
        app = create_app()
        client = TestClient(app)

        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.model_name = "test_model"
        mock_result.total_samples = 2
        mock_result.avg_edit_distance = 0.1
        mock_result.avg_similarity_ratio = 0.9
        mock_result.avg_bleu = 0.85
        mock_result.avg_section_f1 = 0.8
        mock_result.avg_table_f1 = 0.7
        mock_result.avg_equation_f1 = 0.9
        mock_result.avg_citation_f1 = 0.6
        mock_result.compilability_rate = 1.0
        mock_result.avg_compilation_time = 0.5
        mock_result.avg_image_similarity = 0.88
        mock_result.avg_reference_integrity = 0.95

        with patch("structured_ocr.api.routers.evaluate.BenchmarkRunner", return_value=mock_runner):
            mock_runner.run.return_value = mock_result
            response = client.post(
                "/evaluate",
                json={
                    "predictions": {"1": "E=mc^2", "2": "x+y"},
                    "references": {"1": "E=mc^2", "2": "x+y"},
                    "model_name": "test_model",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test_model"
        assert data["total_samples"] == 2
        assert data["avg_edit_distance"] == 0.1
        assert data["avg_similarity_ratio"] == 0.9
        assert data["avg_bleu"] == 0.85
        assert data["compilability_rate"] == 1.0

    def test_evaluate_empty_predictions(self):
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/evaluate",
            json={"predictions": {}, "references": {"1": "x"}, "model_name": "m"},
        )
        assert response.status_code == 400
        assert "predictions cannot be empty" in response.json()["detail"]

    def test_evaluate_empty_references(self):
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/evaluate",
            json={"predictions": {"1": "x"}, "references": {}, "model_name": "m"},
        )
        assert response.status_code == 400
        assert "references cannot be empty" in response.json()["detail"]

    def test_evaluate_runner_error(self):
        app = create_app()
        client = TestClient(app)

        with patch("structured_ocr.api.routers.evaluate.BenchmarkRunner") as mock_cls:
            mock_runner = mock_cls.return_value
            mock_runner.run.side_effect = RuntimeError("benchmark crashed")

            response = client.post(
                "/evaluate",
                json={
                    "predictions": {"1": "a"},
                    "references": {"1": "a"},
                    "model_name": "m",
                },
            )

        assert response.status_code == 500
        assert "benchmark crashed" in response.json()["detail"]
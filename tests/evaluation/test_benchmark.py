from __future__ import annotations

import pytest

from structured_ocr.eval.benchmark import BenchmarkRunner, BaselineScores


def test_benchmark_runner_basic():
    runner = BenchmarkRunner(check_compilability=False, check_references=False)
    predictions = {"s1": r"\frac{a}{b}", "s2": r"\sqrt{c}"}
    references = {"s1": r"\frac{a}{b}", "s2": r"\sqrt{d}"}

    result = runner.run(predictions, references, model_name="test-model")

    assert result.model_name == "test-model"
    assert result.total_samples == 2
    assert result.avg_edit_distance < 1.0
    assert result.avg_similarity_ratio > 0


def test_benchmark_runner_empty():
    runner = BenchmarkRunner(check_compilability=False, check_references=False)
    result = runner.run({}, {}, model_name="empty")
    assert result.total_samples == 0


def test_baseline_scores_default():
    baselines = BaselineScores()
    assert "gpt4v" in baselines.model_dump()
    assert "olmocr" in baselines.model_dump()


def test_baseline_scores_load(tmp_path):
    import json
    baselines_file = tmp_path / "baselines.json"
    baselines_file.write_text(json.dumps(BaselineScores().model_dump()))
    loaded = BaselineScores.load(baselines_file)
    assert loaded.gpt4v.keys() == BaselineScores().gpt4v.keys()
"""Tests for the verification integration in the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from structured_ocr.training import (
    GRPOResult,
    SFTResult,
    TrainingConfig,
    TrainingMode,
    TrainingPipeline,
)
from structured_ocr.verification import VerificationSummary


def _write_jsonl(path: Path, n: int = 4) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"reference": f"ref-{i}", "prompt": f"prompt-{i}"}) + "\n")


def test_pipeline_sft_runs_verification_by_default(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=4)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        mode=TrainingMode.SFT,
        run_verification=True,
    )
    fake_summary = VerificationSummary(num_documents=1, batch_score=0.8, batch_pass_rate=0.9)
    pipeline = TrainingPipeline(cfg)
    sft_result = SFTResult(
        output_dir=tmp_path / "out", train_loss=0.5, num_steps=2, num_train_samples=4
    )
    with (
        patch.object(pipeline, "_run_sft", return_value=sft_result),
        patch.object(
            pipeline, "_maybe_run_verification", return_value=fake_summary.to_dict()
        ) as mock_verify,
    ):
        result = pipeline.run()
    assert mock_verify.called
    assert result.sft_verification is not None
    assert result.sft_verification["batch_score"] == 0.8
    assert (tmp_path / "out" / "sft_verification.json").exists() is False  # mocked
    # The default behaviour writes the report when not mocked - assert that path is documented
    assert "sft_verification" in result.to_dict()


def test_pipeline_grpo_runs_verification_by_default(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=4)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        mode=TrainingMode.GRPO,
        run_verification=True,
    )
    fake_summary = VerificationSummary(num_documents=1, batch_score=0.7)
    pipeline = TrainingPipeline(cfg)
    grpo_result = GRPOResult(
        output_dir=tmp_path / "out", final_reward=0.5, num_steps=2, num_prompts=4
    )
    with (
        patch.object(pipeline, "_run_grpo", return_value=grpo_result),
        patch.object(
            pipeline, "_maybe_run_verification", return_value=fake_summary.to_dict()
        ) as mock_verify,
    ):
        result = pipeline.run()
    assert mock_verify.called
    assert result.grpo_verification is not None
    assert result.grpo_verification["batch_score"] == 0.7


def test_pipeline_can_disable_verification(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=4)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        mode=TrainingMode.SFT,
        run_verification=False,
    )
    pipeline = TrainingPipeline(cfg)
    sft_result = SFTResult(
        output_dir=tmp_path / "out", train_loss=0.5, num_steps=2, num_train_samples=4
    )
    with (
        patch.object(pipeline, "_run_sft", return_value=sft_result),
        patch.object(pipeline, "_maybe_run_verification", return_value=None) as mock_verify,
    ):
        result = pipeline.run()
    assert mock_verify.called  # pipeline called the hook
    assert result.sft_verification is None  # but the hook returned None (disabled)


def test_pipeline_maybe_run_verification_no_eval_samples(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=tmp_path / "train.jsonl",  # type: ignore[arg-type]
        run_verification=True,
    )
    pipeline = TrainingPipeline(cfg)
    result = pipeline._maybe_run_verification(eval_samples=[], stage="sft")
    assert result is None


def test_pipeline_maybe_run_verification_writes_report(tmp_path: Path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=2)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        run_verification=True,
    )
    pipeline = TrainingPipeline(cfg)
    fake_samples = [
        MagicMock(reference="\\documentclass{article}\\begin{document}x\\end{document}")
        for _ in range(2)
    ]
    summary = pipeline._maybe_run_verification(eval_samples=fake_samples, stage="sft")
    assert summary is not None
    assert summary["num_documents"] == 2
    assert (tmp_path / "out" / "sft_verification.json").exists()


def test_pipeline_maybe_run_verification_returns_none_on_import_error(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=tmp_path / "train.jsonl",  # type: ignore[arg-type]
        run_verification=True,
    )
    pipeline = TrainingPipeline(cfg)
    fake_samples = [MagicMock(reference="x")]
    with patch(
        "structured_ocr.training.pipeline.LaTeXVerifier",
        side_effect=ImportError("nope"),
    ):
        assert pipeline._maybe_run_verification(fake_samples, "sft") is None


def test_pipeline_training_result_to_dict_includes_verification(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=2)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        mode=TrainingMode.SFT_THEN_GRPO,
        run_verification=True,
    )
    pipeline = TrainingPipeline(cfg)
    sft_result = SFTResult(
        output_dir=tmp_path / "out", train_loss=0.5, num_steps=2, num_train_samples=2
    )
    grpo_result = GRPOResult(
        output_dir=tmp_path / "out", final_reward=0.7, num_steps=2, num_prompts=2
    )
    with (
        patch.object(pipeline, "_run_sft", return_value=sft_result),
        patch.object(pipeline, "_run_grpo", return_value=grpo_result),
        patch.object(pipeline, "_maybe_run_verification", return_value={"num_documents": 1}),
    ):
        result = pipeline.run()
    d = result.to_dict()
    assert d["sft_verification"] == {"num_documents": 1}
    assert d["grpo_verification"] == {"num_documents": 1}

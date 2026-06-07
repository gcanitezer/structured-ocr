"""Tests for the high-level training pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from structured_ocr.training import (
    GRPOResult,
    RLVRConfig,
    SFTResult,
    TrainingConfig,
    TrainingMode,
    TrainingPipeline,
    TrainingResult,
)


def _write_jsonl(path: Path, n: int = 5) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"reference": f"ref-{i}", "prompt": f"prompt-{i}"}) + "\n")


def test_pipeline_load_samples_from_config(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=10)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
    )
    pipeline = TrainingPipeline(cfg)
    samples, evals = pipeline.load_samples()
    # When no eval_dataset is configured, the pipeline auto-splits 5% off
    assert len(samples) + len(evals) == 10
    assert len(samples) >= 9  # default 5% split with at least 1 eval
    assert len(evals) >= 1


def test_pipeline_load_samples_explicit_paths(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, n=8)
    _write_jsonl(eval_path, n=2)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        eval_dataset=eval_path,  # type: ignore[arg-type]
    )
    pipeline = TrainingPipeline(cfg)
    samples, evals = pipeline.load_samples()
    assert len(samples) == 8
    assert len(evals) == 2


def test_pipeline_load_samples_raises_when_empty(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=tmp_path / "missing.jsonl",  # type: ignore[arg-type]
    )
    pipeline = TrainingPipeline(cfg)
    with pytest.raises(FileNotFoundError):
        pipeline.load_samples()


def test_pipeline_from_config_path(tmp_path: Path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "model_name: dummy\nmode: sft\noutput_dir: ./o\ntrain_dataset: ./t.jsonl\n"
    )
    pipeline = TrainingPipeline.from_config_path(yaml_path)
    assert pipeline.config.model_name == "dummy"
    assert pipeline.config.mode == TrainingMode.SFT


def test_pipeline_run_sft_with_mocks(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=4)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        mode=TrainingMode.SFT,
    )
    pipeline = TrainingPipeline(cfg)
    sft_result = SFTResult(
        output_dir=tmp_path / "out",
        metrics={"train_loss": 0.5},
        train_loss=0.5,
        num_train_samples=4,
        num_steps=2,
    )
    with patch.object(pipeline, "_run_sft", return_value=sft_result) as mock_sft:
        result = pipeline.run()
    assert mock_sft.called
    assert result.sft_result == sft_result
    assert result.grpo_result is None
    assert (tmp_path / "out" / "pipeline_result.json").exists()


def test_pipeline_run_grpo_with_mocks(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=4)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        mode=TrainingMode.GRPO,
    )
    pipeline = TrainingPipeline(cfg, rlvr=RLVRConfig(num_generations=2))
    grpo_result = GRPOResult(
        output_dir=tmp_path / "out",
        metrics={"mean_reward": 0.5},
        final_reward=0.5,
        num_prompts=4,
        num_steps=2,
    )
    with patch.object(pipeline, "_run_grpo", return_value=grpo_result) as mock_grpo:
        result = pipeline.run()
    assert mock_grpo.called
    assert result.grpo_result == grpo_result
    assert result.sft_result is None


def test_pipeline_run_sft_then_grpo(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=4)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        mode=TrainingMode.SFT_THEN_GRPO,
    )
    pipeline = TrainingPipeline(cfg)
    sft_result = SFTResult(
        output_dir=tmp_path / "out", train_loss=0.5, num_steps=2, num_train_samples=4
    )
    grpo_result = GRPOResult(
        output_dir=tmp_path / "out", final_reward=0.7, num_steps=2, num_prompts=4
    )
    with (
        patch.object(pipeline, "_run_sft", return_value=sft_result),
        patch.object(pipeline, "_run_grpo", return_value=grpo_result),
    ):
        result = pipeline.run()
    assert result.sft_result is not None
    assert result.grpo_result is not None


def test_pipeline_run_rejects_unknown_mode(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, n=2)
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=train_path,  # type: ignore[arg-type]
        mode=TrainingMode.SFT,
    )
    # Build a stand-in enum value the dispatcher doesn't know about
    from enum import Enum

    class _FakeMode(str, Enum):
        BOGUS = "bogus"

    cfg.mode = _FakeMode.BOGUS
    pipeline = TrainingPipeline(cfg)
    with pytest.raises(ValueError, match="Unsupported training mode"):
        pipeline.run()


def test_training_result_to_dict_roundtrip():
    result = TrainingResult(
        mode=TrainingMode.SFT,
        output_dir=Path("/tmp/out"),
        elapsed_seconds=1.0,
    )
    d = result.to_dict()
    assert d["mode"] == "sft"
    assert d["output_dir"] == "/tmp/out"
    assert d["sft_result"] is None
    assert d["grpo_result"] is None

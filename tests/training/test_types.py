"""Tests for training configuration dataclasses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from structured_ocr.training import (
    LoRAConfig,
    QuantizationConfig,
    RewardConfig,
    TrainingConfig,
    TrainingMode,
)


def test_training_mode_enum_values():
    assert TrainingMode.SFT.value == "sft"
    assert TrainingMode.GRPO.value == "grpo"
    assert TrainingMode.SFT_THEN_GRPO.value == "sft_then_grpo"


def test_lora_config_defaults():
    cfg = LoRAConfig()
    assert cfg.enabled is False
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == pytest.approx(0.05)
    assert "q_proj" in cfg.target_modules
    assert cfg.use_qlora is False


def test_quantization_config_defaults():
    cfg = QuantizationConfig()
    assert cfg.enabled is False
    assert cfg.load_in_4bit is True
    assert cfg.load_in_8bit is False
    assert cfg.bnb_4bit_quant_type == "nf4"


def test_reward_config_defaults_sum_to_one():
    cfg = RewardConfig()
    total = sum(cfg.as_dict().values())
    assert abs(total - 1.0) < 1e-6


def test_reward_config_validates_sum():
    cfg = RewardConfig(equation_accuracy=0.5, equation_syntax=0.5)
    cfg.equation_accuracy = 0.4
    with pytest.raises(ValueError, match="sum to ~1.0"):
        cfg.validate()


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.mode == TrainingMode.SFT
    assert cfg.learning_rate > 0
    assert cfg.num_train_epochs >= 1
    assert isinstance(cfg.lora, LoRAConfig)
    assert isinstance(cfg.quantization, QuantizationConfig)
    assert isinstance(cfg.reward_weights, RewardConfig)


def test_training_config_path_coercion(tmp_path: Path):
    out = tmp_path / "out"
    train_ds = tmp_path / "train.jsonl"
    cfg = TrainingConfig(
        output_dir=out,  # type: ignore[arg-type]
        train_dataset=train_ds,  # type: ignore[arg-type]
    )
    assert isinstance(cfg.output_dir, Path)
    assert isinstance(cfg.train_dataset, Path)
    assert cfg.output_dir == out


def test_training_config_to_json_roundtrip(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=tmp_path / "train.jsonl",  # type: ignore[arg-type]
        mode=TrainingMode.GRPO,
    )
    payload = cfg.to_json()
    restored = TrainingConfig.from_json(payload)
    assert restored.mode == TrainingMode.GRPO
    assert restored.output_dir == cfg.output_dir


def test_training_config_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "model_name: foo\n"
        "mode: sft_then_grpo\n"
        "lora:\n  enabled: true\n  r: 8\n"
    )
    cfg = TrainingConfig.from_yaml(yaml_path)
    assert cfg.model_name == "foo"
    assert cfg.mode == TrainingMode.SFT_THEN_GRPO
    assert cfg.lora.enabled is True
    assert cfg.lora.r == 8


def test_training_config_dict_roundtrip():
    cfg = TrainingConfig()
    data = cfg.to_dict()
    restored = TrainingConfig.from_dict(data)
    assert restored.mode == cfg.mode
    assert restored.learning_rate == cfg.learning_rate

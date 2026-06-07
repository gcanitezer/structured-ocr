"""Tests for the GRPO trainer and RLVR configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from structured_ocr.training import (
    GRPOResult,
    GRPOTrainer,
    HAS_TRANSFORMERS,
    RLVRConfig,
    TrainingConfig,
    TrainingMode,
)
from structured_ocr.training.grpo_trainer import (
    REWARD_NAMES,
    _GroupedRewardScorer,
    _group_normalize,
    _safe_mean,
)
from structured_ocr.training.reward_functions import RewardFunction, RewardWeights


def test_rlvr_config_defaults():
    cfg = RLVRConfig()
    assert cfg.num_generations == 4
    assert cfg.kl_coef == pytest.approx(0.05)
    assert cfg.use_group_normalization is True
    assert cfg.max_new_tokens > 0


def test_rlvr_config_to_and_from_dict():
    cfg = RLVRConfig(num_generations=8, kl_coef=0.1, clip_ratio=0.3)
    restored = RLVRConfig.from_dict(cfg.to_dict())
    assert restored.num_generations == 8
    assert restored.kl_coef == pytest.approx(0.1)
    assert restored.clip_ratio == pytest.approx(0.3)


def test_group_normalize_zero_centered():
    rewards = [1.0, 3.0]
    advantages = _group_normalize(rewards)
    assert len(advantages) == 2
    # Sum to ~0
    assert abs(sum(advantages)) < 1e-6


def test_group_normalize_empty():
    assert _group_normalize([]) == []


def test_safe_mean():
    assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert _safe_mean([]) == 0.0


def test_scorer_returns_rewards_and_details():
    rf = RewardFunction(weights=RewardWeights())
    scorer = _GroupedRewardScorer(rf)
    rewards, details = scorer(
        prompts=["p1", "p2"],
        completions=["\\section{x}", "\\section{y}"],
        references=["\\section{x}", "\\section{y}"],
    )
    assert len(rewards) == 2
    assert all(r >= 0 for r in rewards)
    assert all(d.total_reward >= 0 for d in details)
    for d in details:
        assert set(d.components.keys()) == set(REWARD_NAMES)


def test_scorer_handles_compute_exception(monkeypatch):
    rf = RewardFunction(weights=RewardWeights())
    scorer = _GroupedRewardScorer(rf)
    # Force an exception path
    def _bad_compute(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(rf, "compute", _bad_compute)
    rewards, details = scorer(["p"], ["c"], ["r"])
    assert rewards == [0.0]
    assert details[0].total_reward == 0.0


def test_scorer_group_advantages():
    rf = RewardFunction(weights=RewardWeights())
    scorer = _GroupedRewardScorer(rf)
    advs = scorer.group_advantages([1.0, 2.0, 3.0, 4.0])
    assert len(advs) == 4
    assert abs(sum(advs)) < 1e-6


def test_grpo_result_to_dict(tmp_path: Path):
    result = GRPOResult(
        output_dir=tmp_path / "out",
        metrics={"mean_reward": 0.7},
        final_reward=0.7,
        num_prompts=10,
        num_steps=20,
    )
    d = result.to_dict()
    assert d["final_reward"] == 0.7
    assert d["num_prompts"] == 10
    assert isinstance(d["output_dir"], str)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers required")
def test_grpo_trainer_initializes(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "grpo",  # type: ignore[arg-type]
        train_dataset=tmp_path / "train.jsonl",  # type: ignore[arg-type]
        mode=TrainingMode.GRPO,
    )
    trainer = GRPOTrainer(cfg)
    assert trainer.output_dir.exists()
    assert trainer.uses_trl is False
    assert trainer.uses_lora is False
    assert trainer.uses_qlora is False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers required")
def test_grpo_trainer_uses_supplied_rlvr(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "grpo",  # type: ignore[arg-type]
        train_dataset=tmp_path / "train.jsonl",  # type: ignore[arg-type]
        mode=TrainingMode.GRPO,
    )
    rlvr = RLVRConfig(num_generations=8, kl_coef=0.2)
    trainer = GRPOTrainer(cfg, rlvr=rlvr)
    assert trainer.rlvr.num_generations == 8
    assert trainer.rlvr.kl_coef == pytest.approx(0.2)


def test_grpo_trainer_rejects_invalid_samples(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "grpo",  # type: ignore[arg-type]
        train_dataset=tmp_path / "train.jsonl",  # type: ignore[arg-type]
    )
    trainer = GRPOTrainer(cfg)
    with pytest.raises(ValueError, match="No training samples"):
        trainer.train([])

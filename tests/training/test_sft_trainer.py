"""Tests for the SFT trainer module (covers SFTDataset, _pad_collator, SFTTrainer metadata)."""

from __future__ import annotations

from pathlib import Path

import pytest

from structured_ocr.training import (
    HAS_TRANSFORMERS,
    LoRAConfig,
    QuantizationConfig,
    SFTDataset,
    SFTResult,
    TrainingConfig,
)
from structured_ocr.training.dataset_utils import PreparedSample
from structured_ocr.training.sft_trainer import _pad_collator  # noqa: F401


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    add_bos_token = False

    def __call__(self, text, add_special_tokens=True, truncation=False):
        # Trivial whitespace tokenizer that maps each word to a stable integer.
        tokens = text.split()
        return {"input_ids": [abs(hash(w)) % 10000 for w in tokens]}


def _sample(i: int) -> PreparedSample:
    return PreparedSample(
        prompt=f"prompt {i}",
        reference=f"answer {i}",
        image=None,
        metadata={},
    )


def test_sft_dataset_encode_masks_prompt():
    tok = _FakeTokenizer()
    ds = SFTDataset([_sample(0)], tokenizer=tok, max_seq_length=64, mask_prompt=True)
    encoded = ds.encode(0)
    assert "input_ids" in encoded
    assert "labels" in encoded
    assert "attention_mask" in encoded
    # The reference adds an EOS, so the answer portion grows by 1 token.
    n_prompt = len("prompt 0".split())
    n_answer = len("answer 0".split()) + 1  # +1 for EOS
    masked = sum(1 for x in encoded["labels"] if x == -100)
    assert masked == n_prompt
    # All non-masked labels are real token ids (not -100)
    assert sum(1 for x in encoded["labels"] if x != -100) == n_answer


def test_sft_dataset_no_mask_when_disabled():
    tok = _FakeTokenizer()
    ds = SFTDataset([_sample(0)], tokenizer=tok, max_seq_length=64, mask_prompt=False)
    encoded = ds.encode(0)
    assert -100 not in encoded["labels"]


def test_sft_dataset_truncates_long_sequences():
    tok = _FakeTokenizer()
    long_sample = PreparedSample(
        prompt="p",  # 1 token
        reference="a " * 50,  # 50 answer tokens
        image=None,
        metadata={},
    )
    ds = SFTDataset([long_sample], tokenizer=tok, max_seq_length=10, mask_prompt=True)
    encoded = ds.encode(0)
    assert len(encoded["input_ids"]) <= 10


def test_sft_dataset_raises_without_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        SFTDataset([_sample(0)], tokenizer=None)


def test_pad_collator_pads_to_max_length():
    feats = [
        {"input_ids": [1, 2], "labels": [10, 20], "attention_mask": [1, 1]},
        {"input_ids": [1], "labels": [11], "attention_mask": [1]},
    ]
    out = _pad_collator(feats, pad_token_id=0)
    assert out["input_ids"] == [[1, 2], [1, 0]]
    assert out["attention_mask"] == [[1, 1], [1, 0]]
    assert out["labels"] == [[10, 20], [11, -100]]


def test_sft_result_to_dict_serializable(tmp_path: Path):
    result = SFTResult(
        output_dir=tmp_path / "out",
        metrics={"train_loss": 0.5},
        train_loss=0.5,
        num_train_samples=10,
        num_steps=5,
        epochs=1.0,
        elapsed_seconds=12.0,
    )
    d = result.to_dict()
    assert d["train_loss"] == 0.5
    assert d["num_train_samples"] == 10
    assert isinstance(d["output_dir"], str)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers required")
def test_sft_trainer_estimates_steps(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "sft",  # type: ignore[arg-type]
        train_dataset=tmp_path / "train.jsonl",  # type: ignore[arg-type]
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
    )
    from structured_ocr.training import SFTTrainer

    trainer = SFTTrainer(cfg)
    steps = trainer.estimate_steps(16)
    # 16 / (2*4) = 2 per epoch * 2 epochs = 4
    assert steps == 4


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers required")
def test_sft_trainer_setup_creates_output_dir(tmp_path: Path):
    cfg = TrainingConfig(
        output_dir=tmp_path / "out",  # type: ignore[arg-type]
        train_dataset=tmp_path / "train.jsonl",  # type: ignore[arg-type]
    )
    from structured_ocr.training import SFTTrainer

    trainer = SFTTrainer(cfg)
    assert trainer.output_dir.exists()


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers required")
def test_sft_trainer_formatting_func():
    cfg = TrainingConfig(
        output_dir="/tmp/sft-out-doesnt-matter",  # type: ignore[arg-type]
        train_dataset="/tmp/sft-train-doesnt-matter",  # type: ignore[arg-type]
    )
    from structured_ocr.training import SFTTrainer

    trainer = SFTTrainer(cfg)
    fmt = trainer._formatting_func()
    out = fmt({"prompt": "P", "reference": "R"})
    assert "P" in out and "R" in out


def test_lora_and_quantization_attach_to_training_config():
    cfg = TrainingConfig()
    cfg.lora = LoRAConfig(enabled=True, r=8)
    cfg.quantization = QuantizationConfig(enabled=True, load_in_4bit=True)
    assert cfg.lora.r == 8
    assert cfg.quantization.load_in_4bit is True

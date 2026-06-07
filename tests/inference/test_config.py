"""Tests for InferConfig dataclass."""

from __future__ import annotations

from structured_ocr.inference.config import InferConfig


def test_infer_config_defaults():
    cfg = InferConfig()
    assert cfg.backend == "pix2text"
    assert cfg.device == "auto"
    assert cfg.max_new_tokens == 2048
    assert cfg.temperature == 0.1
    assert cfg.batch_size == 1
    assert cfg.timeout == 120
    assert cfg.model_name is None


def test_infer_config_custom_values():
    cfg = InferConfig(
        backend="huggingface",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda:0",
        max_new_tokens=1024,
        temperature=0.2,
        batch_size=4,
        timeout=60,
    )
    assert cfg.backend == "huggingface"
    assert cfg.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg.device == "cuda:0"
    assert cfg.max_new_tokens == 1024
    assert cfg.temperature == 0.2
    assert cfg.batch_size == 4
    assert cfg.timeout == 60


def test_infer_config_backend_case_sensitivity():
    cfg = InferConfig(backend="Pix2Text")
    assert cfg.backend == "Pix2Text"


def test_infer_config_partial_override():
    cfg = InferConfig(backend="huggingface", batch_size=8)
    assert cfg.backend == "huggingface"
    assert cfg.batch_size == 8
    assert cfg.device == "auto"
    assert cfg.temperature == 0.1


def test_infer_config_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(InferConfig)
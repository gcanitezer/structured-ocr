"""Tests for InferenceEngine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from structured_ocr.inference.config import InferConfig
from structured_ocr.inference.engine import InferenceEngine


@pytest.fixture
def mock_pix2text_backend():
    with patch("structured_ocr.inference.engine.Pix2TextBackend") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance, mock_cls


@pytest.fixture
def mock_hf_backend():
    with patch("structured_ocr.inference.engine.HFBackend") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance, mock_cls


def test_engine_default_config():
    engine = InferenceEngine()
    assert engine.config.backend == "pix2text"
    assert engine._backend is None


def test_engine_custom_config():
    cfg = InferConfig(backend="huggingface", device="cpu")
    engine = InferenceEngine(config=cfg)
    assert engine.config.backend == "huggingface"
    assert engine.config.device == "cpu"


def test_engine_get_backend_creates_pix2text(mock_pix2text_backend):
    instance, mock_cls = mock_pix2text_backend
    engine = InferenceEngine()
    backend = engine._get_backend()
    mock_cls.assert_called_once_with(
        model_name=None, device="auto", timeout=120
    )
    assert backend == instance
    assert engine._backend == instance


def test_engine_get_backend_creates_hf(mock_hf_backend):
    instance, mock_cls = mock_hf_backend
    cfg = InferConfig(backend="huggingface")
    engine = InferenceEngine(config=cfg)
    backend = engine._get_backend()
    mock_cls.assert_called_once_with(
        model_name=None, device="auto", max_new_tokens=2048, temperature=0.1, timeout=120
    )
    assert backend == instance


def test_engine_get_backend_caches(mock_pix2text_backend):
    instance, mock_cls = mock_pix2text_backend
    engine = InferenceEngine()
    b1 = engine._get_backend()
    b2 = engine._get_backend()
    assert b1 is b2
    mock_cls.assert_called_once()


def test_engine_unknown_backend():
    engine = InferenceEngine(config=InferConfig(backend="invalid"))
    with pytest.raises(ValueError, match="Unknown backend"):
        engine._get_backend()


def test_engine_infer_with_pix2text(mock_pix2text_backend):
    instance, mock_cls = mock_pix2text_backend
    instance.infer.return_value.latex = "x^2"
    instance.infer.return_value.confidence = 1.0
    instance.infer.return_value.processing_time_ms = 10.0

    engine = InferenceEngine()
    img = Image.new("RGB", (100, 30))
    result = engine.infer(img)
    instance.infer.assert_called_once_with(img)
    assert result.latex == "x^2"


def test_engine_infer_with_path(mock_pix2text_backend, tmp_path):
    instance, mock_cls = mock_pix2text_backend
    instance.infer.return_value.latex = "E=mc^2"

    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 30))
    img.save(img_path)

    engine = InferenceEngine()
    result = engine.infer(str(img_path))
    assert result.latex == "E=mc^2"


def test_engine_infer_with_pathlib(mock_pix2text_backend, tmp_path):
    instance, mock_cls = mock_pix2text_backend
    instance.infer.return_value.latex = "E=mc^2"

    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 30))
    img.save(img_path)

    engine = InferenceEngine()
    result = engine.infer(Path(img_path))
    assert result.latex == "E=mc^2"


def test_engine_infer_with_override_config(mock_pix2text_backend):
    instance, mock_cls = mock_pix2text_backend
    instance.infer.return_value.latex = "x^2"

    engine = InferenceEngine()
    override = InferConfig(device="cpu")
    img = Image.new("RGB", (100, 30))
    result = engine.infer(img, config=override)
    assert result.latex == "x^2"


def test_engine_infer_override_does_not_mutate(mock_pix2text_backend):
    instance, mock_cls = mock_pix2text_backend

    engine = InferenceEngine()
    original = engine.config
    override = InferConfig(device="cpu")

    img = Image.new("RGB", (100, 30))
    engine.infer(img, config=override)
    assert engine.config is original
    assert engine.config.device == "auto"


def test_engine_infer_batch(mock_pix2text_backend):
    instance, mock_cls = mock_pix2text_backend
    instance.infer.side_effect = [
        MagicMock(latex="a", confidence=1.0, processing_time_ms=5.0),
        MagicMock(latex="b", confidence=1.0, processing_time_ms=5.0),
    ]

    engine = InferenceEngine()
    imgs = [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]
    results = engine.infer_batch(imgs)
    assert len(results) == 2
    assert results[0].latex == "a"
    assert results[1].latex == "b"
    assert instance.infer.call_count == 2


def test_engine_infer_batch_empty(mock_pix2text_backend):
    instance, mock_cls = mock_pix2text_backend
    engine = InferenceEngine()
    results = engine.infer_batch([])
    assert results == []
    instance.infer.assert_not_called()


def test_engine_close(mock_pix2text_backend):
    instance, mock_cls = mock_pix2text_backend
    engine = InferenceEngine()
    engine._get_backend()
    engine.close()
    instance.close.assert_called_once()
    assert engine._backend is None


def test_engine_close_no_backend():
    engine = InferenceEngine()
    engine.close()


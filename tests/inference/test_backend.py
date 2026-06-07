"""Tests for Backend ABC and concrete backends."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from structured_ocr.inference.backend import Backend
from structured_ocr.inference.hf_backend import HFBackend
from structured_ocr.inference.pix2text_backend import Pix2TextBackend


def test_backend_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        Backend()


@pytest.fixture
def mock_pix2text_module():
    fake_module = types.ModuleType("pix2text")
    fake_module.Pix2Text = MagicMock()
    with patch.dict("sys.modules", {"pix2text": fake_module}):
        yield fake_module


@pytest.fixture
def mock_transformers_module():
    fake_module = types.ModuleType("transformers")
    fake_module.AutoModelForCausalLM = MagicMock()
    fake_module.AutoProcessor = MagicMock()
    with patch.dict("sys.modules", {"transformers": fake_module}):
        yield fake_module


class TestPix2TextBackend:

    def test_init_defaults(self):
        backend = Pix2TextBackend()
        assert backend.model_name == "Pix2Text"
        assert backend.device == "auto"
        assert backend.timeout == 120
        assert backend._engine is None

    def test_init_custom(self):
        backend = Pix2TextBackend(model_name="custom", device="cpu", timeout=30)
        assert backend.model_name == "custom"
        assert backend.device == "cpu"
        assert backend.timeout == 30

    def test_get_engine_import_error(self):
        backend = Pix2TextBackend()
        with patch.dict("sys.modules", {"pix2text": None}):
            with pytest.raises(ImportError, match="pix2text is required"):
                backend._get_engine()

    def test_get_engine_creates_and_caches(self, mock_pix2text_module):
        backend = Pix2TextBackend()
        engine1 = backend._get_engine()
        engine2 = backend._get_engine()
        assert engine1 is engine2
        mock_pix2text_module.Pix2Text.assert_called_once()

    def test_get_engine_passes_device(self, mock_pix2text_module):
        backend = Pix2TextBackend(device="cpu")
        backend._get_engine()
        mock_pix2text_module.Pix2Text.assert_called_once_with(device="cpu")

    def test_infer_with_string_result(self):
        backend = Pix2TextBackend()
        mock_engine = MagicMock()
        mock_engine.return_value = "\\frac{x}{y}"
        with patch.object(backend, "_get_engine", return_value=mock_engine):
            result = backend.infer(MagicMock())
        assert result.latex == "\\frac{x}{y}"
        assert result.confidence == 1.0
        assert result.model_name == "Pix2Text"

    def test_infer_with_dict_result(self):
        backend = Pix2TextBackend()
        mock_engine = MagicMock()
        mock_engine.return_value = {"text": "x^2", "confidence": 0.95, "elements": {"formula": 1}}
        with patch.object(backend, "_get_engine", return_value=mock_engine):
            result = backend.infer(MagicMock())
        assert result.latex == "x^2"
        assert result.confidence == 0.95
        assert result.detected_elements == {"formula": 1}

    def test_infer_with_list_result(self):
        backend = Pix2TextBackend()
        mock_engine = MagicMock()
        mock_engine.return_value = [
            {"text": "a", "confidence": 0.9},
            {"text": "b", "confidence": 0.8},
        ]
        with patch.object(backend, "_get_engine", return_value=mock_engine):
            result = backend.infer(MagicMock())
        assert result.latex == "a\nb"
        assert result.confidence == pytest.approx(0.85)

    def test_infer_with_exception(self):
        backend = Pix2TextBackend()
        mock_engine = MagicMock()
        mock_engine.side_effect = RuntimeError("GPU OOM")
        with patch.object(backend, "_get_engine", return_value=mock_engine):
            result = backend.infer(MagicMock())
        assert result.latex == ""
        assert result.confidence == 0.0
        assert len(result.warnings) == 1
        assert "GPU OOM" in result.warnings[0]

    def test_infer_confidence_capped_at_one(self):
        backend = Pix2TextBackend()
        mock_engine = MagicMock()
        mock_engine.return_value = {"text": "x", "confidence": 1.5}
        with patch.object(backend, "_get_engine", return_value=mock_engine):
            result = backend.infer(MagicMock())
        assert result.confidence == 1.0

    def test_close(self):
        backend = Pix2TextBackend()
        backend._engine = MagicMock()
        backend.close()
        assert backend._engine is None


class TestHFBackend:

    def test_init_defaults(self):
        backend = HFBackend()
        assert backend.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert backend.device == "auto"
        assert backend.max_new_tokens == 2048
        assert backend.temperature == 0.1
        assert backend.timeout == 120

    def test_init_custom(self):
        backend = HFBackend(
            model_name="custom/model",
            device="cuda:0",
            max_new_tokens=512,
            temperature=0.0,
            timeout=30,
        )
        assert backend.model_name == "custom/model"
        assert backend.device == "cuda:0"
        assert backend.max_new_tokens == 512
        assert backend.temperature == 0.0

    def test_load_model_import_error(self):
        backend = HFBackend()
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers is required"):
                backend._load_model()

    def test_load_model_creates_processor_and_model(self, mock_transformers_module):
        backend = HFBackend()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_transformers_module.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers_module.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        backend._load_model()
        assert backend._processor is mock_processor
        assert backend._model is mock_model

    def test_load_model_caches(self, mock_transformers_module):
        backend = HFBackend()
        backend._load_model()
        backend._load_model()
        mock_transformers_module.AutoProcessor.from_pretrained.assert_called_once()
        mock_transformers_module.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_load_model_with_explicit_device(self, mock_transformers_module):
        backend = HFBackend(device="cuda:0")
        backend._load_model()
        kwargs = mock_transformers_module.AutoModelForCausalLM.from_pretrained.call_args[1]
        assert kwargs["device_map"] == "cuda:0"

    def test_infer_success(self):
        backend = HFBackend()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor.tokenizer.eos_token_id = 2
        mock_model.device = "cpu"

        input_ids = MagicMock()
        input_ids.shape = [1, 5]
        proc_result = MagicMock()
        proc_result.to.return_value = {"input_ids": input_ids}
        mock_processor.return_value = proc_result

        generated_ids = MagicMock()
        generated_ids.shape = [1, 10]
        mock_model.generate.return_value = generated_ids
        mock_processor.tokenizer.decode.return_value = " \\boxed{42} "

        backend._processor = mock_processor
        backend._model = mock_model

        result = backend.infer(MagicMock())
        assert result.latex == "\\boxed{42}"
        assert result.confidence == 1.0
        assert result.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_infer_handles_exception(self):
        backend = HFBackend()
        with patch.object(backend, "_load_model", side_effect=RuntimeError("no GPU")):
            result = backend.infer(MagicMock())
        assert result.latex == ""
        assert result.confidence == 0.0
        assert len(result.warnings) == 1

    def test_infer_does_sample_when_temp_positive(self):
        backend = HFBackend(temperature=0.5)
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_processor.tokenizer.eos_token_id = 2

        input_ids = MagicMock()
        input_ids.shape = [1, 5]
        proc_result = MagicMock()
        proc_result.to.return_value = {"input_ids": input_ids}
        mock_processor.return_value = proc_result

        generated_ids = MagicMock()
        generated_ids.shape = [1, 10]
        mock_model.generate.return_value = generated_ids
        mock_processor.tokenizer.decode.return_value = "x"

        backend._processor = mock_processor
        backend._model = mock_model

        backend.infer(MagicMock())
        kwargs = mock_model.generate.call_args[1]
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.5

    def test_close(self):
        backend = HFBackend()
        backend._model = MagicMock()
        backend._processor = MagicMock()
        backend.close()
        assert backend._model is None
        assert backend._processor is None
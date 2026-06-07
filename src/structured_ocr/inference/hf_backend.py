"""Hugging Face Transformers backend for OCR inference."""

from __future__ import annotations

import logging
import time
from typing import Any

from structured_ocr.data.types import OCRResult
from structured_ocr.inference.backend import Backend

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are a LaTeX OCR system. Given an image of a mathematical expression, "
    "document, or formula, extract the content into valid LaTeX code. "
    "Output ONLY the LaTeX code without any explanation, markdown formatting, "
    "or triple backticks. Preserve the exact mathematical structure."
)


class HFBackend(Backend):
    """Backend using Hugging Face Transformers with a VLM (e.g. Qwen2.5-VL).

    Loads the model once and reuses it across inference calls. Serves as a
    fallback when Pix2Text is unavailable or when configured explicitly.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name or "Qwen/Qwen2.5-VL-7B-Instruct"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._model: Any = None
        self._processor: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            processor_kwargs: dict[str, Any] = {}
            model_kwargs: dict[str, Any] = {
                "torch_dtype": "auto",
                "trust_remote_code": True,
            }

            if self.device and self.device != "auto":
                model_kwargs["device_map"] = self.device
            else:
                model_kwargs["device_map"] = "auto"

            self._processor = AutoProcessor.from_pretrained(
                self.model_name, **processor_kwargs
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )
        except ImportError:
            raise ImportError(
                "transformers is required for HFBackend. "
                "Install with: pip install transformers"
            )

    def infer(self, image: Any) -> OCRResult:
        """Run VLM inference on an image to extract LaTeX."""
        self._load_model()
        start = time.time()

        try:
            inputs = self._processor(
                text=PROMPT_TEMPLATE,
                images=image,
                return_tensors="pt",
            ).to(self._model.device)

            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

            generated = generated_ids[:, inputs["input_ids"].shape[1]:]
            latex = self._processor.tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()

            elapsed = (time.time() - start) * 1000

            return OCRResult(
                latex=latex,
                confidence=1.0,
                processing_time_ms=elapsed,
                model_name=self.model_name,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return OCRResult(
                latex="",
                confidence=0.0,
                processing_time_ms=elapsed,
                model_name=self.model_name,
                warnings=[f"HFBackend inference failed: {e}"],
            )

    def close(self) -> None:
        self._model = None
        self._processor = None

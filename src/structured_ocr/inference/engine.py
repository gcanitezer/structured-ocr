"""Inference engine that orchestrates backends for OCR."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from structured_ocr.data.types import OCRResult
from structured_ocr.inference.backend import Backend
from structured_ocr.inference.config import InferConfig
from structured_ocr.inference.hf_backend import HFBackend
from structured_ocr.inference.pix2text_backend import Pix2TextBackend

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Orchestrates OCR inference via configurable backends.

    Supports single-image and batch inference. Falls back from Pix2Text
    to Hugging Face when the primary backend is unavailable.
    """

    def __init__(self, config: InferConfig | None = None) -> None:
        self.config = config or InferConfig()
        self._backend: Backend | None = None
        self._needs_init = True

    def _get_backend(self) -> Backend:
        if self._backend is not None:
            return self._backend
        backend = self._create_backend()
        self._backend = backend
        return backend

    def _create_backend(self) -> Backend:
        backend_name = self.config.backend.lower()
        if backend_name == "pix2text":
            try:
                return Pix2TextBackend(
                    model_name=self.config.model_name,
                    device=self.config.device,
                    timeout=self.config.timeout,
                )
            except ImportError:
                logger.warning(
                    "Pix2Text not available, falling back to Hugging Face backend"
                )
                return self._create_hf_backend()
        elif backend_name == "huggingface":
            return self._create_hf_backend()
        else:
            raise ValueError(
                f"Unknown backend '{backend_name}'. "
                "Supported: 'pix2text', 'huggingface'"
            )

    def _create_hf_backend(self) -> HFBackend:
        return HFBackend(
            model_name=self.config.model_name,
            device=self.config.device,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )

    def _load_image(self, image: Any) -> Any:
        if isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB")
        return image

    def infer(self, image: Any, config: InferConfig | None = None) -> OCRResult:
        """Run OCR inference on a single image.

        Args:
            image: PIL Image, numpy array, or path to an image file.
            config: Optional override config for this inference call.

        Returns:
            OCRResult with extracted LaTeX and metadata.
        """
        if config is not None:
            self.config = config
            self._backend = None

        backend = self._get_backend()
        loaded = self._load_image(image)
        return backend.infer(loaded)

    def infer_batch(
        self, images: Sequence[Any], config: InferConfig | None = None
    ) -> list[OCRResult]:
        """Run OCR inference on a batch of images.

        Args:
            images: Sequence of PIL Images, numpy arrays, or file paths.
            config: Optional override config for this batch.

        Returns:
            List of OCRResult objects, one per input image.
        """
        return [self.infer(img, config) for img in images]

    def close(self) -> None:
        """Release backend resources."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None
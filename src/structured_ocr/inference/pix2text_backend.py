"""Pix2Text backend for OCR inference."""

from __future__ import annotations

import logging
import time
from typing import Any

from structured_ocr.data.types import OCRResult
from structured_ocr.inference.backend import Backend

logger = logging.getLogger(__name__)


class Pix2TextBackend(Backend):
    """Backend using the Pix2Text library for layout detection + OCR.

    Handles layout detection, formula OCR, and table OCR separately,
    then merges results into a single LaTeX string.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name or "Pix2Text"
        self.device = device
        self.timeout = timeout
        self._engine: Any = None

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from pix2text import Pix2Text

            kwargs: dict[str, Any] = {}
            if self.device and self.device != "auto":
                kwargs["device"] = self.device
            if self.model_name and self.model_name != "Pix2Text":
                kwargs["model_name"] = self.model_name
            self._engine = Pix2Text(**kwargs)
        except ImportError:
            raise ImportError(
                "pix2text is required for Pix2TextBackend. "
                "Install with: pip install pix2text"
            )
        return self._engine

    def infer(self, image: Any) -> OCRResult:
        """Run Pix2Text inference on an image."""
        engine = self._get_engine()
        start = time.time()

        try:
            result = engine(image)
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return OCRResult(
                latex="",
                confidence=0.0,
                processing_time_ms=elapsed,
                model_name=self.model_name,
                warnings=[f"Pix2Text inference failed: {e}"],
            )

        elapsed = (time.time() - start) * 1000

        latex = ""
        detected_elements: dict[str, Any] = {}
        confidence = 0.0

        if isinstance(result, str):
            latex = result
            confidence = 1.0
        elif isinstance(result, dict):
            latex = result.get("text", result.get("latex", ""))
            confidence = float(result.get("confidence", 1.0))
            detected_elements = result.get("elements", {})
        elif hasattr(result, "text"):
            latex = result.text
            confidence = getattr(result, "confidence", 1.0)
            detected_elements = getattr(result, "elements", {})
        elif isinstance(result, list):
            parts = []
            confidences: list[float] = []
            for item in result:
                if isinstance(item, dict):
                    parts.append(item.get("text", item.get("latex", "")))
                    confidences.append(float(item.get("confidence", 1.0)))
                else:
                    parts.append(str(item))
                    confidences.append(1.0)
            latex = "\n".join(parts)
            confidence = sum(confidences) / len(confidences) if confidences else 1.0
        else:
            latex = str(result)

        return OCRResult(
            latex=latex,
            confidence=min(confidence, 1.0),
            processing_time_ms=elapsed,
            model_name=self.model_name,
            detected_elements=detected_elements,
        )

    def close(self) -> None:
        self._engine = None
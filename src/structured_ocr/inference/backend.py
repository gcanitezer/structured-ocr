"""Abstract base class for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from structured_ocr.data.types import OCRResult


class Backend(ABC):
    """Abstract base class for OCR inference backends."""

    @abstractmethod
    def infer(self, image: Any) -> OCRResult:
        """Run OCR inference on an image and return an OCRResult.

        Args:
            image: A PIL Image or numpy array representing the input image.

        Returns:
            OCRResult containing the extracted LaTeX and metadata.
        """
        ...

    def close(self) -> None:
        """Release any resources held by the backend."""
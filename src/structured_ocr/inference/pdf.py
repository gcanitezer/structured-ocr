"""PDF batch processing for OCR inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from PIL import Image

from structured_ocr.data.types import OCRResult
from structured_ocr.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


def extract_images_from_pdf(pdf_path: str | Path, dpi: int = 150) -> List[Image.Image]:
    """Extract pages from a PDF as PIL Images.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering pages (default 150).

    Returns:
        List of PIL Images, one per page.

    Raises:
        ImportError: If pdf2image is not installed.
        ValueError: If the PDF cannot be processed.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF processing. "
            "Install with: pip install pdf2image"
        )

    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info("Extracted %d pages from %s", len(images), pdf_path)
    return images


def batch_infer(images: List[Image.Image], engine: InferenceEngine) -> List[OCRResult]:
    """Run OCR inference on a list of PIL Images.

    Args:
        images: List of PIL Images to process.
        engine: Initialized InferenceEngine instance.

    Returns:
        List of OCRResult objects, one per page.
    """
    results = engine.infer_batch(images)
    for i, result in enumerate(results):
        result.page_number = i + 1
    return results
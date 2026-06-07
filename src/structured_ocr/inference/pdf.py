"""PDF batch processing for OCR inference."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

from structured_ocr.data.types import OCRResult
from structured_ocr.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


def extract_images_from_pdf(pdf_path: str | Path, dpi: int = 150) -> List[Path]:
    """Extract pages from a PDF as image files.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering pages (default 150).

    Returns:
        List of Paths to temporary image files, one per page.

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

    out_dir = Path(tempfile.mkdtemp(prefix="latexocr_pdf_"))
    images = convert_from_path(str(pdf_path), dpi=dpi)
    paths: list[Path] = []
    for i, img in enumerate(images):
        page_path = out_dir / f"page_{i + 1:04d}.png"
        img.save(str(page_path), "PNG")
        paths.append(page_path)

    logger.info("Extracted %d pages from %s", len(paths), pdf_path)
    return paths


def batch_infer(pdf_path: str | Path, engine: InferenceEngine) -> List[OCRResult]:
    """Run OCR inference on all pages of a PDF.

    Args:
        pdf_path: Path to the PDF file.
        engine: Initialized InferenceEngine instance.

    Returns:
        List of OCRResult objects, one per page.
    """
    images = extract_images_from_pdf(pdf_path)
    results = engine.infer_batch(images)
    for i, result in enumerate(results):
        result.page_number = i + 1
    return results
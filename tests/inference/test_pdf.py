"""Tests for PDF batch processing."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from structured_ocr.inference.pdf import batch_infer, extract_images_from_pdf


@pytest.fixture
def mock_pdf2image():
    fake_module = types.ModuleType("pdf2image")
    fake_module.convert_from_path = MagicMock()
    with patch.dict("sys.modules", {"pdf2image": fake_module}):
        yield fake_module


class TestExtractImagesFromPDF:

    def test_extract_images_success(self, mock_pdf2image):
        mock_images = [Image.new("RGB", (100, 50)), Image.new("RGB", (100, 50))]
        mock_pdf2image.convert_from_path.return_value = mock_images
        result = extract_images_from_pdf("/path/to/doc.pdf", dpi=200)
        mock_pdf2image.convert_from_path.assert_called_once_with("/path/to/doc.pdf", dpi=200)
        assert len(result) == 2
        assert result[0].size == (100, 50)

    def test_extract_images_import_error(self):
        with patch.dict("sys.modules", {"pdf2image": None}):
            with pytest.raises(ImportError, match="pdf2image is required"):
                extract_images_from_pdf("test.pdf")

    def test_extract_images_custom_dpi(self, mock_pdf2image):
        mock_images = [Image.new("RGB", (100, 50))]
        mock_pdf2image.convert_from_path.return_value = mock_images
        extract_images_from_pdf("test.pdf", dpi=300)
        mock_pdf2image.convert_from_path.assert_called_once_with("test.pdf", dpi=300)

    def test_extract_images_empty_pdf(self, mock_pdf2image):
        mock_pdf2image.convert_from_path.return_value = []
        result = extract_images_from_pdf("empty.pdf")
        assert result == []

    def test_extract_images_with_path_object(self, mock_pdf2image):
        mock_images = [Image.new("RGB", (100, 50))]
        mock_pdf2image.convert_from_path.return_value = mock_images
        pdf_path = Path("/path/to/file.pdf")
        extract_images_from_pdf(pdf_path)
        mock_pdf2image.convert_from_path.assert_called_once_with(str(pdf_path), dpi=150)


class TestBatchInfer:

    def test_batch_infer_assigns_page_numbers(self):
        mock_engine = MagicMock()
        mock_engine.infer_batch.return_value = [
            MagicMock(latex="a", page_number=0),
            MagicMock(latex="b", page_number=0),
        ]
        images = [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]
        results = batch_infer(images, mock_engine)
        assert len(results) == 2
        assert results[0].page_number == 1
        assert results[1].page_number == 2

    def test_batch_infer_empty(self):
        mock_engine = MagicMock()
        mock_engine.infer_batch.return_value = []
        results = batch_infer([], mock_engine)
        assert results == []

    def test_batch_infer_calls_infer_batch(self):
        mock_engine = MagicMock()
        mock_engine.infer_batch.return_value = []
        images = [Image.new("RGB", (10, 10))]
        batch_infer(images, mock_engine)
        mock_engine.infer_batch.assert_called_once_with(images)


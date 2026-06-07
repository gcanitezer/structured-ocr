"""Unit tests for data types."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from structured_ocr.data.types import (
    BenchmarkResult,
    BoundingBox,
    DocumentNode,
    DocumentStructure,
    EvaluationResult,
    OCRResult,
)


class TestBoundingBox(unittest.TestCase):
    def test_valid_bbox(self):
        bb = BoundingBox(0, 0, 100, 200)
        self.assertEqual(bb.width, 100)
        self.assertEqual(bb.height, 200)
        self.assertEqual(bb.area, 20000)

    def test_invalid_bbox_raises(self):
        with self.assertRaises(ValueError):
            BoundingBox(100, 0, 50, 200)

    def test_invalid_bbox_y_raises(self):
        with self.assertRaises(ValueError):
            BoundingBox(0, 100, 50, 50)


class TestDocumentStructure(unittest.TestCase):
    def test_empty_document(self):
        doc = DocumentStructure(raw_latex="")
        self.assertEqual(doc.page_count, 1)

    def test_sections_list(self):
        nodes = [DocumentNode("section", "A"), DocumentNode("section", "B")]
        doc = DocumentStructure(sections=nodes)
        self.assertEqual(len(doc.sections), 2)

    def test_section_hierarchy_valid(self):
        nodes = [
            DocumentNode("section", "Introduction"),
            DocumentNode("section", "Methods"),
            DocumentNode("section", "Results"),
        ]
        doc = DocumentStructure(sections=nodes, page_count=1)
        self.assertEqual(doc.page_count, 1)

    def test_tree_similarity_same(self):
        doc = DocumentStructure(raw_latex="A B C D E", sections=[DocumentNode("section", "A")])
        self.assertEqual(doc.compute_tree_similarity(doc), 1.0)

    def test_tree_similarity_different(self):
        d1 = DocumentStructure(raw_latex="A B C", sections=[DocumentNode("section", "A")])
        d2 = DocumentStructure(raw_latex="X Y Z", sections=[DocumentNode("section", "X")])
        self.assertEqual(d1.compute_tree_similarity(d2), 0.0)

    def test_structural_overlap(self):
        d1 = DocumentStructure(
            sections=[DocumentNode("section", "A"), DocumentNode("paragraph", "P1")]
        )
        d2 = DocumentStructure(
            sections=[DocumentNode("section", "A"), DocumentNode("paragraph", "P2")]
        )
        overlap = d1.structural_overlap(d2)
        self.assertIn("section", overlap)
        self.assertIn("paragraph", overlap)
        self.assertEqual(overlap["section"], 1)

    def test_empty_page_count(self):
        doc = DocumentStructure()
        self.assertEqual(doc.page_count, 1)


class TestOCRResultModel(unittest.TestCase):
    def test_valid_ocr_result(self):
        r = OCRResult(latex="\\section{Intro}", confidence=0.9, processing_time_ms=42.0)
        self.assertEqual(r.confidence, 0.9)

    def test_ocr_result_invalid_confidence_raises(self):
        with self.assertRaises(Exception):
            OCRResult(latex="test", confidence=1.5)

    def test_ocr_result_defaults(self):
        r = OCRResult()
        self.assertEqual(r.latex, "")
        self.assertEqual(r.confidence, 0.0)
        self.assertEqual(r.page_number, 1)


class TestEvaluationResultModel(unittest.TestCase):
    def test_passing_evaluation(self):
        r = EvaluationResult(
            test_name="t", passed=True, score=1.0, details="ok", metrics={"f1": 0.95}
        )
        self.assertTrue(r.passed)
        self.assertEqual(r.metrics["f1"], 0.95)

    def test_score_out_of_range_raises(self):
        with self.assertRaises(Exception):
            EvaluationResult(test_name="t", passed=True, score=1.5)


class TestBenchmarkResultModel(unittest.TestCase):
    def test_passing_benchmark(self):
        import datetime

        r = BenchmarkResult(
            "latency", 1200.0, "ms", 5000.0, True, datetime.datetime.now().isoformat()
        )
        self.assertTrue(r.passed)
        self.assertLess(r.value, r.threshold)

    def test_failure(self):
        import datetime

        r = BenchmarkResult("accuracy", 0.5, "f1", 0.7, False, datetime.datetime.now().isoformat())
        self.assertFalse(r.passed)

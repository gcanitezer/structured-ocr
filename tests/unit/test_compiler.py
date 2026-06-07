"""Unit tests for the real compiler (no stubs)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from structured_ocr.verification.compiler import (
    LaTeXCompiler, CompilationOutcome, CompilationResult,
    parse_log_errors, parse_log_warnings,
)
from structured_ocr.evaluation.structural import StructuralEvaluator


class TestLaTeXCompiler(unittest.TestCase):
    def test_compiler_initialization(self):
        c = LaTeXCompiler(engine="pdflatex", timeout=10)
        self.assertEqual(c.engine, "pdflatex")
        self.assertEqual(c.timeout, 10.0)

    def test_compiler_unsupported_engine_raises(self):
        with self.assertRaises(ValueError):
            LaTeXCompiler(engine="bogus")

    def test_compiler_invalid_timeout_raises(self):
        with self.assertRaises(ValueError):
            LaTeXCompiler(timeout=0)

    def test_compiler_invalid_passes_raises(self):
        with self.assertRaises(ValueError):
            LaTeXCompiler(passes=0)

    def test_empty_source_returns_empty_source_outcome(self):
        c = LaTeXCompiler(timeout=5)
        result = c.compile_string("")
        self.assertEqual(result.outcome, CompilationOutcome.EMPTY_SOURCE)

    def test_whitespace_only_source_returns_empty_source(self):
        c = LaTeXCompiler(timeout=5)
        result = c.compile_string("   \n  \t  ")
        self.assertEqual(result.outcome, CompilationOutcome.EMPTY_SOURCE)

    def test_is_available_returns_bool(self):
        c = LaTeXCompiler()
        self.assertIsInstance(c.is_available, bool)

    def test_compile_string_returns_compilation_result(self):
        c = LaTeXCompiler(timeout=5)
        result = c.compile_string("\\documentclass{article}\\begin{document}A\\end{document}")
        self.assertIsInstance(result, CompilationResult)
        self.assertIn(result.outcome, (CompilationOutcome.SUCCESS, CompilationOutcome.COMPILER_NOT_FOUND, CompilationOutcome.FAILED))

    def test_compilation_result_has_expected_attrs(self):
        c = LaTeXCompiler(timeout=5)
        result = c.compile_string("\\documentclass{article}\\begin{document}A\\end{document}")
        for attr in ("outcome", "returncode", "engine", "passes", "elapsed_seconds",
                     "errors", "warnings", "message"):
            self.assertTrue(hasattr(result, attr), f"missing attr {attr}")

    def test_compile_file_nonexistent(self):
        c = LaTeXCompiler(timeout=5)
        result = c.compile_file("/nonexistent/file.tex")
        self.assertEqual(result.outcome, CompilationOutcome.IO_ERROR)

    def test_compile_convenience_with_string(self):
        c = LaTeXCompiler(timeout=5)
        result = c.compile("\\documentclass{article}\\begin{document}B\\end{document}")
        self.assertIsInstance(result, CompilationResult)

    def test_to_dict_serializable(self):
        result = CompilationResult(outcome=CompilationOutcome.SUCCESS, engine="pdflatex")
        d = result.to_dict()
        self.assertEqual(d["outcome"], "success")
        self.assertEqual(d["engine"], "pdflatex")

    def test_succeeded_property(self):
        result = CompilationResult(outcome=CompilationOutcome.SUCCESS)
        self.assertTrue(result.succeeded)
        result = CompilationResult(outcome=CompilationOutcome.FAILED)
        self.assertFalse(result.succeeded)

    def test_score_property(self):
        self.assertEqual(CompilationResult(outcome=CompilationOutcome.SUCCESS).score, 1.0)
        self.assertEqual(CompilationResult(outcome=CompilationOutcome.FAILED).score, 0.0)
        self.assertEqual(CompilationResult(outcome=CompilationOutcome.COMPILER_NOT_FOUND).score, 0.5)


class TestParseLogFunctions(unittest.TestCase):
    def test_parse_log_errors_empty(self):
        self.assertEqual(parse_log_errors(""), [])

    def test_parse_log_warnings_empty(self):
        self.assertEqual(parse_log_warnings(""), [])

    def test_parse_log_errors_finds_errors(self):
        log = "! Undefined control sequence.\n! Emergency stop."
        errors = parse_log_errors(log)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Undefined control sequence" in e for e in errors))

    def test_parse_log_warnings_finds_warnings(self):
        log = "LaTeX Warning: No \\author given.\nLaTeX Warning: Citation undefined."
        warnings = parse_log_warnings(log)
        self.assertGreaterEqual(len(warnings), 2)


class TestStructuralEvaluator(unittest.TestCase):
    def setUp(self):
        self.eval = StructuralEvaluator()

    def test_section_f1_perfect(self):
        from structured_ocr.data.types import DocumentStructure as DS, DocumentNode as DN
        pred = DS(sections=[DN("section", "intro")])
        gold = DS(sections=[DN("section", "intro")])
        self.assertAlmostEqual(self.eval.compute_section_f1(pred, gold), 1.0)

    def test_section_f1_mismatch(self):
        from structured_ocr.data.types import DocumentStructure as DS, DocumentNode as DN
        pred = DS(sections=[DN("section", "intro")])
        gold = DS(sections=[DN("section", "different")])
        self.assertAlmostEqual(self.eval.compute_section_f1(pred, gold), 0.0)

    def test_detect_labels(self):
        labels = self.eval.extract_labels(r"\label{sec:a} \label{eq:b}")
        self.assertIn("sec:a", labels)
        self.assertIn("eq:b", labels)

    def test_detect_refs(self):
        refs = self.eval.extract_refs(r"\ref{sec:a} \eqref{eq:b} \autoref{fig:c} \Cref{tab:d}")
        self.assertIn("sec:a", refs)
        self.assertIn("fig:c", refs)
        self.assertIn("tab:d", refs)

    def test_table_structure_valid(self):
        res = self.eval.verify_table_structure(
            r"\begin{table}\centering\begin{tabular}{lcr}A\\\end{tabular}\end{table}"
        )
        self.assertEqual(res["table_count"], 1)
        self.assertEqual(res["tabular_count"], 1)

    def test_table_structure_no_table(self):
        res = self.eval.verify_table_structure("\\section{Text}\nNo tables here.")
        self.assertEqual(res["table_count"], 0)

    def test_reference_coverage_full(self):
        coverage = self.eval.evaluate_cross_reference_integrity(r"\label{a}\ref{a}")["reference_coverage"]
        self.assertEqual(coverage, 1.0)

    def test_hierarchy_valid(self):
        from structured_ocr.data.types import DocumentStructure as DS, DocumentNode as DN
        doc = DS(sections=[DN("section", "Introduction"), DN("section", "Methods")])
        res = self.eval.verify_section_hierarchy(doc)
        self.assertTrue(res["correct_order"])
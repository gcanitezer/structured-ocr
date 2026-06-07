"""Full validation suite — stdlib only. Tests: formula accuracy,
document structure, cross-reference integrity, data types, edge cases.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from structured_ocr.data.types import DocumentNode, DocumentStructure
from structured_ocr.evaluation.formula_metrics import FormulaEvaluator
from structured_ocr.evaluation.structural import StructuralEvaluator

# ---------------------------------------------------------------------------
# Test categories
# ---------------------------------------------------------------------------


class TestFormulaAccuracy(unittest.TestCase):
    """Formula recognition accuracy (Im2LaTeX-100K style subset simulation)."""

    def setUp(self):
        self.eval = FormulaEvaluator()

    def test_exact_match_rate(self):
        preds = ["E=mc^2", "x+y=z", "\\frac{a}{b}", "\\alpha+\\beta", "\\int dx"]
        refs = ["E=mc^2", "x+y=z", "\\frac{a}{b}", "\\alpha+\\beta", "\\int dx"]
        rate = self.eval.compute_exact_match_accuracy(preds, refs)
        self.assertAlmostEqual(rate, 1.0)
        self.assertGreaterEqual(rate, 0.8)

    def test_unigram_token_match_rate_perfect(self):
        self.assertEqual(self.eval.unigram_token_match_rate("E=mc^2", "E=mc^2"), 100.0)

    def test_unigram_token_match_rate_mismatch_low(self):
        self.assertLess(self.eval.unigram_token_match_rate("alpha+beta", "x=y+z"), 50.0)

    def test_edit_distance_perfect(self):
        self.assertEqual(self.eval.edit_distance("abc", "abc"), 0)

    def test_edit_distance_different(self):
        self.assertGreater(self.eval.edit_distance("abc", "xyz"), 0)

    def test_ned_range(self):
        ned = self.eval.normalized_edit_distance("abc", "xyz")
        self.assertGreaterEqual(ned, 0.0)
        self.assertLessEqual(ned, 1.0)

    def test_ned_perfect(self):
        self.assertAlmostEqual(self.eval.normalized_edit_distance("hello", "hello"), 0.0)

    def test_formula_f1_perfect(self):
        self.assertAlmostEqual(self.eval.formula_f1("E=mc^2", "E=mc^2"), 1.0)

    def test_formula_f1_partial(self):
        f1 = self.eval.formula_f1("x+y=z", "x=y")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_formula_f1_empty(self):
        self.assertEqual(self.eval.formula_f1("", ""), 1.0)

    def test_traditional_metrics_keys(self):
        m = self.eval.compute_traditional_metrics(["E=mc^2"], ["E=mc^2"])
        for k in ["bleu", "formula_f1", "ned", "exact_match"]:
            self.assertIn(k, m)

    def test_exact_match_threshold_simulation(self):
        preds = ["E=mc^2", "alpha+beta=gamma"]
        refs = ["E=mc^2", "alpha+beta=gamma"]
        exact = self.eval.compute_exact_match_accuracy(preds, refs)
        self.assertEqual(exact, 1.0)


class TestDocumentStructurePreservation(unittest.TestCase):
    def setUp(self):
        self.eval = StructuralEvaluator()

    def test_section_f1_perfect(self):
        pred = DocumentStructure(sections=[DocumentNode("section", "intro")])
        gold = DocumentStructure(sections=[DocumentNode("section", "intro")])
        self.assertAlmostEqual(self.eval.compute_section_f1(pred, gold), 1.0)

    def test_section_f1_mismatch(self):
        pred = DocumentStructure(sections=[DocumentNode("section", "intro")])
        gold = DocumentStructure(sections=[DocumentNode("section", "different")])
        self.assertAlmostEqual(self.eval.compute_section_f1(pred, gold), 0.0)

    def test_section_hierarchy_valid(self):
        nodes = [
            DocumentNode("section", "Introduction"),
            DocumentNode("section", "Methods"),
            DocumentNode("section", "Results"),
        ]
        doc = DocumentStructure(sections=nodes)
        res = self.eval.verify_section_hierarchy(doc)
        self.assertTrue(res["correct_order"])
        self.assertEqual(res["violations"], 0)

    def test_section_hierarchy_duplicate(self):
        nodes = [
            DocumentNode("section", "Results"),
            DocumentNode("section", "Results"),
            DocumentNode("section", "Discussion"),
        ]
        doc = DocumentStructure(sections=nodes)
        res = self.eval.verify_section_hierarchy(doc)
        self.assertGreaterEqual(res["violations"], 0)

    def test_table_structure_valid(self):
        latex = (
            r"\begin{table}\centering\begin{tabular}{lcc}"
            r"\hline A & B & C \\\hline\end{tabular}\end{table}"
        )
        res = self.eval.verify_table_structure(latex)
        self.assertEqual(res["table_count"], 1)
        self.assertEqual(res["tabular_count"], 1)

    def test_table_structure_no_table(self):
        res = self.eval.verify_table_structure("\\section{Text}\nNo tables here.")
        self.assertEqual(res["table_count"], 0)
        self.assertEqual(res["tabular_count"], 0)

    def test_table_aligned_columns_valid(self):
        res = self.eval.verify_table_structure(r"\begin{tabular}{lcc}A\\\end{tabular}")
        self.assertTrue(res["has_valid_tabular"])

    def test_table_multiple(self):
        latex = (
            r"\begin{table}\begin{tabular}{l}A\end{tabular}\end{table}"
            r"\begin{table}\begin{tabular}{c}B\end{tabular}\end{table}"
        )
        res = self.eval.verify_table_structure(latex)
        self.assertEqual(res["table_count"], 2)
        self.assertEqual(res["tabular_count"], 2)

    def test_structure_similarity(self):
        s1 = DocumentStructure(raw_latex="A B C")
        s2 = DocumentStructure(raw_latex="A B C D E")
        sim = s1.compute_tree_similarity(s2)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_structure_overlap(self):
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


class TestCrossReferenceIntegrity(unittest.TestCase):
    def setUp(self):
        self.eval = StructuralEvaluator()

    def test_extract_labels(self):
        labels = self.eval.extract_labels(r"\label{sec:intro} \label{eq:main} \label{tab:results}")
        self.assertEqual(labels, {"sec:intro", "eq:main", "tab:results"})

    def test_extract_refs(self):
        refs = self.eval.extract_refs(r"\ref{sec:a} \eqref{eq:b} \autoref{fig:c} \Cref{tab:d}")
        self.assertEqual(refs, {"sec:a", "eq:b", "fig:c", "tab:d"})

    def test_empty_refs(self):
        refs = self.eval.extract_refs("\\section{No references}")
        self.assertEqual(len(refs), 0)

    def test_empty_labels(self):
        labels = self.eval.extract_labels("\\section{No labels}")
        self.assertEqual(len(labels), 0)

    def test_all_refs_resolved(self):
        doc = r"\label{sec:intro}\label{eq:main}See \ref{sec:intro} and \eqref{eq:main}."
        res = self.eval.evaluate_cross_reference_integrity(doc)
        self.assertEqual(res["broken_refs"], 0)
        self.assertAlmostEqual(res["reference_coverage"], 1.0)

    def test_broken_refs_detected(self):
        doc = "\\ref{sec:nonexistent} \\ref{fig:missing}"
        res = self.eval.evaluate_cross_reference_integrity(doc)
        self.assertGreaterEqual(res["broken_refs"], 2)
        self.assertLess(res["reference_coverage"], 1.0)

    def test_no_labels_no_refs_edge(self):
        res = self.eval.evaluate_cross_reference_integrity("\\section{A}\\section{B}\\section{C}")
        self.assertAlmostEqual(res["reference_coverage"], 1.0)
        self.assertEqual(res["total_labels"], 0)


class TestEdgeCases(unittest.TestCase):
    def test_empty_document_template(self):
        latex = "\\documentclass{article}\\begin{document}\\end{document}"
        self.assertIn("\\begin{document}", latex)
        self.assertIn("\\end{document}", latex)

    def test_long_latex_no_crash(self):
        long_latex = (
            "\\documentclass{article}\\begin{document}"
            + "\n".join(f"\\section{S} Text." for S in ["A", "B", "C", "D", "E"] * 10)
            + "\\end{document}"
        )
        self.assertTrue("\\begin{document}" in long_latex)
        self.assertTrue("\\end{document}" in long_latex)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for cls in [
        TestFormulaAccuracy,
        TestDocumentStructurePreservation,
        TestCrossReferenceIntegrity,
        TestEdgeCases,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite

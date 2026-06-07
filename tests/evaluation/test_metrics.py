from __future__ import annotations

import pytest
from pathlib import Path

from structured_ocr.eval.metrics import (
    calculate_edit_distance,
    calculate_bleu,
    calculate_structural_metrics,
)


def test_edit_distance_identical():
    text = "Hello world"
    result = calculate_edit_distance(text, text)
    assert result["edit_distance"] == 0.0
    assert result["similarity_ratio"] == 1.0
    assert result["levenshtein_distance"] == 0


def test_edit_distance_different():
    pred = "Hello world"
    ref = "Hello there"
    result = calculate_edit_distance(pred, ref)
    assert result["edit_distance"] > 0
    assert result["similarity_ratio"] < 1.0


def test_bleu_identical():
    text = r"\frac{a}{b} + \sqrt{c}"
    result = calculate_bleu(text, text)
    assert result["bleu"] > 0.9


def test_bleu_different():
    pred = r"\frac{a}{b}"
    ref = r"\sqrt{c}"
    result = calculate_bleu(pred, ref)
    assert result["bleu"] < 0.5


def test_structural_metrics_sections():
    pred = r"\section{Introduction} \section{Methods}"
    ref = r"\section{Introduction} \section{Results} \section{Discussion}"
    result = calculate_structural_metrics(pred, ref)
    assert result.section_precision == 0.5
    assert result.section_recall == 1.0 / 3
    assert result.section_f1 > 0


def test_structural_metrics_equations():
    pred = r"\frac{a}{b} \int_0^1 x dx"
    ref = r"\frac{a}{b} \sqrt{c}"
    result = calculate_structural_metrics(pred, ref)
    assert result.equation_precision > 0


def test_structural_metrics_tables():
    pred = r"\begin{table} \begin{tabular}{a}\end{tabular}\end{table}"
    ref = r"\begin{table} \begin{tabular}{b}\end{tabular}\end{table}"
    result = calculate_structural_metrics(pred, ref)
    assert result.table_precision > 0


def test_structural_metrics_citations():
    pred = r"\cite{smith2020} \cite{jones2021}"
    ref = r"\cite{smith2020}"
    result = calculate_structural_metrics(pred, ref)
    assert result.citation_precision == 0.5
    assert result.citation_recall == 1.0
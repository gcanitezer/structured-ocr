from __future__ import annotations

import pytest

from structured_ocr.eval.references import ReferenceIntegrityChecker


def test_reference_checker_empty():
    checker = ReferenceIntegrityChecker()
    result = checker.check("")
    assert result.total_labels_defined == 0
    assert result.total_labels_used == 0
    assert result.label_integrity_score == 1.0
    assert result.citation_integrity_score == 1.0


def test_reference_checker_labels():
    checker = ReferenceIntegrityChecker()
    ref = r"\label{sec:intro} \ref{sec:intro} \ref{sec:missing}"
    result = checker.check(ref)
    assert result.total_labels_defined == 1
    assert result.total_labels_used == 2
    assert len(result.label_errors) == 1
    assert "Undefined label: sec:missing" in result.label_errors[0]


def test_reference_checker_citations():
    checker = ReferenceIntegrityChecker()
    ref = r"\cite{smith2020} \cite{jones2021}"
    result = checker.check(ref)
    assert result.total_citations == 2
    assert result.total_references == 0


def test_reference_checker_bibitems():
    checker = ReferenceIntegrityChecker()
    ref = r"\bibitem{smith2020} Smith et al. \cite{smith2020}"
    result = checker.check(ref)
    assert result.total_references == 1
    assert result.total_citations == 1
    assert result.citation_integrity_score == 1.0
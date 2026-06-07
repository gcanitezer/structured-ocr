from __future__ import annotations

import re
from typing import List, Set

from pydantic import BaseModel, Field


class ReferenceReport(BaseModel):
    label_errors: List[str] = Field(default_factory=list)
    citation_errors: List[str] = Field(default_factory=list)
    reference_errors: List[str] = Field(default_factory=list)

    total_labels_defined: int = 0
    total_labels_used: int = 0
    total_citations: int = 0
    total_references: int = 0

    label_integrity_score: float = 0.0
    citation_integrity_score: float = 0.0
    overall_integrity_score: float = 0.0


class ReferenceIntegrityChecker:
    def __init__(self):
        pass

    def check(self, latex_source: str, require_bib: bool = True) -> ReferenceReport:
        defined_labels = self._extract_labels(latex_source)
        used_labels = self._extract_label_usages(latex_source)
        citations = self._extract_citation_keys(latex_source)
        bib_items = self._extract_bib_items(latex_source)
        bib_file_refs = self._extract_bib_file_references(latex_source)

        label_errors = []
        for label in used_labels:
            if label not in defined_labels:
                label_errors.append(f"Undefined label: {label}")

        citation_errors = []
        available_keys = set(bib_items) | set(bib_file_refs)
        for cite_key in citations:
            if cite_key not in available_keys:
                citation_errors.append(f"Missing citation key: {cite_key}")

        ref_errors = []
        if require_bib and (citations or bib_items):
            if not bib_file_refs and not bib_items:
                ref_errors.append("Citations present but no bibliography defined")

        label_score = self._compute_label_score(defined_labels, used_labels)
        cite_score = self._compute_citation_score(citations, available_keys)
        overall = _weighted_average([label_score, cite_score], [0.5, 0.5])

        return ReferenceReport(
            label_errors=label_errors,
            citation_errors=citation_errors,
            reference_errors=ref_errors,
            total_labels_defined=len(defined_labels),
            total_labels_used=len(used_labels),
            total_citations=len(citations),
            total_references=len(bib_items) + len(bib_file_refs),
            label_integrity_score=label_score,
            citation_integrity_score=cite_score,
            overall_integrity_score=overall,
        )

    def _extract_labels(self, text: str) -> Set[str]:
        pattern = r"\\label\{([^}]*)\}"
        return set(re.findall(pattern, text))

    def _extract_label_usages(self, text: str) -> Set[str]:
        pattern = r"\\(?:ref|pageref|eqref)\{([^}]*)\}"
        return set(re.findall(pattern, text))

    def _extract_citation_keys(self, text: str) -> Set[str]:
        pattern = r"\\(?:cite|citestyle|citemanager)(?:\[.*?\])?\{([^}]*)\}"
        keys = set()
        for match in re.findall(pattern, text):
            for key in match.split(","):
                keys.add(key.strip())
        return keys

    def _extract_bib_items(self, text: str) -> Set[str]:
        pattern = r"\\bibitem(?:\[.*?\\])?\{([^}]*)\}"
        return set(re.findall(pattern, text))

    def _extract_bib_file_references(self, text: str) -> Set[str]:
        pattern = r"\\bibliography\{([^}]*)\}"
        return set(re.findall(pattern, text))

    def _compute_label_score(self, defined: Set[str], used: Set[str]) -> float:
        if not used:
            return 1.0
        correct = len(used & defined)
        return correct / len(used)

    def _compute_citation_score(self, cited: Set[str], available: Set[str]) -> float:
        if not cited:
            return 1.0
        correct = len(cited & available)
        return correct / len(cited)


def _weighted_average(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weights[: len(values)])
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def check_reference_integrity(latex_source: str) -> ReferenceReport:
    checker = ReferenceIntegrityChecker()
    return checker.check(latex_source)

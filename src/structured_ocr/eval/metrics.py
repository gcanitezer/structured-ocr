from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel


class StructuralMetricsResult(BaseModel):
    section_precision: float
    section_recall: float
    section_f1: float
    table_precision: float
    table_recall: float
    table_f1: float
    equation_precision: float
    equation_recall: float
    equation_f1: float
    citation_precision: float
    citation_recall: float
    citation_f1: float

    section_count: int = 0
    predicted_section_count: int = 0
    table_count: int = 0
    predicted_table_count: int = 0
    equation_count: int = 0
    predicted_equation_count: int = 0
    citation_count: int = 0
    predicted_citation_count: int = 0


def calculate_edit_distance(prediction: str, reference: str) -> Dict[str, float]:
    sm = SequenceMatcher(None, prediction, reference)
    ratio = sm.ratio()
    return {
        "edit_distance": 1 - ratio,
        "similarity_ratio": ratio,
        "levenshtein_distance": _levenshtein_distance(prediction, reference),
    }


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_bleu(prediction: str, reference: str, n_gram: int = 4) -> Dict[str, float]:
    pred_tokens = _tokenize_latex(prediction)
    ref_tokens = _tokenize_latex(reference)

    scores = {}
    for n in range(1, n_gram + 1):
        scores[f"bleu_{n}"] = _bleu_n_gram(pred_tokens, ref_tokens, n)

    scores["bleu"] = _geometric_mean([scores[f"bleu_{n}"] for n in range(1, n_gram + 1)])
    scores["bleu_modified"] = _modified_precision_bleu(pred_tokens, ref_tokens, n_gram)

    return scores


def _tokenize_latex(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    tokens = re.findall(r"\\[a-zA-Z]+|{[^{}]*}|[a-zA-Z0-9]+|[^\s]", text)
    return tokens


def _bleu_n_gram(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    pred_ngrams = [tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)]
    ref_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]

    if not pred_ngrams:
        return 0.0

    matches = sum(1 for ng in pred_ngrams if ng in ref_ngrams)
    return matches / len(pred_ngrams)


def _geometric_mean(values: List[float]) -> float:
    values = [max(v, 1e-10) for v in values]
    log_sum = sum(np.log(v) for v in values)
    return float(np.exp(log_sum / len(values)))


def _modified_precision_bleu(pred_tokens: List[str], ref_tokens: List[str], n_gram: int) -> float:
    scores = []
    for n in range(1, n_gram + 1):
        score = _bleu_n_gram(pred_tokens, ref_tokens, n)
        scores.append(score)

    bp = _brevity_penalty(len(pred_tokens), len(ref_tokens))
    return bp * _geometric_mean(scores)


def _brevity_penalty(pred_len: int, ref_len: int) -> float:
    if pred_len >= ref_len:
        return 1.0
    return float(np.exp(1 - ref_len / pred_len)) if pred_len > 0 else 0.0


def _extract_latex_commands(text: str, commands: List[str]) -> List[str]:
    pattern = r"\\(" + "|".join(commands) + r")(?:\{([^}]*)\})?"
    return re.findall(pattern, text)


def _extract_construct(text: str, start: str, end: str) -> List[str]:
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)
    return re.findall(pattern, text, re.DOTALL)


def _extract_labels(text: str) -> List[str]:
    pattern = r"\\label\{([^}]*)\}"
    return re.findall(pattern, text)


def _extract_citations(text: str) -> List[str]:
    pattern = r"\\(?:cite|bibitem)(?:\[.*?\])?\{([^}]*)\}"
    return [c for c in re.findall(pattern, text)]


def calculate_structural_metrics(prediction: str, reference: str) -> StructuralMetricsResult:
    pred_sections, ref_sections = _extract_sections(prediction, reference)
    pred_tables, ref_tables = _extract_tables(prediction, reference)
    pred_equations, ref_equations = _extract_equations(prediction, reference)
    pred_citations, ref_citations = _extract_citations_list(prediction, reference)

    sec_p, sec_r = _precision_recall(pred_sections, ref_sections)
    tab_p, tab_r = _precision_recall(pred_tables, ref_tables)
    eq_p, eq_r = _precision_recall(pred_equations, ref_equations)
    cit_p, cit_r = _precision_recall(pred_citations, ref_citations)

    return StructuralMetricsResult(
        section_precision=sec_p,
        section_recall=sec_r,
        section_f1=_f1_score(sec_p, sec_r),
        table_precision=tab_p,
        table_recall=tab_r,
        table_f1=_f1_score(tab_p, tab_r),
        equation_precision=eq_p,
        equation_recall=eq_r,
        equation_f1=_f1_score(eq_p, eq_r),
        citation_precision=cit_p,
        citation_recall=cit_r,
        citation_f1=_f1_score(cit_p, cit_r),
        section_count=len(ref_sections),
        predicted_section_count=len(pred_sections),
        table_count=len(ref_tables),
        predicted_table_count=len(pred_tables),
        equation_count=len(ref_equations),
        predicted_equation_count=len(pred_equations),
        citation_count=len(ref_citations),
        predicted_citation_count=len(pred_citations),
    )


def _extract_sections(prediction: str, reference: str) -> Tuple[set, set]:
    pred_sections = set(
        _extract_latex_commands(prediction, ["section", "subsection", "subsubsection", "chapter"])
    )
    ref_sections = set(
        _extract_latex_commands(reference, ["section", "subsection", "subsubsection", "chapter"])
    )
    return pred_sections, ref_sections


def _extract_tables(prediction: str, reference: str) -> Tuple[set, set]:
    pred_tables = set()
    ref_tables = set()

    for table_content in _extract_construct(prediction, "\\begin{table", "\\end{table}"):
        pred_tables.add(table_content[:50])
    for table_content in _extract_construct(reference, "\\begin{table", "\\end{table}"):
        ref_tables.add(table_content[:50])

    for tabular in _extract_construct(prediction, "\\begin{tabular", "\\end{tabular}"):
        pred_tables.add(tabular[:50])
    for tabular in _extract_construct(reference, "\\begin{tabular", "\\end{tabular}"):
        ref_tables.add(tabular[:50])

    return pred_tables, ref_tables


def _extract_equations(prediction: str, reference: str) -> Tuple[set, set]:
    pred_equations = set()
    ref_equations = set()

    for eq in _extract_construct(prediction, "\\begin{equation", "\\end{equation}"):
        pred_equations.add(eq[:100])
    for eq in _extract_construct(reference, "\\begin{equation", "\\end{equation}"):
        ref_equations.add(eq[:100])

    for eq in _extract_construct(prediction, "\\begin{align", "\\end{align}"):
        pred_equations.add(eq[:100])
    for eq in _extract_construct(reference, "\\begin{align", "\\end{align}"):
        ref_equations.add(eq[:100])

    for display_math in _extract_construct(prediction, "\\[", "\\]"):
        pred_equations.add(display_math[:100])
    for display_math in _extract_construct(reference, "\\[", "\\]"):
        ref_equations.add(display_math[:100])

    inline_eqs = _extract_latex_commands(
        prediction, ["frac", "sqrt", "sum", "int", "lim", "prod", "infty"]
    )
    inline_refs = _extract_latex_commands(
        reference, ["frac", "sqrt", "sum", "int", "lim", "prod", "infty"]
    )
    pred_equations.update(inline_eqs)
    ref_equations.update(inline_refs)

    return pred_equations, ref_equations


def _extract_citations_list(prediction: str, reference: str) -> Tuple[set, set]:
    return set(_extract_citations(prediction)), set(_extract_citations(reference))


def _precision_recall(pred: set, ref: set) -> Tuple[float, float]:
    prec = _precision(pred, ref)
    rec = _recall(pred, ref)
    return prec, rec


def _precision(pred: set, ref: set) -> float:
    if not pred:
        return 1.0 if not ref else 0.0
    return len(pred & ref) / len(pred)


def _recall(pred: set, ref: set) -> float:
    if not ref:
        return 1.0 if not pred else 0.0
    return len(pred & ref) / len(ref)


def _f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

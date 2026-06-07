from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark import BaselineScores, BenchmarkResult


def generate_report(
    result: BenchmarkResult,
    baseline: Optional[BaselineScores] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": result.model_name,
        "summary": {
            "total_samples": result.total_samples,
            "avg_edit_distance": round(result.avg_edit_distance, 4),
            "avg_similarity_ratio": round(result.avg_similarity_ratio, 4),
            "avg_bleu": round(result.avg_bleu, 4),
            "structural_f1": {
                "section": round(result.avg_section_f1, 4),
                "table": round(result.avg_table_f1, 4),
                "equation": round(result.avg_equation_f1, 4),
                "citation": round(result.avg_citation_f1, 4),
            },
            "compilability_rate": round(result.compilability_rate, 4),
            "avg_compilation_time_seconds": round(result.avg_compilation_time, 3),
            "avg_image_similarity": round(result.avg_image_similarity, 4),
            "avg_reference_integrity": round(result.avg_reference_integrity, 4),
        },
        "details": result.details,
    }

    if baseline:
        report["comparison"] = _compare_with_baseline(result, baseline)

    if model_config:
        report["model_config"] = model_config

    report["per_sample_analysis"] = _analyze_per_sample(result.per_sample)

    return report


def _compare_with_baseline(result: BenchmarkResult, baseline: BaselineScores) -> Dict[str, Any]:
    def get_comparison(metric: str, baseline_key: str) -> Dict[str, Any]:
        our_value = getattr(result, f"avg_{metric}", None)
        if our_value is None:
            our_value = getattr(result, metric, None)
        if our_value is None:
            return {"status": "no_data"}

        min(
            [
                ("gpt4v", baseline.gpt4v.get(baseline_key)),
                ("olmocr", baseline.olmocr.get(baseline_key)),
            ],
            key=lambda x: x[1] if x[1] is not None else float("inf"),
        )

        better_than = []
        if baseline.gpt4v.get(baseline_key) is not None:
            better = our_value > baseline.gpt4v[baseline_key]
            if metric in ["edit_distance"]:
                better = our_value < baseline.gpt4v[baseline_key]
            better_than.append(("GPT-4V", better))

        if baseline.olmocr.get(baseline_key) is not None:
            better = our_value > baseline.olmocr[baseline_key]
            if metric in ["edit_distance"]:
                better = our_value < baseline.olmocr[baseline_key]
            better_than.append(("olmOCR2", better))

        return {
            "our_score": round(our_value, 4),
            "baseline_scores": {
                "gpt4v": baseline.gpt4v.get(baseline_key),
                "olmocr": baseline.olmocr.get(baseline_key),
            },
            "better_than": better_than,
        }

    return {
        "transcription_fidelity": {
            "edit_distance": get_comparison("edit_distance", "edit_distance"),
            "similarity_ratio": get_comparison("similarity_ratio", "similarity_ratio"),
            "bleu": get_comparison("bleu", "bleu"),
        },
        "structural_faithfulness": {
            "section_f1": get_comparison("section_f1", "section_f1"),
            "table_f1": get_comparison("table_f1", "table_f1"),
            "equation_f1": get_comparison("equation_f1", "equation_f1"),
            "citation_f1": get_comparison("citation_f1", "citation_f1"),
        },
        "compilability": {
            "rate": get_comparison("compilability_rate", "compilability_rate"),
        },
    }


def _analyze_per_sample(per_sample: List[Any]) -> Dict[str, Any]:
    if not per_sample:
        return {"error": "No samples to analyze"}

    sims = [s.similarity_ratio for s in per_sample]
    edit_dists = [s.edit_distance for s in per_sample]

    return {
        "similarity_ratio": {
            "min": round(min(sims), 4),
            "max": round(max(sims), 4),
            "median": round(sorted(sims)[len(sims) // 2], 4),
        },
        "edit_distance": {
            "min": round(min(edit_dists), 4),
            "max": round(max(edit_dists), 4),
            "median": round(sorted(edit_dists)[len(edit_dists) // 2], 4),
        },
        "high_error_samples": [
            {"sample_id": s.sample_id, "edit_distance": s.edit_distance}
            for s in per_sample
            if s.edit_distance > 0.3
        ][:10],
        "low_compilability_samples": [
            {
                "sample_id": s.sample_id,
                "compilable": s.compilability.compilable if s.compilability else None,
            }
            for s in per_sample
            if s.compilability and not s.compilability.compilable
        ][:10],
    }


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

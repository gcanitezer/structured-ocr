from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .compilability import CompilabilityChecker, CompilabilityResult
from .metrics import (
    StructuralMetricsResult,
    calculate_bleu,
    calculate_edit_distance,
    calculate_structural_metrics,
)
from .references import ReferenceIntegrityChecker, ReferenceReport


@dataclass
class SingleResult:
    sample_id: str
    prediction: str
    reference: str
    image_path: Optional[Path]
    edit_distance: float
    similarity_ratio: float
    bleu: float
    structural: Optional[StructuralMetricsResult]
    compilability: Optional[CompilabilityResult]
    reference_integrity: Optional[ReferenceReport]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    model_name: str
    total_samples: int
    avg_edit_distance: float
    avg_similarity_ratio: float
    avg_bleu: float
    avg_section_f1: float
    avg_table_f1: float
    avg_equation_f1: float
    avg_citation_f1: float
    compilability_rate: float
    avg_compilation_time: float
    avg_image_similarity: float
    avg_reference_integrity: float
    per_sample: List[SingleResult] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    def __init__(self, check_compilability: bool = True, check_references: bool = True):
        self.check_compilability = check_compilability
        self.check_references = check_references
        self._compilability_checker = CompilabilityChecker() if check_compilability else None
        self._reference_checker = ReferenceIntegrityChecker() if check_references else None

    def run(
        self,
        predictions: Dict[str, str],
        references: Dict[str, str],
        image_paths: Optional[Dict[str, Path]] = None,
        model_name: str = "unknown",
    ) -> BenchmarkResult:
        per_sample = []
        total_edit_dist = 0.0
        total_sim = 0.0
        total_bleu = 0.0
        total_sec_f1 = 0.0
        total_tab_f1 = 0.0
        total_eq_f1 = 0.0
        total_cit_f1 = 0.0
        compilable_count = 0
        total_comp_time = 0.0
        total_img_sim = 0.0
        total_ref_int = 0.0

        sample_ids = set(predictions.keys()) & set(references.keys())

        for sid in sample_ids:
            pred = predictions[sid]
            ref = references[sid]

            edit_metrics = calculate_edit_distance(pred, ref)
            bleu_metrics = calculate_bleu(pred, ref)
            structural = calculate_structural_metrics(pred, ref)

            comp_result = None
            if self.check_compilability and pred.strip():
                comp_result = self._compilability_checker.check(pred)
                if comp_result.compilable:
                    compilable_count += 1
                    total_comp_time += comp_result.elapsed_seconds
                    if image_paths and sid in image_paths:
                        img_sim = self._compilability_checker.compare_rendered_images(
                            pred, image_paths[sid]
                        )
                        if img_sim is not None:
                            total_img_sim += img_sim

            ref_result = None
            if self.check_references and pred.strip():
                ref_result = self._reference_checker.check(pred)
                total_ref_int += ref_result.overall_integrity_score

            total_edit_dist += edit_metrics["edit_distance"]
            total_sim += edit_metrics["similarity_ratio"]
            total_bleu += bleu_metrics["bleu_modified"]
            total_sec_f1 += structural.section_f1
            total_tab_f1 += structural.table_f1
            total_eq_f1 += structural.equation_f1
            total_cit_f1 += structural.citation_f1

            per_sample.append(
                SingleResult(
                    sample_id=sid,
                    prediction=pred,
                    reference=ref,
                    image_path=image_paths.get(sid) if image_paths else None,
                    edit_distance=edit_metrics["edit_distance"],
                    similarity_ratio=edit_metrics["similarity_ratio"],
                    bleu=bleu_metrics["bleu_modified"],
                    structural=structural,
                    compilability=comp_result,
                    reference_integrity=ref_result,
                )
            )

        n = len(per_sample)
        if n == 0:
            return BenchmarkResult(
                model_name=model_name,
                total_samples=0,
                avg_edit_distance=1.0,
                avg_similarity_ratio=0.0,
                avg_bleu=0.0,
                avg_section_f1=0.0,
                avg_table_f1=0.0,
                avg_equation_f1=0.0,
                avg_citation_f1=0.0,
                compilability_rate=0.0,
                avg_compilation_time=0.0,
                avg_image_similarity=0.0,
                avg_reference_integrity=0.0,
                per_sample=[],
            )

        comp_rate = compilable_count / n if self.check_compilability else 0.0
        avg_comp_time = (total_comp_time / compilable_count) if compilable_count > 0 else 0.0
        img_count = sum(
            1 for s in per_sample if s.compilability and s.compilability.compilable and s.image_path
        )
        avg_img_sim = total_img_sim / img_count if img_count > 0 else 0.0
        ref_count = sum(1 for s in per_sample if s.reference_integrity)
        avg_ref_int = total_ref_int / ref_count if ref_count > 0 else 0.0

        return BenchmarkResult(
            model_name=model_name,
            total_samples=n,
            avg_edit_distance=total_edit_dist / n,
            avg_similarity_ratio=total_sim / n,
            avg_bleu=total_bleu / n,
            avg_section_f1=total_sec_f1 / n,
            avg_table_f1=total_tab_f1 / n,
            avg_equation_f1=total_eq_f1 / n,
            avg_citation_f1=total_cit_f1 / n,
            compilability_rate=comp_rate,
            avg_compilation_time=avg_comp_time,
            avg_image_similarity=avg_img_sim,
            avg_reference_integrity=avg_ref_int,
            per_sample=per_sample,
        )


def run_benchmark(
    predictions: Dict[str, str],
    references: Dict[str, str],
    image_paths: Optional[Dict[str, Path]] = None,
    model_name: str = "unknown",
) -> BenchmarkResult:
    runner = BenchmarkRunner()
    return runner.run(predictions, references, image_paths, model_name)


class BaselineScores(BaseModel):
    gpt4v: Dict[str, float] = {
        "edit_distance": 0.15,
        "similarity_ratio": 0.85,
        "bleu": 0.75,
        "section_f1": 0.80,
        "table_f1": 0.65,
        "equation_f1": 0.70,
        "citation_f1": 0.60,
        "compilability_rate": 0.90,
    }
    olmocr: Dict[str, float] = {
        "edit_distance": 0.18,
        "similarity_ratio": 0.82,
        "bleu": 0.70,
        "section_f1": 0.75,
        "table_f1": 0.60,
        "equation_f1": 0.68,
        "citation_f1": 0.55,
        "compilability_rate": 0.85,
    }

    @classmethod
    def load(cls, path: Path) -> "BaselineScores":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

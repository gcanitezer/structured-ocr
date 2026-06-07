from __future__ import annotations

from fastapi import APIRouter, HTTPException

from structured_ocr.api.models.requests import EvaluateRequest
from structured_ocr.api.models.responses import EvaluateResponse
from structured_ocr.eval.benchmark import BenchmarkRunner

router = APIRouter()


@router.post("", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    if not request.predictions:
        raise HTTPException(status_code=400, detail="predictions cannot be empty")
    if not request.references:
        raise HTTPException(status_code=400, detail="references cannot be empty")

    try:
        runner = BenchmarkRunner()
        result = runner.run(
            predictions=request.predictions,
            references=request.references,
            model_name=request.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

    return EvaluateResponse(
        model_name=result.model_name,
        total_samples=result.total_samples,
        avg_edit_distance=result.avg_edit_distance,
        avg_similarity_ratio=result.avg_similarity_ratio,
        avg_bleu=result.avg_bleu,
        avg_section_f1=result.avg_section_f1,
        avg_table_f1=result.avg_table_f1,
        avg_equation_f1=result.avg_equation_f1,
        avg_citation_f1=result.avg_citation_f1,
        compilability_rate=result.compilability_rate,
        avg_compilation_time=result.avg_compilation_time,
        avg_image_similarity=result.avg_image_similarity,
        avg_reference_integrity=result.avg_reference_integrity,
    )

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from structured_ocr.api.dependencies import get_verifier
from structured_ocr.api.models.requests import VerifyRequest
from structured_ocr.api.models.responses import VerifyResponse
from structured_ocr.verification import LaTeXVerifier

router = APIRouter()


@router.post("", response_model=VerifyResponse)
async def verify(
    request: VerifyRequest,
    verifier: LaTeXVerifier = Depends(get_verifier),
) -> VerifyResponse:
    if not request.latex.strip():
        raise HTTPException(status_code=400, detail="LaTeX source cannot be empty")

    try:
        result = verifier.verify(request.latex)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}")

    return VerifyResponse(
        passed=result.passed,
        total_score=result.total_score,
        compilation={
            "outcome": result.compilation.outcome.value,
            "elapsed_seconds": result.compilation.elapsed_seconds,
            "log_summary": "\n".join(result.compilation.errors) if result.compilation.errors else result.compilation.stderr[:500],
        }
        if result.compilation
        else None,
        components=[
            {
                "name": c.name,
                "passed": c.passed,
                "score": c.score,
                "details": c.details,
            }
            for c in (result.components or [])
        ],
        errors=[str(e) for e in (result.errors or [])],
    )

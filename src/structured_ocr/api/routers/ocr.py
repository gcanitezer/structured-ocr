from __future__ import annotations

import io
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from structured_ocr.api.dependencies import get_inference_engine
from structured_ocr.api.models.requests import OCRRequest
from structured_ocr.api.models.responses import OCRResponse
from structured_ocr.inference.config import InferConfig
from structured_ocr.inference.engine import InferenceEngine

router = APIRouter()


@router.post("", response_model=OCRResponse)
async def ocr(
    file: UploadFile = File(...),
    request: OCRRequest = Depends(),
    engine: InferenceEngine = Depends(get_inference_engine),
) -> OCRResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = _load_image(contents)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    config = _build_config(request)
    start = time.monotonic()
    try:
        result = engine.infer(image, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    elapsed = (time.monotonic() - start) * 1000

    return OCRResponse(
        latex=result.latex,
        confidence=result.confidence,
        processing_time_ms=elapsed,
        model_name=result.model_name or "unknown",
        detected_elements=result.detected_elements or {},
    )


@router.post("/batch", response_model=list[OCRResponse])
async def ocr_batch(
    files: list[UploadFile] = File(...),
    request: OCRRequest = Depends(),
    engine: InferenceEngine = Depends(get_inference_engine),
) -> list[OCRResponse]:
    images = []
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {f.filename} must be an image")
        contents = await f.read()
        images.append(_load_image(contents))

    config = _build_config(request)
    start = time.monotonic()
    results = engine.infer_batch(images, config=config)
    elapsed = (time.monotonic() - start) * 1000

    return [
        OCRResponse(
            latex=r.latex,
            confidence=r.confidence,
            processing_time_ms=elapsed / len(results),
            model_name=r.model_name or "unknown",
            detected_elements=r.detected_elements or {},
        )
        for r in results
    ]


def _build_config(request: OCRRequest) -> InferConfig | None:
    overrides = {}
    if request.backend is not None:
        overrides["backend"] = request.backend
    if request.device is not None:
        overrides["device"] = request.device
    if request.max_new_tokens is not None:
        overrides["max_new_tokens"] = request.max_new_tokens
    if not overrides:
        return None
    return InferConfig(**overrides)


def _load_image(contents: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(contents)).convert("RGB")

from __future__ import annotations

import uuid

from fastapi import APIRouter

from structured_ocr.api.models.requests import TrainRequest
from structured_ocr.api.models.responses import TrainResponse, TrainStatusResponse

router = APIRouter()

_jobs: dict[str, str] = {}


@router.post("", response_model=TrainResponse)
async def train(request: TrainRequest) -> TrainResponse:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = "queued"
    return TrainResponse(job_id=job_id, status="queued")


@router.get("/status/{job_id}", response_model=TrainStatusResponse)
async def train_status(job_id: str) -> TrainStatusResponse:
    status = _jobs.get(job_id, "unknown")
    return TrainStatusResponse(job_id=job_id, status=status)
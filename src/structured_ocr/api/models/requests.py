from __future__ import annotations

from pydantic import BaseModel


class OCRRequest(BaseModel):
    backend: str | None = None
    device: str | None = None
    max_new_tokens: int | None = None


class BatchOCRRequest(BaseModel):
    backend: str | None = None
    device: str | None = None
    max_new_tokens: int | None = None


class VerifyRequest(BaseModel):
    latex: str


class EvaluateRequest(BaseModel):
    predictions: dict[str, str]
    references: dict[str, str]
    model_name: str = "unknown"


class TrainRequest(BaseModel):
    config: dict
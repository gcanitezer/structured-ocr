from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class OCRResponse(BaseModel):
    latex: str
    confidence: float
    processing_time_ms: float
    model_name: str
    detected_elements: dict


class VerifyResponse(BaseModel):
    passed: bool
    total_score: float
    compilation: dict | None = None
    components: list[dict] | None = None
    errors: list[str] | None = None


class EvaluateResponse(BaseModel):
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


class TrainResponse(BaseModel):
    job_id: str
    status: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str = ""
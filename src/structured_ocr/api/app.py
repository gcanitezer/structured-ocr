from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from structured_ocr.api.routers import evaluate, ocr, train, verify


def create_app() -> FastAPI:
    app = FastAPI(
        title="Structured OCR API",
        version="0.1.0",
        description="REST API for LaTeX OCR inference, verification, and evaluation",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ocr.router, prefix="/ocr", tags=["ocr"])
    app.include_router(verify.router, prefix="/verify", tags=["verify"])
    app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluate"])
    app.include_router(train.router, prefix="/train", tags=["train"])

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    return app

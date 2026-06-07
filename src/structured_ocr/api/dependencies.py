from __future__ import annotations

import logging

from structured_ocr.inference.engine import InferenceEngine
from structured_ocr.verification import LaTeXVerifier

logger = logging.getLogger(__name__)

_engine: InferenceEngine | None = None
_verifier: LaTeXVerifier | None = None


def get_inference_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def get_verifier() -> LaTeXVerifier:
    global _verifier
    if _verifier is None:
        _verifier = LaTeXVerifier()
    return _verifier

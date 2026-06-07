"""Configuration for inference engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferConfig:
    """Configuration for the OCR inference engine."""

    backend: str = "pix2text"
    model_name: Optional[str] = None
    device: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    batch_size: int = 1
    timeout: int = 120

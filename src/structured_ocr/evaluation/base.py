from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from structured_ocr.data.types import EvaluationResult


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> EvaluationResult:
        raise NotImplementedError

    def _validate_input(self, predicted: Any, reference: Any) -> None:
        if predicted is None or reference is None:
            raise ValueError("Predicted and reference must not be None")

    def _clean_token(self, token: str) -> str:
        return (
            token.lower()
            .strip()
            .replace("\\ ", "\\")
            .replace("  ", " ")
            .strip("$")
            .replace("$", "")
        )
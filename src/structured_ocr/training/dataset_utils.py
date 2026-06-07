"""Dataset utilities for the LaTeX OCR training pipeline.

Supports loading from JSON / JSONL / CSV, converting the synthetic corpus
emitted by :mod:`structured_ocr.corpus.generator` into instruction-tuning
records consumable by TRL / HuggingFace SFTTrainer, and writing split
files for train / eval.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


INSTRUCTION_PROMPT = (
    "Transcribe the following document image into compilable LaTeX. "
    "Preserve the section hierarchy, equations, tables, and references."
)


@dataclass
class PreparedSample:
    """A single training / eval record ready for the trainer."""

    prompt: str
    reference: str
    image: Optional[str] = None
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "reference": self.reference,
            "image": self.image,
            "metadata": self.metadata,
        }


class DatasetUtils:
    """Helpers for loading, formatting, and splitting the training corpus."""

    SUPPORTED_FORMATS = {".json", ".jsonl", ".csv"}

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def load(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported dataset format: {suffix} "
                f"(supported: {sorted(self.SUPPORTED_FORMATS)})"
            )
        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "samples" in data:
                return list(data["samples"])
            return list(data)
        if suffix == ".jsonl":
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def prepare(
        self,
        records: Sequence[Dict[str, Any]],
        instruction: str = INSTRUCTION_PROMPT,
    ) -> List[PreparedSample]:
        """Convert raw records into :class:`PreparedSample` objects."""
        samples: List[PreparedSample] = []
        for record in records:
            reference = (
                record.get("reference")
                or record.get("latex")
                or record.get("output")
                or record.get("target")
                or ""
            )
            if not reference:
                logger.debug("Skipping record with no reference LaTeX")
                continue
            image = (
                record.get("image")
                or record.get("image_path")
                or record.get("image_uri")
            )
            prompt = record.get("prompt") or self._build_prompt(instruction, image, record)
            samples.append(
                PreparedSample(
                    prompt=prompt,
                    reference=reference,
                    image=image,
                    metadata={
                        k: v
                        for k, v in record.items()
                        if k not in {"reference", "latex", "output", "target", "image", "prompt"}
                    },
                )
            )
        return samples

    def prepare_from_file(
        self, path: Union[str, Path], instruction: str = INSTRUCTION_PROMPT
    ) -> List[PreparedSample]:
        return self.prepare(self.load(path), instruction)

    def split(
        self,
        samples: Sequence[PreparedSample],
        eval_ratio: float = 0.05,
        shuffle: bool = True,
    ) -> Tuple[List[PreparedSample], List[PreparedSample]]:
        """Split samples into train / eval partitions."""
        items = list(samples)
        if shuffle:
            self.rng.shuffle(items)
        if not items or eval_ratio <= 0:
            return items, []
        eval_count = max(1, int(len(items) * eval_ratio))
        eval_count = min(eval_count, max(1, len(items) - 1))
        return items[eval_count:], items[:eval_count]

    def write_jsonl(
        self, samples: Sequence[PreparedSample], path: Union[str, Path]
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        return path

    def write_hf_dataset(
        self, samples: Sequence[PreparedSample], output_dir: Union[str, Path]
    ) -> Path:
        """Write a HuggingFace :class:`Dataset` to ``output_dir``."""
        from datasets import Dataset

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ds = Dataset.from_list([s.to_dict() for s in samples])
        ds.save_to_disk(str(output_dir))
        return output_dir

    def statistics(self, samples: Sequence[PreparedSample]) -> Dict[str, Any]:
        if not samples:
            return {"count": 0}
        prompt_lens = [len(s.prompt) for s in samples]
        ref_lens = [len(s.reference) for s in samples]
        return {
            "count": len(samples),
            "avg_prompt_len": sum(prompt_lens) / len(prompt_lens),
            "avg_reference_len": sum(ref_lens) / len(ref_lens),
            "max_prompt_len": max(prompt_lens),
            "max_reference_len": max(ref_lens),
            "min_prompt_len": min(prompt_lens),
            "min_reference_len": min(ref_lens),
            "with_image": sum(1 for s in samples if s.image),
        }

    @staticmethod
    def _build_prompt(
        instruction: str, image: Optional[str], record: Dict[str, Any]
    ) -> str:
        parts = [instruction]
        if image:
            parts.append(f"[image: {image}]")
        template_name = record.get("template") or record.get("type")
        if template_name:
            parts.append(f"Document type: {template_name}")
        return "\n".join(parts)

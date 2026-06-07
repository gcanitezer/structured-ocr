"""High-level training pipeline orchestrator.

Combines :class:`DatasetUtils`, :class:`SFTTrainer`, and
:class:`GRPOTrainer` into a single, end-to-end training pipeline
that supports three modes:

* :attr:`TrainingMode.SFT` - supervised fine-tuning only.
* :attr:`TrainingMode.GRPO` - GRPO / RLVR fine-tuning only (requires
  a base model produced by a prior SFT stage or a pretrained model).
* :attr:`TrainingMode.SFT_THEN_GRPO` - run SFT, then continue with
  GRPO from the resulting checkpoint.

The pipeline is a convenience wrapper used by the CLI and the
``train_sft`` / ``train_grpo`` entry points.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .dataset_utils import DatasetUtils, PreparedSample
from .grpo_trainer import GRPOResult, GRPOTrainer, RLVRConfig
from .reward_functions import RewardFunction
from .sft_trainer import SFTResult, SFTTrainer
from .types import TrainingConfig, TrainingMode

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Aggregate result returned by :meth:`TrainingPipeline.run`."""

    mode: TrainingMode
    output_dir: Path
    sft_result: Optional[SFTResult] = None
    grpo_result: Optional[GRPOResult] = None
    elapsed_seconds: float = 0.0
    config_snapshot: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "output_dir": str(self.output_dir),
            "sft_result": self.sft_result.to_dict() if self.sft_result else None,
            "grpo_result": self.grpo_result.to_dict() if self.grpo_result else None,
            "elapsed_seconds": self.elapsed_seconds,
            "config_snapshot": self.config_snapshot,
            "extra": self.extra,
        }


class TrainingPipeline:
    """End-to-end training pipeline for the LaTeX OCR system.

    Parameters
    ----------
    config:
        :class:`TrainingConfig` describing model, data, optimizer, and
        training-mode settings.
    rlvr:
        Optional :class:`RLVRConfig` for the GRPO stage. Ignored if
        :attr:`TrainingConfig.mode` is :attr:`TrainingMode.SFT`.
    reward_function:
        Optional pre-built :class:`RewardFunction`. If not provided,
        one is constructed from ``config.reward_weights``.
    dataset_utils:
        Optional pre-built :class:`DatasetUtils`. If not provided,
        one is constructed with the seed from ``config``.
    """

    def __init__(
        self,
        config: TrainingConfig,
        rlvr: Optional[RLVRConfig] = None,
        reward_function: Optional[RewardFunction] = None,
        dataset_utils: Optional[DatasetUtils] = None,
    ) -> None:
        self.config = config
        self.rlvr = rlvr or RLVRConfig()
        self.reward_function = reward_function or RewardFunction(
            weights=config.reward_weights
        )
        self.dataset_utils = dataset_utils or DatasetUtils(seed=config.seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config_path(
        cls,
        config_path: Union[str, Path],
        rlvr: Optional[RLVRConfig] = None,
    ) -> "TrainingPipeline":
        config = TrainingConfig.from_yaml(config_path)
        return cls(config=config, rlvr=rlvr)

    def load_samples(
        self,
        train_path: Optional[Union[str, Path]] = None,
        eval_path: Optional[Union[str, Path]] = None,
    ) -> tuple[List[PreparedSample], List[PreparedSample]]:
        train_path = train_path or self.config.train_dataset
        samples = self.dataset_utils.prepare_from_file(train_path)
        eval_samples: List[PreparedSample] = []
        if eval_path is not None:
            eval_samples = self.dataset_utils.prepare_from_file(eval_path)
        elif self.config.eval_dataset is not None:
            eval_samples = self.dataset_utils.prepare_from_file(self.config.eval_dataset)
        if not eval_samples:
            samples, eval_samples = self.dataset_utils.split(samples)
        return samples, eval_samples

    def run(
        self,
        train_path: Optional[Union[str, Path]] = None,
        eval_path: Optional[Union[str, Path]] = None,
    ) -> TrainingResult:
        """Execute the configured training mode and return a result."""
        start = time.time()
        samples, eval_samples = self.load_samples(train_path, eval_path)
        if not samples:
            raise ValueError("No training samples found")
        logger.info(
            "Loaded %d train / %d eval samples; mode=%s",
            len(samples),
            len(eval_samples),
            self.config.mode.value,
        )
        sft_result: Optional[SFTResult] = None
        grpo_result: Optional[GRPOResult] = None
        if self.config.mode == TrainingMode.SFT:
            sft_result = self._run_sft(samples, eval_samples)
        elif self.config.mode == TrainingMode.GRPO:
            grpo_result = self._run_grpo(samples)
        elif self.config.mode == TrainingMode.SFT_THEN_GRPO:
            sft_result = self._run_sft(samples, eval_samples)
            self._switch_to_grpo_base_model(sft_result)
            grpo_result = self._run_grpo(samples)
        else:
            raise ValueError(f"Unsupported training mode: {self.config.mode}")
        elapsed = time.time() - start
        result = TrainingResult(
            mode=self.config.mode,
            output_dir=self.output_dir,
            sft_result=sft_result,
            grpo_result=grpo_result,
            elapsed_seconds=elapsed,
            config_snapshot=self.config.to_dict(),
        )
        self._write_result(result)
        return result

    def run_sft(
        self,
        samples: Sequence[PreparedSample],
        eval_samples: Optional[Sequence[PreparedSample]] = None,
    ) -> SFTResult:
        return self._run_sft(samples, eval_samples)

    def run_grpo(self, samples: Sequence[PreparedSample]) -> GRPOResult:
        return self._run_grpo(samples)

    def _run_sft(
        self,
        samples: Sequence[PreparedSample],
        eval_samples: Optional[Sequence[PreparedSample]],
    ) -> SFTResult:
        trainer = SFTTrainer(self.config)
        result = trainer.train(samples, eval_samples)
        return result

    def _run_grpo(self, samples: Sequence[PreparedSample]) -> GRPOResult:
        trainer = GRPOTrainer(
            self.config,
            rlvr=self.rlvr,
            reward_function=self.reward_function,
        )
        return trainer.train(samples)

    def _switch_to_grpo_base_model(self, sft_result: SFTResult) -> None:
        """Point the training config at the SFT output for the GRPO stage."""
        if sft_result is None or sft_result.output_dir is None:
            return
        ckpt = sft_result.output_dir
        if (ckpt / "adapter_model.bin").exists() or (ckpt / "adapter_model.safetensors").exists():
            self.config.model_name = str(ckpt)
        else:
            self.config.model_name = str(ckpt)

    def _write_result(self, result: TrainingResult) -> None:
        try:
            with open(self.output_dir / "pipeline_result.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write pipeline result: %s", exc)


__all__ = ["TrainingPipeline", "TrainingResult"]

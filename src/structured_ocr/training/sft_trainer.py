"""Supervised fine-tuning (SFT) trainer for the LaTeX OCR pipeline.

The trainer prefers TRL's :class:`SFTTrainer` when available, and falls
back to :class:`transformers.Trainer` with a manual tokenization
collation function when TRL is not installed. Both code paths support
LoRA / QLoRA via PEFT and bitsandbytes quantization when those
libraries are available, otherwise the full model is fine-tuned.

The trainer is designed to consume :class:`PreparedSample` records
produced by :class:`DatasetUtils` and to write reproducible
checkpoints, training metrics, and configuration snapshots.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .dataset_utils import PreparedSample
from .types import LoRAConfig, QuantizationConfig, TrainingConfig, TrainingMode

logger = logging.getLogger(__name__)

try:  # Optional integrations - keep module importable without TRL/PEFT
    import torch  # type: ignore
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore
except Exception:  # pragma: no cover - torch is a hard dependency
    torch = None  # type: ignore
    _TorchDataset = object  # type: ignore

try:
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False

try:
    from datasets import Dataset as _HFDataset  # type: ignore
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    from peft import LoraConfig as _PeftLoraConfig  # type: ignore
    from peft import PeftModel  # type: ignore
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    import bitsandbytes as bnb  # type: ignore  # noqa: F401
    HAS_BNB = True
except Exception:
    HAS_BNB = False

try:
    from trl import SFTConfig as _TRLSFTConfig  # type: ignore
    from trl import SFTTrainer as _TRLSFTTrainer  # type: ignore
    HAS_TRL = True
except Exception:
    HAS_TRL = False


@dataclass
class SFTResult:
    """Outcome of a supervised fine-tuning run."""

    output_dir: Path
    metrics: Dict[str, float] = field(default_factory=dict)
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    num_train_samples: int = 0
    num_eval_samples: int = 0
    num_steps: int = 0
    epochs: float = 0.0
    elapsed_seconds: float = 0.0
    checkpoint_paths: List[Path] = field(default_factory=list)
    config_snapshot: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d["checkpoint_paths"] = [str(p) for p in self.checkpoint_paths]
        return d


class SFTDataset:
    """In-memory dataset converting :class:`PreparedSample` to model inputs."""

    def __init__(
        self,
        samples: Sequence[PreparedSample],
        tokenizer: Any,
        max_seq_length: int = 2048,
        mask_prompt: bool = True,
        prompt_field: str = "prompt",
        reference_field: str = "reference",
    ) -> None:
        if tokenizer is None:
            raise ValueError("SFTDataset requires a tokenizer")
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.prompt_field = prompt_field
        self.reference_field = reference_field

    def __len__(self) -> int:
        return len(self.samples)

    def encode(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        prompt = getattr(sample, self.prompt_field)
        reference = getattr(sample, self.reference_field)
        tok = self.tokenizer
        prompt_ids = tok(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        answer_ids = tok(reference, add_special_tokens=False, truncation=False)["input_ids"]
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            answer_ids = answer_ids + [tok.eos_token_id]
        bos: List[int] = []
        if hasattr(tok, "bos_token_id") and tok.bos_token_id is not None and getattr(
            tok, "add_bos_token", False
        ):
            bos = [tok.bos_token_id]
        full = bos + prompt_ids + answer_ids
        if len(full) > self.max_seq_length:
            overflow = len(full) - self.max_seq_length
            if overflow < len(answer_ids):
                answer_ids = answer_ids[:-overflow]
            else:
                answer_ids = []
            full = bos + prompt_ids + answer_ids
        labels = list(full) if not self.mask_prompt else (
            [-100] * (len(bos) + len(prompt_ids)) + list(answer_ids)
        )
        attn = [1] * len(full)
        return {
            "input_ids": full,
            "labels": labels,
            "attention_mask": attn,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.encode(idx)


def _pad_collator(features: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    if not features:
        return {}
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids: List[List[int]] = []
    attn: List[List[int]] = []
    labels: List[List[int]] = []
    for f in features:
        pad = max_len - len(f["input_ids"])
        input_ids.append(list(f["input_ids"]) + [pad_token_id] * pad)
        attn.append(list(f["attention_mask"]) + [0] * pad)
        labels.append(list(f["labels"]) + [-100] * pad)
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }


class SFTTrainer:
    """High-level supervised fine-tuning orchestrator.

    Parameters
    ----------
    config:
        A :class:`TrainingConfig` describing model, dataset, optimizer,
        and training-stage options.
    """

    def __init__(self, config: TrainingConfig) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for SFTTrainer. "
                "Install with: pip install transformers>=4.35.0"
            )
        self.config = config
        self.tokenizer: Any = None
        self.model: Any = None
        self.trainer: Any = None
        self.train_dataset: Any = None
        self.eval_dataset: Any = None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_paths: List[Path] = []

    @property
    def uses_trl(self) -> bool:
        return bool(HAS_TRL)

    @property
    def uses_lora(self) -> bool:
        return bool(self.config.lora.enabled and HAS_PEFT)

    @property
    def uses_qlora(self) -> bool:
        return bool(self.config.lora.enabled and self.config.lora.use_qlora and HAS_PEFT and HAS_BNB)

    def setup_tokenizer(self) -> Any:
        if self.tokenizer is not None:
            return self.tokenizer
        cfg = self.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def build_model(self) -> Any:
        if self.model is not None:
            return self.model
        cfg = self.config
        torch_dtype = None
        if cfg.bf16:
            torch_dtype = getattr(torch, "bfloat16", None)
        elif cfg.fp16:
            torch_dtype = getattr(torch, "float16", None)
        quant_cfg = cfg.quantization
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if quant_cfg.enabled and HAS_BNB:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                compute_dtype = getattr(torch, quant_cfg.bnb_4bit_compute_dtype, torch.float16)
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quant_cfg.load_in_4bit,
                    load_in_8bit=quant_cfg.load_in_8bit,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
                )
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning("Failed to apply BitsAndBytes config: %s", exc)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        if cfg.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:  # pragma: no cover
                pass
        if cfg.lora.enabled and HAS_PEFT:
            if quant_cfg.enabled and HAS_BNB:
                try:
                    self.model = prepare_model_for_kbit_training(
                        self.model, use_gradient_checkpointing=cfg.gradient_checkpointing
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("prepare_model_for_kbit_training failed: %s", exc)
            target_modules = list(cfg.lora.target_modules)
            lora_cfg = _PeftLoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.lora_alpha,
                lora_dropout=cfg.lora.lora_dropout,
                target_modules=target_modules,
                bias=cfg.lora.bias,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            try:
                self.model.print_trainable_parameters()
            except Exception:  # pragma: no cover
                pass
        return self.model

    def prepare_dataset(
        self,
        samples: Sequence[PreparedSample],
        eval_samples: Optional[Sequence[PreparedSample]] = None,
    ) -> Tuple[Any, Optional[Any]]:
        if not samples:
            raise ValueError("No training samples provided")
        tokenizer = self.setup_tokenizer()
        if HAS_DATASETS:
            train_dicts = [s.to_dict() for s in samples]
            self.train_dataset = _HFDataset.from_list(train_dicts)
            if eval_samples:
                self.eval_dataset = _HFDataset.from_list([s.to_dict() for s in eval_samples])
            else:
                self.eval_dataset = None
        else:  # pragma: no cover - datasets is widely available
            self.train_dataset = SFTDataset(
                samples, tokenizer, max_seq_length=self.config.max_seq_length
            )
            self.eval_dataset = (
                SFTDataset(eval_samples, tokenizer, max_seq_length=self.config.max_seq_length)
                if eval_samples
                else None
            )
        return self.train_dataset, self.eval_dataset

    def _formatting_func(self) -> Any:
        def _format(example: Dict[str, Any]) -> str:
            return f"{example.get('prompt', '')}\n{example.get('reference', '')}"

        return _format

    def _build_training_args(self) -> Any:
        cfg = self.config
        common: Dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
            "logging_steps": cfg.logging_steps,
            "save_steps": cfg.save_steps,
            "eval_steps": cfg.eval_steps,
            "save_total_limit": cfg.save_total_limit,
            "bf16": cfg.bf16,
            "fp16": cfg.fp16,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "report_to": list(cfg.report_to),
            "seed": cfg.seed,
            "remove_unused_columns": False,
        }
        if cfg.eval_dataset is not None or self.eval_dataset is not None:
            common["evaluation_strategy"] = "steps"
            common["do_eval"] = True
        if cfg.deepspeed is not None:
            common["deepspeed"] = str(cfg.deepspeed)
        common["fsdp"] = cfg.fsdp
        if HAS_TRL:
            try:
                return _TRLSFTConfig(
                    max_seq_length=cfg.max_seq_length,
                    packing=False,
                    dataset_num_proc=1,
                    **common,
                )
            except TypeError:
                return TrainingArguments(**common)
        return TrainingArguments(**common)

    def _build_trainer(self) -> Any:
        cfg = self.config
        tokenizer = self.setup_tokenizer()
        model = self.build_model()
        args = self._build_training_args()
        if HAS_TRL:
            try:
                self.trainer = _TRLSFTTrainer(
                    model=model,
                    args=args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    tokenizer=tokenizer,
                    formatting_func=self._formatting_func(),
                )
                return self.trainer
            except Exception as exc:  # pragma: no cover - depends on TRL version
                logger.warning("TRL SFTTrainer failed (%s); falling back to transformers.Trainer", exc)
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers Trainer not available")
        pad_id = tokenizer.pad_token_id
        collator = lambda features: _pad_collator(list(features), pad_id)  # noqa: E731
        self.trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
        )
        return self.trainer

    def train(
        self,
        samples: Sequence[PreparedSample],
        eval_samples: Optional[Sequence[PreparedSample]] = None,
    ) -> SFTResult:
        """Run a supervised fine-tuning job.

        Returns
        -------
        :class:`SFTResult` with metrics, checkpoint paths, and timing.
        """
        start = time.time()
        self.prepare_dataset(samples, eval_samples)
        trainer = self._build_trainer()
        train_output = trainer.train()
        elapsed = time.time() - start
        metrics: Dict[str, float] = {}
        train_loss: Optional[float] = None
        eval_loss: Optional[float] = None
        try:
            metrics = dict(train_output.metrics or {})
            train_loss = metrics.get("train_loss")
        except Exception:  # pragma: no cover
            pass
        if self.eval_dataset is not None and hasattr(trainer, "evaluate"):
            try:
                eval_metrics = trainer.evaluate()
                metrics.update({f"eval_{k}": float(v) for k, v in eval_metrics.items()})
                eval_loss = eval_metrics.get("eval_loss")
            except Exception as exc:  # pragma: no cover
                logger.warning("Evaluation failed: %s", exc)
        try:
            trainer.save_model(str(self.output_dir))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(self.output_dir))
        except Exception as exc:  # pragma: no cover
            logger.warning("Saving model failed: %s", exc)
        self._collect_checkpoints()
        result = SFTResult(
            output_dir=self.output_dir,
            metrics=metrics,
            train_loss=train_loss,
            eval_loss=eval_loss,
            num_train_samples=len(samples),
            num_eval_samples=len(eval_samples) if eval_samples else 0,
            num_steps=int(metrics.get("train_steps", 0) or 0),
            epochs=float(metrics.get("epoch", 0.0) or 0.0),
            elapsed_seconds=elapsed,
            checkpoint_paths=list(self._checkpoint_paths),
            config_snapshot=self.config.to_dict(),
        )
        self._write_result(result)
        return result

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        if self.model is None:
            raise RuntimeError("Model has not been built yet")
        target = Path(path) if path else self.output_dir
        target.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save_pretrained(str(target))
        except Exception:  # pragma: no cover
            pass
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(target))
        return target

    def _collect_checkpoints(self) -> None:
        if not self.output_dir.exists():
            return
        for child in sorted(self.output_dir.iterdir()):
            if child.is_dir() and child.name.startswith("checkpoint-"):
                self._checkpoint_paths.append(child)
        if self.output_dir not in self._checkpoint_paths:
            self._checkpoint_paths.insert(0, self.output_dir)

    def _write_result(self, result: SFTResult) -> None:
        try:
            with open(self.output_dir / "sft_result.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            with open(self.output_dir / "training_config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=2, default=str)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write training result: %s", exc)

    def estimate_steps(
        self, num_samples: int, per_device_batch_size: Optional[int] = None
    ) -> int:
        bsz = per_device_batch_size or self.config.per_device_train_batch_size
        accum = self.config.gradient_accumulation_steps
        if bsz <= 0:
            return 0
        steps_per_epoch = max(1, math.ceil(num_samples / max(1, bsz * accum)))
        return int(steps_per_epoch * max(1, self.config.num_train_epochs))


__all__ = [
    "SFTResult",
    "SFTTrainer",
    "SFTDataset",
    "HAS_TRL",
    "HAS_PEFT",
    "HAS_BNB",
    "HAS_DATASETS",
    "HAS_TRANSFORMERS",
]

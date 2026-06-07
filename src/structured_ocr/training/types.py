"""Configuration dataclasses for the Structured OCR training pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TrainingMode(str, Enum):
    """Top-level training mode selection."""

    SFT = "sft"
    GRPO = "grpo"
    SFT_THEN_GRPO = "sft_then_grpo"


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation (LoRA / QLoRA) configuration."""

    enabled: bool = False
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    use_qlora: bool = False


@dataclass
class QuantizationConfig:
    """BitsAndBytes quantization configuration for QLoRA."""

    enabled: bool = False
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class RewardConfig:
    """Per-component reward weights used by the GRPO trainer.

    Nine components are recognized. They map directly to the reward
    functions in :mod:`structured_ocr.training.reward_functions`.
    """

    equation_accuracy: float = 0.15
    equation_syntax: float = 0.15
    table_structure: float = 0.10
    section_hierarchy: float = 0.10
    citation_label_integrity: float = 0.10
    cross_reference_validity: float = 0.10
    compilation_success: float = 0.20
    visual_similarity: float = 0.05
    semantic_coherence: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def validate(self) -> None:
        total = sum(self.as_dict().values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Reward weights must sum to ~1.0 (got {total:.4f}). "
                "Adjust values in training_config.yaml."
            )


@dataclass
class TrainingConfig:
    """End-to-end training configuration.

    This is the single source of truth for both SFT and GRPO stages. It
    can be serialized to / from JSON or YAML.
    """

    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    train_dataset: Path = field(default_factory=lambda: Path("./data/train.jsonl"))
    eval_dataset: Optional[Path] = None
    mode: TrainingMode = TrainingMode.SFT
    max_seq_length: int = 4096
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    seed: int = 42
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    reward_weights: Optional[RewardConfig] = None
    deepspeed: Optional[Path] = None
    fsdp: bool = False
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    run_verification: bool = True
    compiler_engine: str = "pdflatex"
    compiler_timeout: float = 30.0
    compiler_passes: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = TrainingMode(self.mode)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.train_dataset, str):
            self.train_dataset = Path(self.train_dataset)
        if isinstance(self.eval_dataset, str):
            self.eval_dataset = Path(self.eval_dataset)
        if isinstance(self.deepspeed, str):
            self.deepspeed = Path(self.deepspeed)
        if self.reward_weights is None:
            self.reward_weights = RewardConfig()
        if isinstance(self.reward_weights, dict):
            self.reward_weights = RewardConfig(**self.reward_weights)
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)
        if isinstance(self.quantization, dict):
            self.quantization = QuantizationConfig(**self.quantization)
        if self.reward_weights is not None:
            self.reward_weights.validate()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["output_dir"] = str(self.output_dir)
        data["train_dataset"] = str(self.train_dataset)
        if self.eval_dataset is not None:
            data["eval_dataset"] = str(self.eval_dataset)
        if self.deepspeed is not None:
            data["deepspeed"] = str(self.deepspeed)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        cleaned: Dict[str, Any] = dict(data)
        if "mode" in cleaned:
            cleaned["mode"] = TrainingMode(cleaned["mode"])
        return cls(**cleaned)

    @classmethod
    def from_json(cls, text: str) -> "TrainingConfig":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, text_or_path: Union[str, Path]) -> "TrainingConfig":
        import yaml

        if (
            Path(text_or_path).is_file()
            if isinstance(text_or_path, str)
            else text_or_path.is_file()
        ):
            with open(text_or_path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = yaml.safe_load(text_or_path) or {}
        return cls.from_dict(data)

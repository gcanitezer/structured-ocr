"""Training pipeline for Structured OCR (LaTeX OCR models).

Provides supervised fine-tuning (SFT) and GRPO/RLVR reinforcement
learning pipelines, along with configurable reward functions used to
score generated LaTeX against reference documents.
"""

from .types import (
    LoRAConfig,
    QuantizationConfig,
    RewardConfig,
    TrainingConfig,
    TrainingMode,
)
from .dataset_utils import DatasetUtils, PreparedSample
from .reward_functions import (
    LaTeXUnitTestFramework,
    RewardFunction,
    RewardResult,
    RewardWeights,
    UnitTestResult,
)
from .sft_trainer import (
    HAS_BNB as HAS_BNB,
    HAS_DATASETS as HAS_DATASETS,
    HAS_PEFT as HAS_PEFT,
    HAS_TRANSFORMERS as HAS_TRANSFORMERS,
    HAS_TRL as HAS_TRL,
    SFTDataset,
    SFTResult,
    SFTTrainer,
)
from .grpo_trainer import (
    HAS_TRL_GRPO as HAS_TRL_GRPO,
    GRPOResult,
    GRPOTrainer,
    RLVRConfig,
)
from .pipeline import TrainingPipeline, TrainingResult

__all__ = [
    "LoRAConfig",
    "QuantizationConfig",
    "RewardConfig",
    "TrainingConfig",
    "TrainingMode",
    "DatasetUtils",
    "PreparedSample",
    "LaTeXUnitTestFramework",
    "RewardFunction",
    "RewardResult",
    "RewardWeights",
    "UnitTestResult",
    "SFTResult",
    "SFTTrainer",
    "GRPOTrainer",
    "GRPOResult",
    "RLVRConfig",
    "TrainingPipeline",
    "TrainingResult",
]

"""Training pipeline for Structured OCR (LaTeX OCR models).

Provides supervised fine-tuning (SFT) and GRPO/RLVR reinforcement
learning pipelines, along with configurable reward functions used to
score generated LaTeX against reference documents.
"""

from .dataset_utils import DatasetUtils, PreparedSample
from .grpo_trainer import (
    HAS_TRL_GRPO as HAS_TRL_GRPO,
)
from .grpo_trainer import (
    GRPOResult,
    GRPOTrainer,
    RLVRConfig,
)
from .pipeline import TrainingPipeline, TrainingResult
from .reward_functions import (
    LaTeXUnitTestFramework,
    RewardFunction,
    RewardResult,
    UnitTestResult,
)
from .sft_trainer import (
    HAS_BNB as HAS_BNB,
)
from .sft_trainer import (
    HAS_DATASETS as HAS_DATASETS,
)
from .sft_trainer import (
    HAS_PEFT as HAS_PEFT,
)
from .sft_trainer import (
    HAS_TRANSFORMERS as HAS_TRANSFORMERS,
)
from .sft_trainer import (
    HAS_TRL as HAS_TRL,
)
from .sft_trainer import (
    SFTResult,
    SFTTrainer,
)
from .types import (
    LoRAConfig,
    QuantizationConfig,
    RewardConfig,
    TrainingConfig,
    TrainingMode,
)

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
    "UnitTestResult",
    "SFTResult",
    "SFTTrainer",
    "GRPOTrainer",
    "GRPOResult",
    "RLVRConfig",
    "TrainingPipeline",
    "TrainingResult",
]

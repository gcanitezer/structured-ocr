"""GRPO / RLVR trainer for the LaTeX OCR pipeline.

Implements Group Relative Policy Optimization (GRPO) reinforcement
learning with the nine reward functions exposed by
:mod:`structured_ocr.training.reward_functions`. The trainer prefers
TRL's :class:`GRPOTrainer` when available, and falls back to a
lightweight custom training loop using the standard
:class:`transformers.Trainer` infrastructure when TRL is not installed.

Either way, the trainer:

* samples ``num_generations`` completions per prompt,
* scores each completion with :class:`RewardFunction`,
* computes the GRPO advantage (group-normalized reward),
* updates the policy with a clipped surrogate loss plus a KL term
  against a frozen reference model.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .dataset_utils import PreparedSample
from .reward_functions import RewardFunction, RewardResult
from .types import TrainingConfig

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore

try:
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )

    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig as _PeftLoraConfig  # type: ignore
    from peft import (  # type: ignore
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    import bitsandbytes as bnb  # type: ignore  # noqa: F401

    HAS_BNB = True
except Exception:
    HAS_BNB = False

try:
    from trl import GRPOConfig as _TRLGRPOConfig  # type: ignore
    from trl import GRPOTrainer as _TRLGRPOTrainer  # type: ignore

    HAS_TRL_GRPO = True
except Exception:
    HAS_TRL_GRPO = False

try:
    from datasets import Dataset as _HFDataset  # type: ignore

    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


REWARD_NAMES: Tuple[str, ...] = (
    "equation_accuracy",
    "equation_syntax",
    "table_structure",
    "section_hierarchy",
    "citation_label_integrity",
    "cross_reference_validity",
    "compilation_success",
    "visual_similarity",
    "semantic_coherence",
)


@dataclass
class RLVRConfig:
    """Configuration that controls the GRPO / RLVR training loop."""

    num_generations: int = 4
    kl_coef: float = 0.05
    clip_ratio: float = 0.2
    max_new_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    epsilon: float = 1e-4
    reward_temperature: float = 1.0
    use_group_normalization: bool = True
    max_steps: Optional[int] = None
    log_rewards_every: int = 10
    reference_model_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLVRConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GRPOResult:
    """Outcome of a GRPO / RLVR training run."""

    output_dir: Path
    metrics: Dict[str, float] = field(default_factory=dict)
    reward_history: List[Dict[str, float]] = field(default_factory=list)
    final_reward: Optional[float] = None
    final_components: Dict[str, float] = field(default_factory=dict)
    num_prompts: int = 0
    num_steps: int = 0
    elapsed_seconds: float = 0.0
    checkpoint_paths: List[Path] = field(default_factory=list)
    config_snapshot: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d["checkpoint_paths"] = [str(p) for p in self.checkpoint_paths]
        return d


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _group_normalize(rewards: Sequence[float]) -> List[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var) + 1e-6
    return [(r - mean) / std for r in rewards]


class _GroupedRewardScorer:
    """Compute weighted reward per completion and group-normalize advantages."""

    def __init__(self, reward_function: RewardFunction) -> None:
        self.reward_function = reward_function

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        references: Sequence[str],
    ) -> Tuple[List[float], List[RewardResult]]:
        rewards: List[float] = []
        details: List[RewardResult] = []
        for pred, ref in zip(completions, references):
            try:
                result = self.reward_function.compute(pred, ref)
            except Exception as exc:  # pragma: no cover
                logger.warning("Reward computation failed: %s", exc)
                result = RewardResult(
                    total_reward=0.0,
                    components={k: 0.0 for k in REWARD_NAMES},
                    breakdown={k: "error" for k in REWARD_NAMES},
                    passed_tests=0,
                    total_tests=len(REWARD_NAMES),
                )
            rewards.append(float(result.total_reward))
            details.append(result)
        return list(rewards), details

    def group_advantages(self, rewards: Sequence[float]) -> List[float]:
        return _group_normalize(rewards)


class GRPOTrainer:
    """High-level GRPO/RLVR trainer.

    The trainer is initialized with a :class:`TrainingConfig` plus an
    optional :class:`RLVRConfig`. It supports LoRA / QLoRA via PEFT
    and bitsandbytes when available, and uses TRL's
    :class:`GRPOTrainer` when installed.

    A lightweight fallback loop (:class:`_GRPOFallbackLoop`) is
    provided for environments without TRL; it implements the core GRPO
    update equations (clipped surrogate + KL penalty) using the
    HuggingFace model directly.
    """

    def __init__(
        self,
        config: TrainingConfig,
        rlvr: Optional[RLVRConfig] = None,
        reward_function: Optional[RewardFunction] = None,
    ) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for GRPOTrainer. "
                "Install with: pip install transformers>=4.35.0"
            )
        self.config = config
        self.rlvr = rlvr or RLVRConfig()
        self.reward_function = reward_function or RewardFunction(weights=self.config.reward_weights)
        self.tokenizer: Any = None
        self.model: Any = None
        self.ref_model: Any = None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._scorer = _GroupedRewardScorer(self.reward_function)
        self._checkpoint_paths: List[Path] = []

    @property
    def uses_trl(self) -> bool:
        return bool(HAS_TRL_GRPO)

    @property
    def uses_lora(self) -> bool:
        return bool(self.config.lora.enabled and HAS_PEFT)

    @property
    def uses_qlora(self) -> bool:
        return bool(
            self.config.lora.enabled and self.config.lora.use_qlora and HAS_PEFT and HAS_BNB
        )

    def setup_tokenizer(self) -> Any:
        if self.tokenizer is not None:
            return self.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
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
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
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
            except Exception as exc:  # pragma: no cover
                logger.warning("BitsAndBytes config failed: %s", exc)
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
                except Exception:  # pragma: no cover
                    logger.warning("prepare_model_for_kbit_training failed")
            lora_cfg = _PeftLoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.lora_alpha,
                lora_dropout=cfg.lora.dropout
                if hasattr(cfg.lora, "dropout")
                else cfg.lora.lora_dropout,
                target_modules=list(cfg.lora.target_modules),
                bias=cfg.lora.bias,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
        return self.model

    def build_reference_model(self) -> Any:
        """Build a frozen reference model (only used by the fallback loop)."""
        if not HAS_TRANSFORMERS or self.model is None:
            return None
        if self.ref_model is not None:
            return self.ref_model
        ref_name = self.rlvr.reference_model_name or self.config.model_name
        try:
            ref = AutoModelForCausalLM.from_pretrained(
                ref_name,
                trust_remote_code=True,
                torch_dtype=self.model.dtype,
            )
            for p in ref.parameters():
                p.requires_grad_(False)
            ref.eval()
            self.ref_model = ref
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not build reference model: %s", exc)
            self.ref_model = None
        return self.ref_model

    def _trl_grpo_args(self) -> Any:
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
            "save_total_limit": cfg.save_total_limit,
            "bf16": cfg.bf16,
            "fp16": cfg.fp16,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "report_to": list(cfg.report_to),
            "seed": cfg.seed,
            "remove_unused_columns": False,
        }
        if cfg.deepspeed is not None:
            common["deepspeed"] = str(cfg.deepspeed)
        common["fsdp"] = cfg.fsdp
        grpo_specific: Dict[str, Any] = {
            "num_generations": self.rlvr.num_generations,
            "max_completion_length": self.rlvr.max_new_tokens,
            "beta": self.rlvr.kl_coef,
            "epsilon": self.rlvr.epsilon,
        }
        common.update(grpo_specific)
        if HAS_TRL_GRPO:
            try:
                return _TRLGRPOConfig(**common)
            except TypeError:
                return TrainingArguments(**common)
        return TrainingArguments(**common)

    def _build_trl_trainer(
        self,
        samples: Sequence[PreparedSample],
        reward_callable: Optional[Callable[..., Any]] = None,
    ) -> Any:
        if not HAS_TRL_GRPO:
            return None
        tokenizer = self.setup_tokenizer()
        model = self.build_model()
        args = self._trl_grpo_args()
        if HAS_DATASETS:
            ds = _HFDataset.from_list([s.to_dict() for s in samples])
        else:  # pragma: no cover
            ds = list(samples)
        reward_callable = reward_callable or self._trl_reward_function()
        try:
            self.trainer = _TRLGRPOTrainer(
                model=model,
                args=args,
                train_dataset=ds,
                processing_class=tokenizer,
                reward_funcs=reward_callable,
            )
            return self.trainer
        except Exception as exc:  # pragma: no cover
            logger.warning("TRL GRPOTrainer failed (%s); using fallback loop", exc)
            return None

    def _trl_reward_function(self) -> Callable[..., Any]:
        def score_completions(
            prompts: Sequence[str], completions: Sequence[str], **kwargs: Any
        ) -> List[float]:
            references = kwargs.get("reference", [""] * len(completions))
            if isinstance(references, str):
                references = [references] * len(completions)
            rewards, _ = self._scorer(list(prompts), list(completions), list(references))
            return rewards

        return score_completions

    def train(
        self,
        samples: Sequence[PreparedSample],
        reward_function: Optional[RewardFunction] = None,
    ) -> GRPOResult:
        """Run a GRPO / RLVR training job."""
        if reward_function is not None:
            self.reward_function = reward_function
            self._scorer = _GroupedRewardScorer(self.reward_function)
        if not samples:
            raise ValueError("No training samples provided")
        self.setup_tokenizer()
        self.build_model()
        start = time.time()
        if HAS_TRL_GRPO:
            trl_trainer = self._build_trl_trainer(samples)
            if trl_trainer is not None:
                train_output = trl_trainer.train()
                elapsed = time.time() - start
                metrics = dict(getattr(train_output, "metrics", {}) or {})
                try:
                    trl_trainer.save_model(str(self.output_dir))
                except Exception as exc:  # pragma: no cover
                    logger.warning("Saving model failed: %s", exc)
                self._collect_checkpoints()
                return GRPOResult(
                    output_dir=self.output_dir,
                    metrics=metrics,
                    num_prompts=len(samples),
                    num_steps=int(metrics.get("train_steps", 0) or 0),
                    elapsed_seconds=elapsed,
                    checkpoint_paths=list(self._checkpoint_paths),
                    config_snapshot=self._snapshot_config(),
                )
        loop = _GRPOFallbackLoop(
            model=self.model,
            ref_model=self.build_reference_model(),
            tokenizer=self.tokenizer,
            rlvr=self.rlvr,
            training_config=self.config,
            scorer=self._scorer,
            output_dir=self.output_dir,
        )
        result = loop.run(samples)
        self._collect_checkpoints()
        result.config_snapshot = self._snapshot_config()
        result.elapsed_seconds += time.time() - start
        return result

    def _snapshot_config(self) -> Dict[str, Any]:
        return {
            "training": self.config.to_dict(),
            "rlvr": self.rlvr.to_dict(),
        }

    def _collect_checkpoints(self) -> None:
        if not self.output_dir.exists():
            return
        for child in sorted(self.output_dir.iterdir()):
            if child.is_dir() and child.name.startswith("checkpoint-"):
                self._checkpoint_paths.append(child)
        if self.output_dir not in self._checkpoint_paths:
            self._checkpoint_paths.insert(0, self.output_dir)

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


class _GRPOFallbackLoop:
    """Minimal GRPO training loop using a HuggingFace causal LM.

    This loop implements the canonical GRPO objective:

        L = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
            + beta * KL(pi || pi_ref)

    where ``A_t`` is the per-prompt group-normalized reward and
    ``r_t = exp(log_pi(a_t | s_t) - log_pi_old(a_t | s_t))``.

    The loop is intentionally compact and operates on a single
    accelerator device - distributed training is handled by the
    surrounding :class:`GRPOTrainer` when launched via ``torchrun``.
    """

    def __init__(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        rlvr: RLVRConfig,
        training_config: TrainingConfig,
        scorer: _GroupedRewardScorer,
        output_dir: Path,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for the GRPO fallback loop")
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.rlvr = rlvr
        self.cfg = training_config
        self.scorer = scorer
        self.output_dir = output_dir
        self.optimizer: Any = None

    def _build_optimizer(self) -> Any:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return None
        return torch.optim.AdamW(
            params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )

    def _generate(self, prompts: Sequence[str]) -> List[str]:
        device = next(self.model.parameters()).device
        completions: List[str] = []
        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                try:
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=self.rlvr.max_new_tokens,
                        do_sample=True,
                        temperature=self.rlvr.temperature,
                        top_p=self.rlvr.top_p,
                        top_k=self.rlvr.top_k,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_return_sequences=1,
                    )
                except TypeError:
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=self.rlvr.max_new_tokens,
                        do_sample=True,
                        temperature=self.rlvr.temperature,
                        top_p=self.rlvr.top_p,
                        top_k=self.rlvr.top_k,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                if text.startswith(prompt):
                    text = text[len(prompt) :]
                completions.append(text)
        self.model.train()
        return completions

    def _log_probs(self, prompt: str, completion: str) -> "torch.Tensor":
        device = next(self.model.parameters()).device
        full = prompt + completion
        enc = self.tokenizer(full, return_tensors="pt").to(device)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
        input_ids = enc["input_ids"]
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0]
        labels = input_ids[0]
        completion_logits = logits[prompt_ids - 1 : -1]
        completion_labels = labels[prompt_ids:]
        log_probs = F.log_softmax(completion_logits.float(), dim=-1)
        token_lp = log_probs.gather(1, completion_labels.unsqueeze(-1)).squeeze(-1)
        return token_lp.sum()

    def _kl_penalty(self, prompt: str, completion: str) -> float:
        if self.ref_model is None or torch is None:
            return 0.0
        with torch.no_grad():
            new_lp = self._log_probs(prompt, completion)
            device = next(self.ref_model.parameters()).device
            full = prompt + completion
            enc = self.ref_model_input(full).to(device)
            prompt_ids = self.ref_model_prompt_len(prompt)
            out = self.ref_model(input_ids=enc["input_ids"])
            logits = out.logits[0]
            labels = enc["input_ids"][0]
            ref_logits = logits[prompt_ids - 1 : -1]
            ref_labels = labels[prompt_ids:]
            ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
            ref_token_lp = ref_log_probs.gather(1, ref_labels.unsqueeze(-1)).squeeze(-1)
            ref_lp = ref_token_lp.sum()
        kl = (new_lp - ref_lp).item()
        return max(0.0, kl)

    def _ref_model_input(self, full: str) -> Any:
        return self.tokenizer(full, return_tensors="pt")

    def _ref_model_prompt_len(self, prompt: str) -> int:
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]

    def _grpo_loss(
        self,
        advantages: Sequence[float],
        old_log_probs: Sequence["torch.Tensor"],
        new_log_probs: Sequence["torch.Tensor"],
    ) -> "torch.Tensor":
        eps = self.rlvr.epsilon
        losses: List["torch.Tensor"] = []
        for adv, old_lp, new_lp in zip(advantages, old_log_probs, new_log_probs):
            ratio = torch.exp(new_lp - old_lp.detach())
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
            losses.append(-torch.min(unclipped, clipped))
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(losses).mean()

    def _batched(
        self, items: Sequence[PreparedSample], batch_size: int
    ) -> Iterable[List[PreparedSample]]:
        for i in range(0, len(items), batch_size):
            yield list(items[i : i + batch_size])

    def run(self, samples: Sequence[PreparedSample]) -> GRPOResult:
        self.model.train()
        self.optimizer = self._build_optimizer()
        device = next(self.model.parameters()).device
        reward_history: List[Dict[str, float]] = []
        num_steps = 0
        max_steps = self.rlvr.max_steps or float("inf")
        log_every = max(1, int(self.rlvr.log_rewards_every))
        final_reward: Optional[float] = None
        final_components: Dict[str, float] = {}
        for epoch in range(max(1, self.cfg.num_train_epochs)):
            for batch in self._batched(samples, self.cfg.per_device_train_batch_size):
                if num_steps >= max_steps:
                    break
                prompts = [s.prompt for s in batch]
                references = [s.reference for s in batch]
                completions_per_prompt: List[List[str]] = []
                for _ in range(self.rlvr.num_generations):
                    completions_per_prompt.append(self._generate(prompts))
                flat_prompts: List[str] = []
                flat_completions: List[str] = []
                flat_refs: List[str] = []
                for p, gens, ref in zip(prompts, completions_per_prompt, references):
                    for g in gens:
                        flat_prompts.append(p)
                        flat_completions.append(g)
                        flat_refs.append(ref)
                rewards, details = self._scorer(flat_prompts, flat_completions, flat_refs)
                flat_prompts_repeat: List[str] = []
                for p, gens in zip(prompts, completions_per_prompt):
                    flat_prompts_repeat.extend([p] * len(gens))
                advantages = self._compute_advantages(rewards, self.rlvr.num_generations)
                old_log_probs: List["torch.Tensor"] = []
                with torch.no_grad():
                    for p, c in zip(flat_prompts_repeat, flat_completions):
                        old_log_probs.append(self._log_probs(p, c))
                new_log_probs: List["torch.Tensor"] = []
                kl_total = 0.0
                for p, c in zip(flat_prompts_repeat, flat_completions):
                    new_log_probs.append(self._log_probs(p, c))
                    kl_total += self._kl_penalty(p, c)
                loss = self._grpo_loss(advantages, old_log_probs, new_log_probs)
                if self.rlvr.kl_coef > 0 and kl_total:
                    loss = loss + self.rlvr.kl_coef * torch.tensor(kl_total, device=device)
                if self.optimizer is not None and loss.requires_grad:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                num_steps += 1
                mean_reward = _safe_mean(rewards)
                final_reward = mean_reward
                if details:
                    components: Dict[str, float] = {}
                    for name in REWARD_NAMES:
                        components[name] = _safe_mean(
                            [d.components.get(name, 0.0) for d in details]
                        )
                    final_components = components
                if num_steps % log_every == 0:
                    reward_history.append(
                        {
                            "step": num_steps,
                            "epoch": epoch,
                            "mean_reward": mean_reward,
                            "loss": float(loss.detach().item())
                            if hasattr(loss, "detach")
                            else float(loss),
                            "kl": kl_total,
                            "components": final_components,
                        }
                    )
                    logger.info(
                        "GRPO step %d reward=%.3f loss=%.3f kl=%.3f",
                        num_steps,
                        mean_reward,
                        float(loss.detach().item()) if hasattr(loss, "detach") else float(loss),
                        kl_total,
                    )
            if num_steps >= max_steps:
                break
        self._write_artifacts(
            reward_history, final_reward, final_components, num_steps, len(samples)
        )
        return GRPOResult(
            output_dir=self.output_dir,
            metrics={
                "mean_reward": float(final_reward or 0.0),
                "num_steps": float(num_steps),
            },
            reward_history=reward_history,
            final_reward=final_reward,
            final_components=final_components,
            num_prompts=len(samples),
            num_steps=num_steps,
            elapsed_seconds=0.0,
        )

    def _compute_advantages(self, rewards: Sequence[float], group_size: int) -> List[float]:
        if not self.rlvr.use_group_normalization or group_size <= 1:
            return list(rewards)
        groups: List[List[float]] = []
        for i in range(0, len(rewards), group_size):
            groups.append(list(rewards[i : i + group_size]))
        advantages: List[float] = []
        for g in groups:
            mu = sum(g) / len(g)
            var = sum((r - mu) ** 2 for r in g) / len(g)
            std = math.sqrt(var) + 1e-6
            advantages.extend([(r - mu) / std for r in g])
        return advantages

    def _write_artifacts(
        self,
        reward_history: Sequence[Dict[str, float]],
        final_reward: Optional[float],
        final_components: Dict[str, float],
        num_steps: int,
        num_prompts: int,
    ) -> None:
        try:
            with open(self.output_dir / "grpo_reward_history.json", "w") as f:
                json.dump(list(reward_history), f, indent=2)
            with open(self.output_dir / "grpo_result.json", "w") as f:
                json.dump(
                    {
                        "num_prompts": num_prompts,
                        "num_steps": num_steps,
                        "final_reward": final_reward,
                        "final_components": final_components,
                        "reward_history_len": len(reward_history),
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write GRPO artifacts: %s", exc)


__all__ = [
    "GRPOTrainer",
    "RLVRConfig",
    "GRPOResult",
    "REWARD_NAMES",
    "HAS_TRL_GRPO",
    "HAS_PEFT",
    "HAS_BNB",
    "HAS_DATASETS",
    "HAS_TRANSFORMERS",
]

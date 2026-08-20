"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""
import gc
import json
import math
import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm

from empire.training.base_strategy import TrainingStrategy
from empire.training.metrics import VLAMetrics
from empire.utils.run_logger import initialize_run_logger

# Initialize RunLogger =>> Wraps `logging.Logger`
run_logger = initialize_run_logger(__name__)  


BACKBONE_LR_SCHEDULES = {"freeze_warmup_cosine", "freeze_constant", "warmup_cosine", "constant"}
DIT_LR_SCHEDULES = {"warmup_cosine", "constant"}


def _normalize_lr_schedule_name(schedule_type: Optional[str]) -> Optional[str]:
    if schedule_type is None:
        return None

    normalized = schedule_type.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")

    aliases = {
        "cosine": "warmup_cosine",
        "warmup_cos": "warmup_cosine",
        "warmup_cosine_decay": "warmup_cosine",
        "linear_warmup_cosine_decay": "warmup_cosine",
        "linear_warmup_cosine": "warmup_cosine",
        "frozen_warmup_cosine": "freeze_warmup_cosine",
        "freeze_warmup_cos": "freeze_warmup_cosine",
        "frozen_warmup_cos": "freeze_warmup_cosine",
        "frozen_constant": "freeze_constant",
        "freeze": "freeze_constant",
        "freeze_warmup": "freeze_constant",
        "backbone_freeze_warmup": "freeze_constant",
    }
    return aliases.get(normalized, normalized)


def _resolve_step_count(
    value,
    num_training_steps: int,
    name: str,
    default: int = 0,
    negative_means_full_training: bool = False,
) -> int:
    """Resolve a step config where floats in [0, 1] are ratios and ints are absolute steps."""
    if value is None:
        return default

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int step count or a float ratio, not bool!")

    if isinstance(value, int):
        if value < 0 and negative_means_full_training:
            return num_training_steps
        if value < 0:
            raise ValueError(f"{name} must be >= 0!")
        return value

    if isinstance(value, float):
        if value < 0 and negative_means_full_training:
            return num_training_steps
        if value < 0:
            raise ValueError(f"{name} must be >= 0!")
        if value <= 1.0:
            return int(num_training_steps * value)
        if value.is_integer():
            return int(value)
        raise ValueError(
            f"{name}={value} is ambiguous. Use a ratio in [0, 1) or an integer number of steps."
        )

    raise ValueError(f"{name} must be an int step count or a float ratio, got {type(value).__name__}!")


def _freeze_constant_lambda(num_freeze_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < num_freeze_steps:
            return 0.0
        return 1.0

    return lr_lambda


def _warmup_cosine_lambda(
    num_training_steps: int,
    num_warmup_steps: int = 0,
    num_freeze_steps: int = 0,
    min_lr_rate: float = 0.1,
):
    if not 0.0 <= min_lr_rate <= 1.0:
        raise ValueError(f"min_lr_rate must be in [0, 1], got {min_lr_rate}!")

    def lr_lambda(current_step: int) -> float:
        if current_step < num_freeze_steps:
            return 0.0

        shifted_step = current_step - num_freeze_steps
        if num_warmup_steps > 0 and shifted_step < num_warmup_steps:
            return float(shifted_step) / float(max(1, num_warmup_steps))

        decay_steps = max(1, num_training_steps - num_freeze_steps - num_warmup_steps)
        progress = float(shifted_step - num_warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_rate + (1.0 - min_lr_rate) * cosine_decay

    return lr_lambda


def _constant_lambda(_current_step: int) -> float:
    return 1.0


def split_modality_collator(
    vla,
    cognition_token_weight_decay: bool = False,
    move_word_embedding_to_action_model: bool = False,
    verbose: bool = True
):
    """
    Split model parameters into vlm backbone and other (action model) groups with separate decay settings.
    
    Returns:
        Tuple of (backbone_decay, backbone_no_decay, other_decay, other_no_decay) parameter lists
    """
    backbone_decay, backbone_no_decay, other_decay, other_no_decay = [], [], [], []
    
    def is_backbone_param(name: str) -> bool:
        """Check if the parameter is part of the vision or text backbone."""
        if move_word_embedding_to_action_model and "embed_tokens" in name:
            return False
        return "backbone" in name

    for name, param in vla.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check parameters that should not have weight decay
        no_weight_decay = param.ndim <= 1 or name.endswith(".bias")
        if "cognition_token" in name:
            no_weight_decay = not cognition_token_weight_decay
        
        # Categorize parameters
        if no_weight_decay:
            if is_backbone_param(name):
                backbone_no_decay.append(param)
                if verbose:
                    run_logger.info(f"Parameter `{name}` is part of the backbone and has no decay; added to `backbone_no_decay`")
            else:
                other_no_decay.append(param)
                if verbose:
                    run_logger.info(f"Parameter `{name}` is not part of the backbone and has no decay; added to `other_no_decay`")
        else:
            if is_backbone_param(name):
                backbone_decay.append(param)
                if verbose:
                    run_logger.info(f"Parameter `{name}` is part of the backbone and has decay; added to `backbone_decay`")
            else:
                other_decay.append(param)
                if verbose:
                    run_logger.info(f"Parameter `{name}` is not part of the backbone and has decay; added to `other_decay`")
    
    return backbone_decay, backbone_no_decay, other_decay, other_no_decay


class VLAFSDPStrategy(TrainingStrategy):
    """FSDP (Fully Sharded Data Parallel) training strategy for VLA models."""

    def __init__(
        self,
        vla,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        action_model_learning_rate: Optional[float] = None,
        action_model_weight_decay: Optional[float] = None,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        sharding_strategy: str = "shard-grad-op",
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
        cognition_token_weight_decay: bool = False,
        llm_freeze_step: int = 0,
        move_word_embedding_to_action_model: bool = False,
        optimizer_betas: tuple = (0.9, 0.999),
        backbone_lr_scheduler_type: Optional[str] = None,
        dit_lr_scheduler_type: Optional[str] = None,
        backbone_warmup_steps: Optional[float] = None,
        dit_warmup_steps: Optional[float] = None,
        backbone_freeze_steps: Optional[float] = None,
        lr_min_rate: float = 0.1,
        backbone_min_lr_rate: Optional[float] = None,
        dit_min_lr_rate: Optional[float] = None,
        inline_cot_infer_interval: int = 0,
        inline_cot_infer_max_new_tokens: int = 128,
        inline_cot_infer_do_sample: bool = False,
        inline_cot_infer_temperature: float = 1.0,
        inline_cot_infer_top_k: int = 50,
        inline_cot_infer_top_p: float = 0.9,
        inline_cot_infer_output: str = "inline_cot_infer.jsonl",
        inline_cot_infer_at_start: bool = True,
    ) -> None:
        super().__init__(
            vla=vla,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
        )
        # Action model specific parameters
        self.action_model_learning_rate = action_model_learning_rate if action_model_learning_rate is not None else learning_rate
        self.action_model_weight_decay = action_model_weight_decay if action_model_weight_decay is not None else weight_decay
        self.cognition_token_weight_decay = cognition_token_weight_decay
        self.llm_freeze_step = llm_freeze_step
        self.move_word_embedding_to_action_model = move_word_embedding_to_action_model
        self.optimizer_betas = optimizer_betas
        self._llm_unfrozen = False  # Track whether LLM has been unfrozen
        self.backbone_lr_scheduler_type = backbone_lr_scheduler_type
        self.dit_lr_scheduler_type = dit_lr_scheduler_type
        self.backbone_warmup_steps = backbone_warmup_steps
        self.dit_warmup_steps = dit_warmup_steps
        self.backbone_freeze_steps = backbone_freeze_steps
        self.lr_min_rate = lr_min_rate
        self.backbone_min_lr_rate = backbone_min_lr_rate
        self.dit_min_lr_rate = dit_min_lr_rate
        self.inline_cot_infer_interval = int(inline_cot_infer_interval or 0)
        self.inline_cot_infer_max_new_tokens = inline_cot_infer_max_new_tokens
        self.inline_cot_infer_do_sample = inline_cot_infer_do_sample
        self.inline_cot_infer_temperature = inline_cot_infer_temperature
        self.inline_cot_infer_top_k = inline_cot_infer_top_k
        self.inline_cot_infer_top_p = inline_cot_infer_top_p
        self.inline_cot_infer_output = inline_cot_infer_output
        self.inline_cot_infer_at_start = bool(inline_cot_infer_at_start)

        # FSDP-specific parameters
        if sharding_strategy == "shard-grad-op":
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif sharding_strategy == "full-shard":
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise ValueError(f"FSDP sharding strategy '{sharding_strategy}' is not supported!")

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        self.fsdp_save_optimizer_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vla, FSDP), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"
        checkpoint_name = f"epoch={epoch}-step={global_step}.ckpt"
        checkpoint_dir = run_dir / "checkpoints"/ checkpoint_name
        if run_logger.is_rank_zero():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Align all ranks before FSDP starts its state-dict collectives. This is
        # especially important when rank 0 briefly stalls while creating the
        # checkpoint directory on shared storage.
        dist.barrier()
        
        def save_with_time(state_dict, path):
            run_logger.info(f"Saving state dict to {path} start at {datetime.now()}")
            torch.save(state_dict, path)
            run_logger.info(f"Saving state dict to {path} end at {datetime.now()}")
        
        # Gather full state dictionary from shards
        with FSDP.state_dict_type(self.vla, self.fsdp_state_dict_type, self.fsdp_save_policy, self.fsdp_save_optimizer_policy):
            run_logger.info("Gathering model state")
            model_state = self.vla.state_dict()
            run_logger.info("Preparing save checkpoint")
            run_logger.info("Gathering optimizer state")
            optim_state = FSDP.optim_state_dict(self.vla, self.optimizer)
            scheduler_state = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None

            meta_state = {
                "epoch": epoch,
                "global_step": global_step
            }
            if run_logger.is_rank_zero():
                with open(checkpoint_dir / "meta.json", "w") as f:
                    json.dump(meta_state, f)
            dist.barrier()
            if run_logger.is_rank_zero():
                threading.Thread(target=save_with_time, args=(model_state, checkpoint_dir / 'weights.pt')).start()
                threading.Thread(target=save_with_time, args=(optim_state, checkpoint_dir / 'optimizer.pt')).start()
                if scheduler_state is not None:
                    threading.Thread(target=save_with_time, args=(scheduler_state, checkpoint_dir / 'scheduler.pt')).start()
            
            dist.barrier()

    def load_optimizer_and_scheduler(self, checkpoint_folder: str) -> None:
        """Load optimizer and scheduler state from checkpoint."""
        assert isinstance(self.vla, FSDP), "FSDPStrategy.load_optimizer_and_scheduler assumes VLM is already wrapped in FSDP!"
        
        checkpoint_folder = Path(checkpoint_folder)
        optimizer_path = checkpoint_folder / "optimizer.pt"
        
        if not optimizer_path.exists():
            run_logger.warning(f"Optimizer checkpoint not found at {optimizer_path}!")
            return
        
        # Load checkpoint (FSDP handles device placement automatically)
        optim_state_dict = torch.load(optimizer_path, map_location="cpu")
        
        with FSDP.state_dict_type(
            self.vla,
            self.fsdp_state_dict_type,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        ):
            optim_state_dict = FSDP.optim_state_dict_to_load(self.vla, self.optimizer, optim_state_dict)
            # optim_state_dict = FSDP.optim_state_dict_to_load(self.vla, self.optimizer, optim_state_dict["optimizer"])
            self.optimizer.load_state_dict(optim_state_dict)
        
        run_logger.info(f"Loaded optimizer state dict from {optimizer_path}")

        # Load scheduler
        scheduler_path = checkpoint_folder / "scheduler.pt"
        if scheduler_path.exists() and self.lr_scheduler is not None:
            scheduler_state_dict = torch.load(scheduler_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(scheduler_state_dict)
            run_logger.info(f"Loaded scheduler state dict from {scheduler_path}")
        else:
            run_logger.warning(f"Scheduler checkpoint not found at {scheduler_path} or scheduler is None!")

    def _component_schedules_from_lr_scheduler_type(self) -> tuple:
        """Resolve legacy/global scheduler strings into backbone and DiT schedules."""
        lr_scheduler_type = self.lr_scheduler_type or "constant"
        normalized = lr_scheduler_type.strip().lower().replace("-", "_").replace(" ", "_")

        legacy_plus_mappings = {
            "linear_warmup+cosine_decay": ("warmup_cosine", "warmup_cosine"),
        }
        if normalized in legacy_plus_mappings:
            return legacy_plus_mappings[normalized]

        if "+" in normalized:
            component_schedules = {}
            parts = [part.strip("_") for part in normalized.split("+") if part.strip("_")]
            if len(parts) != 2:
                raise ValueError(
                    f"Composite lr_scheduler_type `{self.lr_scheduler_type}` must contain exactly two parts."
                )

            for idx, part in enumerate(parts):
                target = None
                schedule_name = part
                for prefix, prefix_target in (
                    ("backbone_", "backbone"),
                    ("llm_", "backbone"),
                    ("dit_", "dit"),
                    ("action_model_", "dit"),
                    ("action_", "dit"),
                ):
                    if part.startswith(prefix):
                        target = prefix_target
                        schedule_name = part[len(prefix):]
                        break

                if target is None:
                    target = "backbone" if idx == 0 else "dit"

                component_schedules[target] = _normalize_lr_schedule_name(schedule_name)

            if "backbone" not in component_schedules or "dit" not in component_schedules:
                raise ValueError(
                    f"Composite lr_scheduler_type `{self.lr_scheduler_type}` must specify backbone and dit schedules."
                )
            return component_schedules["backbone"], component_schedules["dit"]

        schedule = _normalize_lr_schedule_name(lr_scheduler_type)
        legacy_mappings = {
            "constant": ("constant", "constant"),
            "warmup_cosine": ("warmup_cosine", "warmup_cosine"),
            "freeze_constant": ("freeze_constant", "constant"),
            "freeze_warmup_cosine": ("freeze_warmup_cosine", "constant"),
            # Backward-compatible names used by existing configs.
            "backbone_freeze_warmup_action_constant": ("freeze_constant", "constant"),
            "backbone_freeze_warmup_constant": ("freeze_constant", "constant"),
            "backbone_freeze_warmup_dit_constant": ("freeze_constant", "constant"),
        }

        if schedule not in legacy_mappings:
            raise ValueError(
                f"Learning Rate Schedule `{self.lr_scheduler_type}` is not supported. "
                "Use backbone_lr_scheduler_type/dit_lr_scheduler_type, or a composite string like "
                "`backbone-freeze-warmup-cosine+dit-constant`."
            )

        return legacy_mappings[schedule]

    def _resolve_component_lr_scheduler_config(self, num_training_steps: int) -> dict:
        legacy_backbone_schedule, legacy_dit_schedule = self._component_schedules_from_lr_scheduler_type()
        backbone_schedule = _normalize_lr_schedule_name(self.backbone_lr_scheduler_type) or legacy_backbone_schedule
        dit_schedule = _normalize_lr_schedule_name(self.dit_lr_scheduler_type) or legacy_dit_schedule

        if backbone_schedule not in BACKBONE_LR_SCHEDULES:
            raise ValueError(
                f"Backbone LR schedule `{backbone_schedule}` is not supported. "
                f"Supported: {sorted(BACKBONE_LR_SCHEDULES)}"
            )
        if dit_schedule not in DIT_LR_SCHEDULES:
            raise ValueError(
                f"DiT LR schedule `{dit_schedule}` is not supported. Supported: {sorted(DIT_LR_SCHEDULES)}"
            )

        default_warmup = self.warmup_ratio if self.warmup_ratio is not None else 0
        backbone_warmup_value = self.backbone_warmup_steps
        if backbone_warmup_value is None:
            backbone_warmup_value = default_warmup
        dit_warmup_value = self.dit_warmup_steps
        if dit_warmup_value is None:
            dit_warmup_value = default_warmup

        backbone_freeze_value = self.backbone_freeze_steps
        if backbone_freeze_value is None:
            backbone_freeze_value = self.llm_freeze_step

        backbone_freeze_steps = _resolve_step_count(
            backbone_freeze_value,
            num_training_steps,
            "backbone_freeze_steps",
            default=0,
            negative_means_full_training=True,
        )
        backbone_warmup_steps = _resolve_step_count(
            backbone_warmup_value,
            num_training_steps,
            "backbone_warmup_steps",
            default=0,
        )
        dit_warmup_steps = _resolve_step_count(
            dit_warmup_value,
            num_training_steps,
            "dit_warmup_steps",
            default=0,
        )

        if not backbone_schedule.startswith("freeze_"):
            backbone_freeze_steps = 0
        if "warmup_cosine" not in backbone_schedule:
            backbone_warmup_steps = 0
        if dit_schedule != "warmup_cosine":
            dit_warmup_steps = 0

        backbone_min_lr_rate = self.backbone_min_lr_rate
        if backbone_min_lr_rate is None:
            backbone_min_lr_rate = self.lr_min_rate
        dit_min_lr_rate = self.dit_min_lr_rate
        if dit_min_lr_rate is None:
            dit_min_lr_rate = self.lr_min_rate

        return {
            "backbone_schedule": backbone_schedule,
            "dit_schedule": dit_schedule,
            "backbone_freeze_steps": backbone_freeze_steps,
            "backbone_warmup_steps": backbone_warmup_steps,
            "dit_warmup_steps": dit_warmup_steps,
            "backbone_min_lr_rate": backbone_min_lr_rate,
            "dit_min_lr_rate": dit_min_lr_rate,
        }

    def _build_component_lr_lambda(
        self,
        schedule_type: str,
        num_training_steps: int,
        warmup_steps: int = 0,
        freeze_steps: int = 0,
        min_lr_rate: float = 0.1,
    ):
        if schedule_type == "constant":
            return _constant_lambda
        if schedule_type == "freeze_constant":
            return _freeze_constant_lambda(freeze_steps)
        if schedule_type == "warmup_cosine":
            return _warmup_cosine_lambda(
                num_training_steps,
                num_warmup_steps=warmup_steps,
                min_lr_rate=min_lr_rate,
            )
        if schedule_type == "freeze_warmup_cosine":
            return _warmup_cosine_lambda(
                num_training_steps,
                num_warmup_steps=warmup_steps,
                num_freeze_steps=freeze_steps,
                min_lr_rate=min_lr_rate,
            )
        raise ValueError(f"Unsupported LR schedule `{schedule_type}`")


    def run_setup(
        self,
        run_dir: Path,
        n_train_examples: int,
        auto_wrap_policy_modules,
        checkpointing_policy_modules,
    ) -> None:
        """Setup FSDP training (wrap model, create optimizer, etc.)."""
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy_modules)

        # Configure FSDP mixed precision policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_buffer_dtype,
                buffer_dtype=reduce_buffer_dtype
            )
        else:
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32
            )

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        self.vla = FSDP(
            self.vla,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )
        
        # Setup gradient checkpointing
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            if checkpointing_policy_modules is not None:
                def check_fn(submodule: nn.Module) -> bool:
                    if isinstance(checkpointing_policy_modules, (list, set)):
                        return any(isinstance(submodule, module) for module in checkpointing_policy_modules)
                    return isinstance(submodule, checkpointing_policy_modules)

                # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
                apply_activation_checkpointing(self.vla, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        dist.barrier()

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        backbone_decay, backbone_no_decay, other_decay, other_no_decay = split_modality_collator(
            self.vla,
            cognition_token_weight_decay=self.cognition_token_weight_decay,
            move_word_embedding_to_action_model=self.move_word_embedding_to_action_model,
            verbose=False
        )
        groups = [
            {"params": backbone_decay, "weight_decay": self.weight_decay, "lr": self.learning_rate},
            {"params": backbone_no_decay, "weight_decay": 0.0, "lr": self.learning_rate},
            {"params": other_decay, "weight_decay": self.action_model_weight_decay, "lr": self.action_model_learning_rate},
            {"params": other_no_decay, "weight_decay": 0.0, "lr": self.action_model_learning_rate},
        ]

        # Create Optimizer & LR Scheduler
        self.optimizer = AdamW(groups, betas=self.optimizer_betas)

        lr_schedule_config = self._resolve_component_lr_scheduler_config(num_training_steps)
        backbone_lr_lambda = self._build_component_lr_lambda(
            lr_schedule_config["backbone_schedule"],
            num_training_steps,
            warmup_steps=lr_schedule_config["backbone_warmup_steps"],
            freeze_steps=lr_schedule_config["backbone_freeze_steps"],
            min_lr_rate=lr_schedule_config["backbone_min_lr_rate"],
        )
        dit_lr_lambda = self._build_component_lr_lambda(
            lr_schedule_config["dit_schedule"],
            num_training_steps,
            warmup_steps=lr_schedule_config["dit_warmup_steps"],
            min_lr_rate=lr_schedule_config["dit_min_lr_rate"],
        )
        self.lr_scheduler = MultiGroupLRScheduler(
            self.optimizer,
            backbone_lr_lambda=backbone_lr_lambda,
            dit_lr_lambda=dit_lr_lambda,
        )
        self.llm_freeze_step = lr_schedule_config["backbone_freeze_steps"]
        self._llm_unfrozen = self.llm_freeze_step == 0

        # Finalize Setup =>> Log!
        scheduler_info = f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
        scheduler_info += (
            f"                 |-> Backbone Schedule = {lr_schedule_config['backbone_schedule']}\n"
            f"                 |-> Backbone Freeze Steps = {lr_schedule_config['backbone_freeze_steps']}\n"
            f"                 |-> Backbone Warmup Steps = {lr_schedule_config['backbone_warmup_steps']}\n"
            f"                 |-> Backbone Min LR Rate = {lr_schedule_config['backbone_min_lr_rate']}\n"
            f"                 |-> DiT Schedule = {lr_schedule_config['dit_schedule']}\n"
            f"                 |-> DiT Warmup Steps = {lr_schedule_config['dit_warmup_steps']}\n"
            f"                 |-> DiT Min LR Rate = {lr_schedule_config['dit_min_lr_rate']}\n"
        )

        run_logger.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {run_logger.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> AdamW Betas = {self.optimizer_betas}\n"
            + scheduler_info +
            f"         |-> LLM Learning Rate = {self.learning_rate}\n"
            f"         |-> Action Model Learning Rate = {self.action_model_learning_rate}\n"
            f"         |-> LLM Weight Decay = {self.weight_decay}\n"
            f"         |-> Action Model Weight Decay = {self.action_model_weight_decay}\n"
            f"         |-> Cognition Token Weight Decay = {self.cognition_token_weight_decay}\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        """Clip gradients using FSDP's built-in gradient clipping."""
        self.vla.clip_grad_norm_(max_norm=self.max_grad_norm)

    def _enable_llm_training(self, current_step: int) -> None:
        """
        Enable LLM training by setting learning rate from 0 to normal value.

        This method is called when reaching the llm_freeze_step to transition
        from Stage 1 (LR=0) to Stage 2 (normal LR). Since LLM parameters are
        already in the optimizer with LR=0, we just need to update the LR.

        Args:
            current_step: The current training step
        """
        run_logger.info(f"Enabling LLM training at step {current_step} (transitioning from Stage 1 to Stage 2)", ctx_level=1)

        # The LR scheduler will automatically handle the LR transition from 0 to normal value.
        run_logger.info("LLM backbone parameters will now have LR > 0 (controlled by scheduler)", ctx_level=2)

    def _get_tokenizer(self):
        module = self.vla.module if isinstance(self.vla, FSDP) else self.vla
        return module.tokenizer

    def _build_cot_prompt_batch(self, input_ids, attention_mask, labels):
        if labels is None:
            return input_ids, attention_mask

        tokenizer = self._get_tokenizer()
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id or 0

        prompt_ids = []
        for batch_idx in range(input_ids.shape[0]):
            if attention_mask is None:
                valid_len = input_ids.shape[1]
            else:
                valid_len = int(attention_mask[batch_idx].to(torch.long).sum().item())

            label_positions = torch.nonzero(labels[batch_idx] != -100, as_tuple=False).flatten()
            if label_positions.numel() > 0:
                prompt_len = int(label_positions[0].item())
            else:
                prompt_len = valid_len
            prompt_len = max(1, min(prompt_len, valid_len))
            prompt_ids.append(input_ids[batch_idx, :prompt_len])

        padded_prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=pad_token_id)
        padded_prompt_mask = padded_prompt_ids.ne(pad_token_id)
        return padded_prompt_ids, padded_prompt_mask

    def _decode_label_batch(self, labels):
        if labels is None:
            return []
        tokenizer = self._get_tokenizer()
        decoded = []
        for batch_idx in range(labels.shape[0]):
            label_ids = labels[batch_idx][labels[batch_idx] != -100]
            decoded.append(tokenizer.decode(label_ids.tolist(), skip_special_tokens=False))
        return decoded

    def _decode_prompt_batch(self, input_ids, attention_mask):
        tokenizer = self._get_tokenizer()
        decoded = []
        for batch_idx in range(input_ids.shape[0]):
            if attention_mask is None:
                valid_len = input_ids.shape[1]
            else:
                valid_len = int(attention_mask[batch_idx].to(torch.long).sum().item())
            decoded.append(tokenizer.decode(input_ids[batch_idx, :valid_len].tolist(), skip_special_tokens=False))
        return decoded

    def _run_inline_cot_inference(
        self,
        batch,
        labels,
        global_step: int,
        run_dir: Path,
        force: bool = False,
    ) -> None:
        if self.inline_cot_infer_interval <= 0:
            return
        if not force and (global_step <= 0 or global_step % self.inline_cot_infer_interval != 0):
            return

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_input_ids, prompt_attention_mask = self._build_cot_prompt_batch(input_ids, attention_mask, labels)

        was_training = self.vla.training
        self.vla.eval()
        try:
            with torch.no_grad():
                output = self.vla.forward(
                    batch["pixel_values"],
                    prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    current_state_mask=batch.get("current_state_mask", None),
                    current_state=batch.get("current_state", None),
                    fov=batch.get("fov", None),
                    mode="cot_generate",
                    max_new_tokens=self.inline_cot_infer_max_new_tokens,
                    do_sample=self.inline_cot_infer_do_sample,
                    temperature=self.inline_cot_infer_temperature,
                    top_k=self.inline_cot_infer_top_k,
                    top_p=self.inline_cot_infer_top_p,
                    skip_special_tokens=False,
                )
        finally:
            if was_training:
                self.vla.train()

        dist.barrier()
        if not run_logger.is_rank_zero():
            return

        predictions = output["cot_predictions"]
        if isinstance(predictions, str):
            predictions = [predictions]

        record = {
            "step": global_step,
            "time": datetime.now().isoformat(),
            "settings": {
                "max_new_tokens": self.inline_cot_infer_max_new_tokens,
                "do_sample": self.inline_cot_infer_do_sample,
                "temperature": self.inline_cot_infer_temperature,
                "top_k": self.inline_cot_infer_top_k,
                "top_p": self.inline_cot_infer_top_p,
            },
            "samples": [],
        }
        prompt_texts = self._decode_prompt_batch(prompt_input_ids, prompt_attention_mask)
        gt_texts = self._decode_label_batch(labels)
        for sample_idx, prediction in enumerate(predictions):
            sample = {
                "sample_idx": sample_idx,
                "prompt": prompt_texts[sample_idx] if sample_idx < len(prompt_texts) else "",
                "prediction": prediction,
            }
            if sample_idx < len(gt_texts):
                sample["gt"] = gt_texts[sample_idx]
            record["samples"].append(sample)

        output_path = run_dir / self.inline_cot_infer_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        run_logger.info(f"Inline CoT inference written to {output_path} at step {global_step}", ctx_level=1)

    def run_training(
        self,
        dataloader,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        epoch_save_interval: int = 1,
        start_epoch: int = 0,
        start_global_step: int = 0,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given dataloader; log losses and action metrics to metrics."""
        vla_dataset = dataloader.dataset

        # Check if we need to enable LLM training at the start (resuming from checkpoint after freeze step)
        if self.llm_freeze_step > 0 and start_global_step >= self.llm_freeze_step and not self._llm_unfrozen:
            self._enable_llm_training(metrics.global_step)
            self._llm_unfrozen = True

        status = metrics.get_status() 
        with tqdm(
            total=(self.epochs * (len(dataloader) // self.grad_accumulation_steps)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not run_logger.is_rank_zero(),
            initial=start_global_step,
        ) as progress:
            train_idx = 0
            epoch = start_epoch
            last_processed_epoch = start_epoch
            last_saved_step = None
            last_saved_epoch = None
            ran_initial_inline_cot_infer = False

            # If max_steps is provided, we may need to iterate over the dataloader
            # multiple times to reach the requested number of optimizer steps. In
            # that case we drive the loop with an explicit iterator that is
            # re-created when exhausted. Otherwise we keep the conventional
            # epoch-limited loop behavior.
            batch_iter = iter(dataloader)
            while True:
                self.vla.train()
                self.optimizer.zero_grad()
                try:
                    batch = next(batch_iter)
                    # batch_idx is not critical here; keep compatibility if needed
                except StopIteration:
                    # finished one pass over dataloader => epoch end
                    # if epoch % epoch_save_interval == 0:
                    #     self.save_checkpoint(
                    #         metrics.run_dir, metrics.global_step, epoch, only_trainable=not save_full_model, is_epoch_end=True
                    #     )
                    gc.collect()
                    torch.cuda.empty_cache()

                    epoch += 1
                    # If max_steps not set, stop when we've completed configured epochs
                    if self.max_steps is None and epoch >= self.epochs:
                        break
                    # otherwise, start another pass over the dataloader
                    batch_iter = iter(dataloader)
                    continue

                # Unpack batch (same as previous implementation)
                last_processed_epoch = epoch
                batch_idx = None
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                input_ids = batch["input_ids"]
                rgb = batch["pixel_values"]
                attention_mask = batch["attention_mask"]
                action_labels = batch["actions"]
                action_masks = batch["action_masks"]
                current_state_mask = batch["current_state_mask"]
                current_state = batch["current_state"]
                fov = batch["fov"]
                obj_mesh_path = batch["obj_mesh_path"]
                raw_labels = batch.get("labels", None)
                # When CoT is disabled, labels may be a zero placeholder from data_utils.
                # In that case, pass None to avoid using dummy labels in forward.
                if raw_labels is None:
                    labels = None
                elif isinstance(raw_labels, torch.Tensor):
                    labels = None if torch.all(raw_labels == 0) else raw_labels
                else:
                    labels = raw_labels

                # if labels is None:
                #     run_logger.info("No labels provided in batch; ensure this is intentional (e.g. CoT disabled) and not a data loading issue!")

                if (
                    self.inline_cot_infer_at_start
                    and not ran_initial_inline_cot_infer
                    and start_global_step == 0
                    and metrics.global_step == 0
                ):
                    self._run_inline_cot_inference(
                        batch=batch,
                        labels=labels,
                        global_step=0,
                        run_dir=metrics.run_dir,
                        force=True,
                    )
                    ran_initial_inline_cot_infer = True

                prediction = self.vla.forward(
                    rgb,
                    input_ids,
                    attention_mask=attention_mask,
                    action_labels=action_labels,
                    action_masks=action_masks,
                    current_state_mask=current_state_mask,
                    current_state=current_state,
                    data_source=['action'],
                    fov=fov,
                    obj_mesh_path=obj_mesh_path,
                    labels=labels,
                )
                loss = prediction["loss"]

                # Commit loss and backward
                # maybe 2 stage train, in stage 1 only llm_loss, stage 2 has the DiT loss
                metrics.commit(
                    loss=loss,
                    left_hand_6d=prediction.get("left_hand_6d", 0.0), 
                    left_hand_joints=prediction.get("left_hand_joints", 0.0),
                    right_hand_6d=prediction.get("right_hand_6d", 0.0),
                    right_hand_joints=prediction.get("right_hand_joints", 0.0),
                    refine_loss=prediction.get("refine_loss", 0.0),
                    llm_loss=prediction.get("llm_loss", 0.0),
                )

                normalized_loss = loss / self.grad_accumulation_steps
                normalized_loss.backward()

                # === Gradient Step ===
                # Step =>> Only if Done w/ Gradient Accumulation
                if (train_idx + 1) % self.grad_accumulation_steps == 0:
                    # Check if we need to enable LLM training (unfreeze by setting LR > 0)
                    if (
                        self.llm_freeze_step > 0
                        and not self._llm_unfrozen
                        and metrics.global_step + 1 >= self.llm_freeze_step
                    ):
                        self._enable_llm_training(metrics.global_step + 1)
                        self._llm_unfrozen = True

                    # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                    self.clip_grad_norm()

                    # Optimizer & LR Scheduler Step
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    self.optimizer.zero_grad()
                    # Compute epoch value using number of completed gradient steps
                    # epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                    # Prepare learning rate metrics
                    lr_dict = {}
                    # Get the appropriate learning rate for logging
                    if isinstance(self.lr_scheduler, MultiGroupLRScheduler):
                        # For multi-group scheduler, log multiple learning rates
                        lr = self.lr_scheduler.get_last_lr()
                        lr_dict['backbone_decay_lr'] = lr[0]       # backbone decay learning rate
                        lr_dict['backbone_no_decay_lr'] = lr[1]    # backbone no decay learning rate
                        lr_dict['action_decay_lr'] = lr[2]         # action decay learning rate
                        lr_dict['action_no_decay_lr'] = lr[3]      # action no decay learning rate
                        current_lr = lr_dict['backbone_decay_lr']  # backbone learning rate
                    else:
                        current_lr = self.lr_scheduler.get_last_lr()[0]

                    metrics.commit(update_step_time=True, global_step=metrics.global_step + 1, epoch=epoch, lr=current_lr, **lr_dict)
                    status = metrics.push()

                    self._run_inline_cot_inference(
                        batch=batch,
                        labels=labels,
                        global_step=metrics.global_step,
                        run_dir=metrics.run_dir,
                    )

                    # Check for Save Interval or Max Steps & Save Checkpoint
                    if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                        (metrics.global_step % save_interval) == 0
                    ):
                        self.save_checkpoint(
                            metrics.run_dir, metrics.global_step, epoch, only_trainable=not save_full_model
                        )
                        last_saved_step = metrics.global_step
                        last_saved_epoch = epoch
                        dist.barrier()

                    if terminate:
                        break
                    if run_logger.is_rank_zero() and (metrics.global_step % 100 == 0):
                        run_logger.info(f"{status}")
                    # Update progress bar
                    progress.set_description(status)
                    progress.update()
                train_idx += 1

                # If termination was triggered inside optimizer-step branch, break outer while
                if self.max_steps is not None and metrics.global_step >= self.max_steps:
                    break

            # End of training loop
            # Save once at the end if the final step was not already checkpointed.
            if metrics.global_step > 0 and (last_saved_step != metrics.global_step or last_saved_epoch != last_processed_epoch):
                self.save_checkpoint(
                    metrics.run_dir, metrics.global_step, last_processed_epoch, only_trainable=not save_full_model
                )
# Custom LR Scheduler for different parameter groups
class MultiGroupLRScheduler:
    """
    Apply separate backbone and DiT learning-rate lambdas to the four optimizer groups.

    Group layout is:
        0: backbone decay
        1: backbone no decay
        2: DiT/action-model decay
        3: DiT/action-model no decay
    """
    def __init__(self, optimizer, backbone_lr_lambda, dit_lr_lambda):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                backbone_lr_lambda,
                backbone_lr_lambda,
                dit_lr_lambda,
                dit_lr_lambda,
            ],
        )

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`."""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the schedulers state."""
        if "backbone_scheduler" in state_dict and "action_model_scheduler" in state_dict:
            converted_state = self.scheduler.state_dict()
            backbone_state = state_dict["backbone_scheduler"]
            action_model_state = state_dict["action_model_scheduler"]
            converted_state["last_epoch"] = max(
                backbone_state.get("last_epoch", converted_state.get("last_epoch", -1)),
                action_model_state.get("last_epoch", converted_state.get("last_epoch", -1)),
            )
            converted_state["_step_count"] = max(
                backbone_state.get("_step_count", converted_state.get("_step_count", 0)),
                action_model_state.get("_step_count", converted_state.get("_step_count", 0)),
            )
            backbone_lrs = backbone_state.get("_last_lr", self.scheduler.get_last_lr()[:2])
            action_model_lrs = action_model_state.get("_last_lr", self.scheduler.get_last_lr()[2:])
            converted_state["_last_lr"] = list(backbone_lrs[:2]) + list(action_model_lrs[:2])
            self.scheduler.load_state_dict(converted_state)
            return

        self.scheduler.load_state_dict(state_dict)

import os
import copy
import json
import torch
import transformers
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

import empire.models.vla as vla
from empire.utils.data_utils import read_dataset_statistics, GaussianNormalizer


def build_vla(configs):
    model_fn = getattr(vla, configs["vla_name"])
    model = model_fn(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        act_model_configs=configs["action_model"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
        repeated_diffusion_steps=configs.get("repeated_diffusion_steps", 8),
        use_state=configs["use_state"],
        use_fov=configs.get("use_fov", True),
    )
    return model

def load_model(configs):
    model_load_path = configs.get("model_load_path", None)
    model = build_vla(
        configs=configs,
    )
    if model_load_path is not None:
        # Check if model_load_path is a file or directory
        if os.path.isfile(model_load_path):
            # If it's a file, load it directly
            checkpoint_path = model_load_path
        else:
            # If it's a directory, look for weights.pt inside
            checkpoint_path = os.path.join(model_load_path, "weights.pt")

        checkpoint_load_device = configs.get("checkpoint_load_device") or os.environ.get("CHECKPOINT_LOAD_DEVICE")
        checkpoint_assign = bool(configs.get("checkpoint_assign_state_dict", False))
        model = load_vla_checkpoint(
            model,
            checkpoint_path,
            map_location=checkpoint_load_device or "cpu",
            assign_state_dict=checkpoint_assign,
        )

    return model

def load_vla_checkpoint(model, checkpoint_path, map_location="cpu", assign_state_dict=False):
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Checkpoint map_location: {map_location}, assign_state_dict: {assign_state_dict}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Check if this is a Stage 1 (VLM-only) checkpoint being loaded into Stage 2 (with DiT)
    # In this case, only DiT/action-model parameters are allowed to be missing.
    training_stage = model.train_setup_configs.get("training_stage", None) if hasattr(model, 'train_setup_configs') else None
    strict_loading = True
    allowed_missing_prefixes = ()

    if training_stage in ("stage2", "dit_only", "stage_all"):
        # Stage 2: Check if loading from Stage 1 checkpoint (no DiT weights)
        checkpoint_keys = set(checkpoint.keys())
        model_has_act_model = hasattr(model, 'act_model') and model.act_model is not None

        if model_has_act_model:
            # Check if checkpoint has DiT-related keys
            has_dit_keys = any('act_model' in k for k in checkpoint_keys)
            if not has_dit_keys:
                print("Loading Stage 1 (VLM-only) checkpoint into Stage 2 (with DiT)")
                print("DiT weights will be randomly initialized")
                strict_loading = False
                allowed_missing_prefixes = ("act_model.",)

    with torch.no_grad():
        load_kwargs = {"strict": strict_loading}
        if assign_state_dict:
            load_kwargs["assign"] = True
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, **load_kwargs)

    if not strict_loading:
        disallowed_missing_keys = [
            key for key in missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        disallowed_unexpected_keys = list(unexpected_keys)

        if disallowed_missing_keys or disallowed_unexpected_keys:
            detail_lines = [
                "Checkpoint is not compatible with the current config.",
                "Only missing act_model.* keys are allowed when loading a Stage 1 checkpoint into Stage 2.",
            ]
            if disallowed_missing_keys:
                detail_lines.append(
                    f"Disallowed missing keys ({len(disallowed_missing_keys)}): "
                    f"{disallowed_missing_keys[:20]}"
                )
            if disallowed_unexpected_keys:
                detail_lines.append(
                    f"Disallowed unexpected keys ({len(disallowed_unexpected_keys)}): "
                    f"{disallowed_unexpected_keys[:20]}"
                )
            raise RuntimeError("\n".join(detail_lines))

        print(f"Missing act_model keys (expected for Stage 1 -> Stage 2): {len(missing_keys)}")
        print("Unexpected keys: 0")

    print("Checkpoint loaded")
    return model

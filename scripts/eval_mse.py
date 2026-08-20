import argparse
import os
import json
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import PaliGemmaProcessor

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from empire.models.vla_builder import load_model
from empire.datasets.dataset import EvalHumanDataset
from empire.utils.data_utils import PaddedCollatorForHandPrediction
from empire.utils.config_utils import load_config


def expand_action_masks(action_masks, action_dim, num_entities=None):
    """
    Expand per-entity action masks to per-dimension masks to match action_dim.

    Args:
        action_masks: [B, T, num_entities] or [B, T, action_dim]
        action_dim: target action dimension
        num_entities: number of entities (auto-detected if None)

    Returns:
        masks_expanded: [B, T, action_dim] bool tensor
    """
    if action_masks.ndim == 2:
        # [B, T] - apply to all action dimensions
        return action_masks.unsqueeze(-1).expand(-1, -1, action_dim) > 0.5

    B, T, mask_dim = action_masks.shape

    if mask_dim == action_dim:
        return action_masks > 0.5

    # Determine number of entities from mask
    if num_entities is None:
        num_entities = mask_dim

    # Expand per-entity mask to per-dimension mask
    if num_entities == 2:
        # [left(51), right(51)] = 102
        masks_expanded = torch.zeros(B, T, action_dim, dtype=torch.bool, device=action_masks.device)
        if action_dim >= 102:
            masks_expanded[:, :, :51] = (action_masks[:, :, 0:1] > 0.5)
            masks_expanded[:, :, 51:102] = (action_masks[:, :, 1:2] > 0.5)
        else:
            # Fallback: tile evenly
            half = action_dim // 2
            masks_expanded[:, :, :half] = (action_masks[:, :, 0:1] > 0.5)
            masks_expanded[:, :, half:] = (action_masks[:, :, 1:2] > 0.5)
    elif num_entities == 3:
        # [left(51), right(51), obj(6)] = 108
        masks_expanded = torch.zeros(B, T, action_dim, dtype=torch.bool, device=action_masks.device)
        if action_dim >= 108:
            masks_expanded[:, :, :51] = (action_masks[:, :, 0:1] > 0.5)
            masks_expanded[:, :, 51:102] = (action_masks[:, :, 1:2] > 0.5)
            masks_expanded[:, :, 102:108] = (action_masks[:, :, 2:3] > 0.5)
        else:
            raise ValueError(f"Cannot expand 3-entity mask to action_dim={action_dim}")
    else:
        raise ValueError(f"Unsupported number of entities: {num_entities}")

    return masks_expanded


def select_best_by_mse(samples, gt_actions, action_masks):
    """
    Select the sample with minimum MSE relative to GT for each batch element.

    Args:
        samples: [B, N, T, action_dim] or [B, T, action_dim] multiple diffusion samples
        gt_actions: [B, T, action_dim] ground truth actions
        action_masks: [B, T, num_entities] or [B, T, action_dim] action masks

    Returns:
        best_samples: [B, T, action_dim] selected predictions with minimum MSE
    """
    # If only 1 sample, return directly
    if samples.ndim == 3:
        return samples

    B, N, T, action_dim = samples.shape

    # Expand masks to per-dimension: [B, T, action_dim]
    num_entities = action_masks.shape[2] if action_masks.ndim == 3 else action_masks.shape[1]
    if action_masks.ndim == 3 and num_entities <= 3:
        masks_expanded = expand_action_masks(action_masks, action_dim, num_entities)
    else:
        masks_expanded = action_masks[:, :, :action_dim] > 0.5

    # Compute MSE: (pred - gt)^2, masked
    # gt_actions: [B, T, action_dim] -> [B, 1, T, action_dim]
    gt_expanded = gt_actions.unsqueeze(1)  # [B, 1, T, action_dim]

    squared_error = (samples - gt_expanded) ** 2  # [B, N, T, action_dim]
    masks_expanded_unsqueezed = masks_expanded.unsqueeze(1).float()  # [B, 1, T, action_dim]

    # Sum over T and action_dim
    masked_error = (squared_error * masks_expanded_unsqueezed).sum(dim=(2, 3))  # [B, N]
    valid_counts = masks_expanded_unsqueezed.sum(dim=(2, 3)).clamp(min=1)  # [B, N]
    mse_per_sample = masked_error / valid_counts  # [B, N]

    # Select best (minimum MSE) index for each batch element
    best_idx = mse_per_sample.argmin(dim=1)  # [B]

    # Gather best samples
    best_samples = samples[torch.arange(B, device=samples.device), best_idx]  # [B, T, action_dim]

    return best_samples


def recover_repeated_samples(samples, batch_size, repeated_diffusion_steps):
    """
    Recover flattened diffusion samples from [B*N, T, D] to [B, N, T, D].

    In `_forward_act_model`, samples are built via:
      x.unsqueeze(0).repeat(N, 1, ...).view(B*N, ...)
    so the flattened order is [n0b0, n0b1, ..., n1b0, n1b1, ...].
    """
    if repeated_diffusion_steps <= 1:
        return samples

    # Already in [B, N, T, D]
    if samples.ndim == 4:
        return samples

    # Expected flattened shape [B*N, T, D]
    if samples.ndim != 3:
        raise ValueError(f"Unexpected samples ndim={samples.ndim}, expected 3 or 4")

    total = samples.shape[0]
    expected = batch_size * repeated_diffusion_steps
    if total != expected:
        raise ValueError(
            f"Cannot recover repeated samples: got first dim={total}, "
            f"expected B*N={expected} (B={batch_size}, N={repeated_diffusion_steps})"
        )

    # [B*N, T, D] -> [N, B, T, D] -> [B, N, T, D]
    samples = samples.view(repeated_diffusion_steps, batch_size, *samples.shape[1:])
    samples = samples.permute(1, 0, 2, 3).contiguous()
    return samples


def batch_metadata(dataset, start, batch_size):
    core = dataset.episodic_dataset_core
    sample_positions = getattr(core, "eval_sample_positions", None)
    metadata = {
        "sample_positions": [],
        "data_indices": [],
        "episode_keys": [],
        "episode_ids": [],
        "frame_ids": [],
    }
    for local_idx in range(batch_size):
        sample_pos = start + local_idx
        original_sample_pos = int(sample_positions[sample_pos]) if sample_positions is not None else sample_pos
        data_id = int(core.sample_indices[sample_pos])
        pair = core.index_frame_pair[data_id]
        episode_key = int(pair[0])
        frame_id = int(pair[1])
        metadata["sample_positions"].append(original_sample_pos)
        metadata["data_indices"].append(data_id)
        metadata["episode_keys"].append(episode_key)
        metadata["episode_ids"].append(str(core.index_to_episode_id[episode_key]))
        metadata["frame_ids"].append(frame_id)
    return metadata


def apply_eval_shard(dataset, num_shards, shard_id):
    if num_shards <= 1:
        return
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

    core = dataset.episodic_dataset_core
    original_sample_indices = np.asarray(core.sample_indices, dtype=np.int64)
    shard_positions = np.arange(len(original_sample_indices), dtype=np.int64)[shard_id::num_shards]
    core.sample_indices = original_sample_indices[shard_positions]
    core.eval_sample_positions = shard_positions
    core.num_valid_frames = len(core.sample_indices)
    dataset._length = len(core)


def run_online_cot_dit(
    model,
    dataset,
    sample_offset,
    batch_size,
    current_state,
    current_state_mask,
    action_masks,
    fov,
    args,
    device,
):
    """Online CoT->DiT eval path (gated by COT_GENERATE_THEN_CONDITION=1).

    For each element of the (already collated) batch we re-derive the RAW coarse
    instruction and RAW image directly from the dataset core -- deliberately NOT using
    the batch's tokenized ``input_ids``, which for the unified use_cot model embed the
    GROUND-TRUTH CoT as a suffix (oracle leakage). The VLM then GENERATES the CoT and the
    DiT is conditioned on the hidden states of [image + prefix + generated CoT].

    Returns:
        samples: tensor on ``device`` shaped [B, N, T, D] when N>1 else [B, T, D]
                 (same convention as ``recover_repeated_samples``).
        cot_texts: list of generated CoT strings, ``len == batch_size``.
    """
    core = dataset.episodic_dataset_core
    normalization = getattr(dataset, "normalization", True)
    per_sample = []
    cot_texts = []
    for i in range(batch_size):
        idx = sample_offset + i
        # Mirror FrameDataset.__getitem__ up to (but excluding) post_transform, so we
        # recover the coarse instruction + raw image instead of the CoT-injected tokens.
        raw = core.__getitem__(idx)
        raw = core.transform_trajectory(raw, normalization)
        raw_image = raw["image_list"][-1]        # HxWx3 uint8 ndarray
        coarse_instruction = raw["instruction"]  # coarse instruction (no GT CoT)

        cot_text, action_np = model.predict_cot_then_action(
            image=raw_image,
            instruction=coarse_instruction,
            current_state=current_state[i:i + 1],
            current_state_mask=current_state_mask[i:i + 1],
            use_ddim=args.use_ddim,
            num_ddim_steps=args.num_ddim_steps,
            cfg_scale=args.cfg_scale,
            action_mask_torch=action_masks[i:i + 1],
            fov=fov[i:i + 1] if fov is not None else None,
            sample_times=args.repeated_diffusion_steps,
            use_cache=False,
        )
        per_sample.append(torch.from_numpy(np.ascontiguousarray(action_np)))  # [N, T, D]
        cot_texts.append(cot_text if isinstance(cot_text, str) else str(cot_text))

    samples = torch.stack(per_sample, dim=0).to(device)  # [B, N, T, D]
    if args.repeated_diffusion_steps <= 1:
        samples = samples[:, 0]  # [B, T, D]
    return samples, cot_texts


def run_online_cot_dit_batched(model, dataset, sample_offset, batch_size,
                               current_state, current_state_mask, action_masks, fov, args, device):
    """Batched prefix-LM single-pass online CoT->DiT (DEFAULT; SINGLEPASS_BATCHED=0 reverts to run_online_cot_dit).
    Same output contract as run_online_cot_dit: (samples[B,N,T,D or B,T,D], cot_texts[B])."""
    core = dataset.episodic_dataset_core
    normalization = getattr(dataset, "normalization", True)
    images, instructions = [], []
    for i in range(batch_size):
        raw = core.__getitem__(sample_offset + i)
        raw = core.transform_trajectory(raw, normalization)
        images.append(raw["image_list"][-1])
        instructions.append(raw["instruction"])
    cot_texts, action_np = model.predict_cot_then_action_batched(
        images=images, instructions=instructions, current_state=current_state,
        current_state_mask=current_state_mask, action_mask_torch=action_masks, fov=fov,
        sample_times=args.repeated_diffusion_steps, cfg_scale=args.cfg_scale,
        use_ddim=args.use_ddim, num_ddim_steps=args.num_ddim_steps)
    samples = torch.from_numpy(action_np).to(device)  # [B, N, T, D]
    if args.repeated_diffusion_steps <= 1:
        samples = samples[:, 0]
    return samples, cot_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint or config")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on")
    parser.add_argument("--config", type=str, default=None, help="Path to model configuration JSON file.")
    parser.add_argument("--repeated_diffusion_steps", type=int, default=5, help="Number of repeated diffusion samples to generate")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale for diffusion")
    parser.add_argument("--use_ddim", action="store_true", default=True, help="Use DDIM sampler")
    parser.add_argument("--num_ddim_steps", type=int, default=10, help="Number of DDIM steps")
    parser.add_argument("--save_repeated_samples", action="store_true", help="Save all repeated diffusion samples as sampled_actions [N,K,T,D].")
    parser.add_argument("--num_shards", type=int, default=1, help="Split eval dataset into this many strided shards.")
    parser.add_argument("--shard_id", type=int, default=0, help="Which strided eval shard this process should run.")
    parser.add_argument(
        "--checkpoint_load_device",
        type=str,
        default=os.environ.get("CHECKPOINT_LOAD_DEVICE", "cpu"),
        help="Device used for torch.load(map_location=...) when loading model_path. Use cuda for faster eval-only loading.",
    )
    parser.add_argument(
        "--checkpoint_assign_state_dict",
        action="store_true",
        default=os.environ.get("CHECKPOINT_ASSIGN_STATE_DICT", "0") == "1",
        help="Use load_state_dict(assign=True). Useful for eval-only direct-GPU checkpoint loading.",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = args.config
    elif os.path.isdir(args.model_path):
        config_path = os.path.join(args.model_path, "config.json")
    else:
        config_path = os.path.join(os.path.dirname(args.model_path), "config.json")

    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at {config_path}")

    config = load_config(config_path)

    if args.model_path:
        config['model_load_path'] = args.model_path
    if args.checkpoint_load_device:
        config["checkpoint_load_device"] = args.checkpoint_load_device
    if args.checkpoint_assign_state_dict:
        config["checkpoint_assign_state_dict"] = True

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(config)
    model.to(args.device)
    model.eval()

    processor = model.processor

    # Create dataset
    print(f"Creating dataset {args.dataset_name} from {args.data_path}...")
    train_dataset_config = config.get('train_dataset', {})

    # Extract configuration parameters with proper defaults
    action_future_window_size = train_dataset_config.get('action_future_window_size', 0)
    if action_future_window_size == 0 and 'fwd_pred_next_n' in config:
        action_future_window_size = config['fwd_pred_next_n'] - 1

    # Get use_obj, use_obj_rel, use_cot from config
    use_obj = train_dataset_config.get('use_obj', False)
    use_obj_rel = train_dataset_config.get('use_obj_rel', False)
    use_cot = train_dataset_config.get('use_cot', False)
    cot_as_instruction = train_dataset_config.get('cot_as_instruction', False)
    # plan-span conditioning: mirror training-time DiT conditioning at eval. Read from
    # the loaded model (single source of truth), falling back to the raw config.
    cot_condition_span = bool(
        getattr(model, "cot_condition_span", config.get("cot_condition_span", False))
    )

    print(f"Configuration:")
    print(f"  use_obj: {use_obj}")
    print(f"  use_obj_rel: {use_obj_rel}")
    print(f"  use_cot: {use_cot}")
    print(f"  cot_as_instruction: {cot_as_instruction}")
    print(f"  cot_condition_span: {cot_condition_span}")
    print(f"  action_future_window_size: {action_future_window_size}")
    print(f"  repeated_diffusion_steps: {args.repeated_diffusion_steps}")

    # Online CoT->DiT conditioning switch (gated by env var; default behavior unchanged).
    # When enabled (COT_GENERATE_THEN_CONDITION=1) for the unified online model
    # (use_cot=True, cot_as_instruction=False) that has a DiT, we DO NOT condition the DiT
    # on the GT-CoT-injected tokenized inputs that EvalHumanDataset/post_transform build.
    # Instead the VLM GENERATES the CoT from the RAW coarse instruction and the DiT is
    # conditioned on the hidden states of [image + prefix + generated CoT].
    cot_generate_then_condition = os.environ.get("COT_GENERATE_THEN_CONDITION", "0") == "1"
    online_cot_dit = (
        cot_generate_then_condition
        and bool(use_cot)
        and not bool(cot_as_instruction)
        and getattr(model, "act_model", None) is not None
    )
    print(f"  COT_GENERATE_THEN_CONDITION: {cot_generate_then_condition}")
    print(f"  online_cot_dit (generate CoT then condition DiT): {online_cot_dit}")

    dataset = EvalHumanDataset(
        dataset_folder=args.data_path,
        dataset_name=args.dataset_name,
        image_past_window_size=train_dataset_config.get('image_past_window_size', 0),
        image_future_window_size=train_dataset_config.get('image_future_window_size', 0),
        action_past_window_size=train_dataset_config.get('action_past_window_size', 0),
        action_future_window_size=action_future_window_size,
        augmentation=False,
        normalization=train_dataset_config.get('normalization', True),
        processor=processor,
        flip_augmentation=0.0,
        set_none_ratio=0.0,
        action_type=train_dataset_config.get('action_type', 'angle'),
        use_rel=train_dataset_config.get('use_rel', False),
        rel_mode=train_dataset_config.get('rel_mode', 'step'),
        load_images=True,
        data_type='human',
        clip_len=None,
        state_mask_prob=0.0,
        use_obj=use_obj,
        use_obj_rel=use_obj_rel,
        target_fps=train_dataset_config.get('target_fps', 12.0),
        source_fps=train_dataset_config.get('source_fps', None),
        source_video_fps=train_dataset_config.get('source_video_fps', 30.0),
        filter_samples_by_fps=train_dataset_config.get('filter_samples_by_fps', True),
        use_cot=use_cot or cot_as_instruction,
        cot_as_instruction=cot_as_instruction,
    )
    # get the global data statistics
    dataset.episodic_dataset_core.set_global_data_statistics(dataset.episodic_dataset_core.data_statistics)
    apply_eval_shard(dataset, args.num_shards, args.shard_id)

    print(f"Dataset size: {len(dataset)}")
    if args.num_shards > 1:
        print(f"Eval shard: {args.shard_id + 1}/{args.num_shards}")

    # Create collator
    collator = PaddedCollatorForHandPrediction(
        model_max_length=config.get('model_max_length', 2048),
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right"
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    # Evaluation loop
    results = {
        "pred_actions": [],
        "gt_actions": [],
        "action_masks": [],
        "current_states": [],
        "sample_positions": [],
        "data_indices": [],
        "episode_keys": [],
        "episode_ids": [],
        "frame_ids": [],
    }
    if args.save_repeated_samples:
        results["sampled_actions"] = []
    if online_cot_dit:
        results["cot_texts"] = []

    print("Starting evaluation (repeated diffusion + best-MSE selection)...")
    sample_offset = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch['pixel_values'].to(args.device)
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            current_state = batch['current_state'].to(args.device)
            current_state_mask = batch['current_state_mask'].to(args.device)
            action_masks = batch['action_masks'].to(args.device)
            fov = batch['fov'].to(args.device)
            gt_actions = batch['actions'].to(args.device)  # Move to device for MSE computation

            # Model inference with repeated diffusion
            if online_cot_dit:
                # Online CoT->DiT: generate the CoT from the RAW coarse instruction, then
                # condition the DiT on [image + prefix + generated CoT] (no GT CoT leakage).
                _rin = run_online_cot_dit if os.environ.get("SINGLEPASS_BATCHED") == "0" else run_online_cot_dit_batched  # batched single-pass is DEFAULT; set SINGLEPASS_BATCHED=0 to revert to the 2-pass path
                samples, batch_cot_texts = _rin(
                    model,
                    dataset,
                    sample_offset,
                    action_masks.shape[0],
                    current_state,
                    current_state_mask,
                    action_masks,
                    fov,
                    args,
                    args.device,
                )
                results["cot_texts"].extend(batch_cot_texts)
            else:
                # plan-span conditioning: pass the GT-CoT labels so the VLM sequence
                # layout matches training (cognition token inserted before the CoT, prefix-LM
                # mask) and restrict the DiT to the CoT-suffix tokens. Default path (flag off)
                # is unchanged: labels are NOT passed and condition_span_mask stays None.
                if cot_condition_span:
                    if batch.get('labels', None) is None:
                        raise ValueError(
                            "cot_condition_span=True requires the eval dataset to provide "
                            "'labels' (build it with use_cot=true)."
                        )
                    labels = batch['labels'].to(args.device)
                    if torch.all(labels == 0):
                        raise ValueError(
                            "cot_condition_span=True but eval batch labels are all-zero "
                            "placeholders; the eval dataset must be built with use_cot=true so "
                            "labels mark the GT CoT suffix."
                        )
                    output_hs, inputs_masks, _, aligned_labels = model.prepare_vlm_features(
                        pixel_values,
                        input_ids,
                        attention_mask,
                        current_state_mask,
                        current_state,
                        fov,
                        use_cache=False,
                        labels=labels,
                        return_aligned_labels=True,
                    )
                    condition_span_mask = (
                        (aligned_labels != -100) if aligned_labels is not None else None
                    )
                else:
                    output_hs, inputs_masks, _ = model.prepare_vlm_features(
                        pixel_values,
                        input_ids,
                        attention_mask,
                        current_state_mask,
                        current_state,
                        fov,
                        use_cache=False,
                    )
                    condition_span_mask = None

                samples, _ = model._forward_act_model(
                    vlm_features=output_hs,
                    attention_mask=inputs_masks,
                    action_masks=action_masks,
                    current_state=current_state,
                    current_state_mask=current_state_mask,
                    mode="eval",
                    repeated_diffusion_steps=args.repeated_diffusion_steps,
                    cfg_scale=args.cfg_scale,
                    use_ddim=args.use_ddim,
                    num_ddim_steps=args.num_ddim_steps,
                    condition_span_mask=condition_span_mask,
                )

                # Recover [B, N, T, D] when repeated diffusion flattens to [B*N, T, D]
                samples = recover_repeated_samples(
                    samples,
                    batch_size=action_masks.shape[0],
                    repeated_diffusion_steps=args.repeated_diffusion_steps,
                )
            # import pdb; pdb.set_trace()
            # Select best sample by MSE relative to GT
            if args.repeated_diffusion_steps > 1:
                pred_actions = select_best_by_mse(samples, gt_actions, action_masks)
            else:
                pred_actions = samples

            # Mask predictions
            pred_actions = pred_actions * action_masks
            if args.save_repeated_samples:
                sampled_actions = samples.unsqueeze(1) if samples.ndim == 3 else samples
                sampled_actions = sampled_actions * action_masks.unsqueeze(1)

            # Store results
            results["pred_actions"].append(pred_actions.cpu().numpy())
            results["gt_actions"].append(gt_actions.cpu().numpy())
            results["action_masks"].append(action_masks.cpu().numpy())
            results["current_states"].append(current_state.cpu().numpy())
            if args.save_repeated_samples:
                results["sampled_actions"].append(sampled_actions.cpu().numpy())

            meta = batch_metadata(dataset, sample_offset, action_masks.shape[0])
            for key, value in meta.items():
                results[key].extend(value)
            sample_offset += action_masks.shape[0]

    # Concatenate results
    for key in ("pred_actions", "gt_actions", "action_masks", "current_states", "sampled_actions"):
        if key in results:
            results[key] = np.concatenate(results[key], axis=0)
    results["data_indices"] = np.asarray(results["data_indices"], dtype=np.int64)
    results["sample_positions"] = np.asarray(results["sample_positions"], dtype=np.int64)
    results["episode_keys"] = np.asarray(results["episode_keys"], dtype=np.int64)
    results["episode_ids"] = np.asarray(results["episode_ids"])
    results["frame_ids"] = np.asarray(results["frame_ids"], dtype=np.int64)
    results["frame_sample_step"] = np.asarray(dataset.episodic_dataset_core.frame_sample_step, dtype=np.float32)
    results["target_fps"] = np.asarray(dataset.episodic_dataset_core.target_fps or -1, dtype=np.float32)
    results["source_fps"] = np.asarray(dataset.episodic_dataset_core.source_fps or -1, dtype=np.float32)
    if "cot_texts" in results:
        # Generated CoT strings from the online CoT->DiT path (object array, optional).
        results["cot_texts"] = np.asarray(results["cot_texts"], dtype=object)

    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(args.output_path, **results)
    print(f"Results saved to {args.output_path}")
    print(f"Total predictions: {results['pred_actions'].shape[0]}")


if __name__ == "__main__":
    main()

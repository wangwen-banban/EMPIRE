#!/usr/bin/env python3
"""Compute EgoDex best-of-K 3D hand-joint metrics from eval_mse.py outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from empire.utils.data_utils import load_normalizer

from egodex_eval_utils import (
    build_mano_layer,
    expand_entity_mask,
    reconstruct_hand_joints,
)


def side_errors(pred, gt, mask):
    out = {}
    total_errors = {
        "mpjpe": 0.0,
        "wrist_mpjpe": 0.0,
        "finger_relative_mpjpe": 0.0,
    }
    total_count = 0
    fde_values = []
    for side_idx, side in enumerate(("left", "right")):
        valid = mask[:, side_idx]
        if not np.any(valid):
            out[f"{side}_mpjpe"] = np.nan
            out[f"{side}_fde"] = np.nan
            out[f"{side}_wrist_mpjpe"] = np.nan
            out[f"{side}_finger_relative_mpjpe"] = np.nan
            out[f"{side}_frames"] = 0
            continue

        pred_joints = pred[f"{side}_joints"]
        gt_joints = gt[f"{side}_joints"]
        joint_dist = np.linalg.norm(pred_joints - gt_joints, axis=-1)
        frame_mpjpe = joint_dist.mean(axis=-1)
        wrist_dist = joint_dist[:, 0]
        pred_finger_relative = pred_joints[:, 1:] - pred_joints[:, :1]
        gt_finger_relative = gt_joints[:, 1:] - gt_joints[:, :1]
        finger_relative_dist = np.linalg.norm(
            pred_finger_relative - gt_finger_relative,
            axis=-1,
        ).mean(axis=-1)

        out[f"{side}_mpjpe"] = float(frame_mpjpe[valid].mean())
        out[f"{side}_frames"] = int(valid.sum())
        final_idx = np.flatnonzero(valid)[-1]
        out[f"{side}_fde"] = float(frame_mpjpe[final_idx])
        out[f"{side}_wrist_mpjpe"] = float(wrist_dist[valid].mean())
        out[f"{side}_finger_relative_mpjpe"] = float(finger_relative_dist[valid].mean())
        total_errors["mpjpe"] += float(frame_mpjpe[valid].sum())
        total_errors["wrist_mpjpe"] += float(wrist_dist[valid].sum())
        total_errors["finger_relative_mpjpe"] += float(finger_relative_dist[valid].sum())
        total_count += int(valid.sum())
        fde_values.append(float(frame_mpjpe[final_idx]))
    for key, total_error in total_errors.items():
        out[key] = float(total_error / total_count) if total_count > 0 else np.nan
    out["fde"] = float(np.mean(fde_values)) if fde_values else np.nan
    out["frames"] = int(total_count)
    return out


def nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")


def summarize(records):
    summary = {
        "num_samples": len(records),
        "overall_mpjpe_m": nanmean([r["mpjpe"] for r in records]),
        "overall_mpjpe_mm": nanmean([r["mpjpe"] for r in records]) * 1000.0,
        "overall_fde_m": nanmean([r["fde"] for r in records]),
        "overall_fde_mm": nanmean([r["fde"] for r in records]) * 1000.0,
        "overall_wrist_mpjpe_m": nanmean([r["wrist_mpjpe"] for r in records]),
        "overall_wrist_mpjpe_mm": nanmean([r["wrist_mpjpe"] for r in records]) * 1000.0,
        "overall_finger_relative_mpjpe_m": nanmean(
            [r["finger_relative_mpjpe"] for r in records]
        ),
        "overall_finger_relative_mpjpe_mm": nanmean(
            [r["finger_relative_mpjpe"] for r in records]
        )
        * 1000.0,
        "left_mpjpe_m": nanmean([r["left_mpjpe"] for r in records]),
        "left_fde_m": nanmean([r["left_fde"] for r in records]),
        "left_wrist_mpjpe_m": nanmean([r["left_wrist_mpjpe"] for r in records]),
        "left_finger_relative_mpjpe_m": nanmean(
            [r["left_finger_relative_mpjpe"] for r in records]
        ),
        "right_mpjpe_m": nanmean([r["right_mpjpe"] for r in records]),
        "right_fde_m": nanmean([r["right_fde"] for r in records]),
        "right_wrist_mpjpe_m": nanmean([r["right_wrist_mpjpe"] for r in records]),
        "right_finger_relative_mpjpe_m": nanmean(
            [r["right_finger_relative_mpjpe"] for r in records]
        ),
        "valid_hand_frames": int(sum(r["frames"] for r in records)),
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="EgoDex best-of-K MPJPE/FDE evaluator.")
    parser.add_argument("--eval_npz", default=None, help="NPZ produced by scripts/eval_mse.py (required unless --merge).")
    parser.add_argument("--config", default=None, help="Checkpoint config JSON (required unless --merge; currently unused by this script).")
    parser.add_argument("--statistics_path", default=os.environ.get("STATISTICS_PATH"))
    parser.add_argument("--mano_model_path", default=os.environ.get("MANO_MODEL_PATH"))
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_npz", default=None, help="Optional compact NPZ with best indices/actions and per-sample metrics.")
    parser.add_argument(
        "--reuse_best_indices_from",
        default=None,
        help=(
            "Optional existing metrics NPZ. Reuse its best_indices instead of "
            "re-evaluating all K candidates; useful when adding metrics without "
            "changing the global-MPJPE best-of-K selection."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    # Strided sharding of the per-window metric compute across GPUs. num_shards=1
    # (the default) is byte-identical to the original single-device behavior.
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Process only windows where (index modulo num_shards) == shard_id.")
    parser.add_argument("--shard_id", type=int, default=0, help="This shard's id in [0, num_shards).")
    # Merge mode: combine shard --output_json files into one metrics json identical to the
    # single-GPU run (concatenate records, sort by sample_index, re-aggregate via summarize()).
    parser.add_argument("--merge", action="store_true", help="Merge shard metrics jsons into one.")
    parser.add_argument("--merge_inputs", nargs="+", default=None,
                        help="Shard metrics json paths to merge (requires --merge).")
    parser.add_argument("--merge_npz_inputs", nargs="+", default=None,
                        help="Optional shard metrics npz paths (paired by position with --merge_inputs); "
                             "merged into --output_npz when given.")
    return parser.parse_args()


def run_merge(args) -> int:
    """Merge shard metrics jsons (and optional npzs) into a result identical to a single-GPU
    run: concatenate per-window records, sort by sample_index, and re-aggregate with the SAME
    summarize() used by the single-device path. MANO reconstruction is per-window deterministic,
    so sharded+merged == single-GPU (given identical GPU arch across shards)."""
    if not args.merge_inputs:
        raise SystemExit("--merge requires --merge_inputs (shard metrics json paths).")
    shard_jsons = []
    for path in args.merge_inputs:
        with open(path) as handle:
            shard_jsons.append(json.load(handle))

    merged_records = []
    for shard in shard_jsons:
        merged_records.extend(shard.get("records", []))
    # Sort by global window index so the merged list is in the exact 0..n-1 order a
    # single-GPU run produces -> identical (order-sensitive) nanmean aggregation.
    merged_records.sort(key=lambda record: record["sample_index"])

    sample_indices = [record["sample_index"] for record in merged_records]
    if len(set(sample_indices)) != len(sample_indices):
        raise SystemExit("--merge: overlapping sample_index across shards (windows double-counted).")

    base = shard_jsons[0]
    output = {
        "eval_npz": base.get("eval_npz"),
        "statistics_path": base.get("statistics_path"),
        "mano_model_path": base.get("mano_model_path"),
        "best_of_k": base.get("best_of_k"),
        "summary": summarize(merged_records),
        "records": merged_records,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(output, handle, indent=2)

    # Optional: rebuild the compact per-window NPZ by scattering each shard's owned windows
    # (identified by their record sample_index) into full-size arrays -> identical to single-GPU.
    if args.output_npz is not None and args.merge_npz_inputs:
        if len(args.merge_npz_inputs) != len(args.merge_inputs):
            raise SystemExit("--merge_npz_inputs must pair 1:1 (by position) with --merge_inputs.")
        shard_npzs = [np.load(path, allow_pickle=True) for path in args.merge_npz_inputs]
        keys = list(shard_npzs[0].files)
        merged_arrays = {key: np.array(shard_npzs[0][key]) for key in keys}
        for shard, npz in zip(shard_jsons, shard_npzs):
            owned = np.asarray(
                [record["sample_index"] for record in shard.get("records", [])],
                dtype=np.int64,
            )
            if owned.size:
                for key in keys:
                    merged_arrays[key][owned] = npz[key][owned]
        for npz in shard_npzs:
            npz.close()
        Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_npz, **merged_arrays)

    summary = output["summary"]
    print(f"Merged {len(shard_jsons)} shard(s) -> {len(merged_records)} windows")
    print(f"Best-of-{output['best_of_k']} EgoDex MPJPE: {summary['overall_mpjpe_mm']:.2f} mm")
    print(f"Best-of-{output['best_of_k']} EgoDex FDE:   {summary['overall_fde_mm']:.2f} mm")
    print(f"Wrote merged metrics to {args.output_json}")
    if args.output_npz is not None and args.merge_npz_inputs:
        print(f"Wrote merged compact arrays to {args.output_npz}")
    return 0


def main() -> int:
    args = parse_args()
    if args.merge:
        return run_merge(args)
    if args.eval_npz is None or args.config is None:
        raise SystemExit("--eval_npz and --config are required unless --merge is set.")
    if not args.statistics_path or not args.mano_model_path:
        raise SystemExit("--statistics_path and --mano_model_path (or matching environment variables) are required.")
    if args.num_shards < 1:
        raise SystemExit(f"--num_shards must be >= 1 (got {args.num_shards}).")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard_id must be in [0, {args.num_shards}) (got {args.shard_id}).")
    data = np.load(args.eval_npz, allow_pickle=True)
    sampled_actions = data["sampled_actions"] if "sampled_actions" in data else data["pred_actions"][:, None]
    gt_actions = data["gt_actions"]
    action_masks = data["action_masks"]
    current_states = data["current_states"]

    if args.limit is not None:
        sampled_actions = sampled_actions[: args.limit]
        gt_actions = gt_actions[: args.limit]
        action_masks = action_masks[: args.limit]
        current_states = current_states[: args.limit]

    normalizer = load_normalizer({"statistics_path": args.statistics_path})
    mano_layer = build_mano_layer(args.mano_model_path, args.device)

    n, k = sampled_actions.shape[:2]
    reused_metrics = None
    reused_best_indices = None
    if args.reuse_best_indices_from is not None:
        reused_metrics = np.load(args.reuse_best_indices_from, allow_pickle=True)
        if "best_indices" not in reused_metrics:
            raise KeyError(f"{args.reuse_best_indices_from} does not contain best_indices")
        reused_best_indices = np.asarray(reused_metrics["best_indices"], dtype=np.int64)
        if len(reused_best_indices) < n:
            raise ValueError(
                f"best_indices has {len(reused_best_indices)} samples, expected at least {n}"
            )
        reused_best_indices = reused_best_indices[:n]
        if np.any((reused_best_indices < 0) | (reused_best_indices >= k)):
            raise ValueError(f"best_indices must be in [0, {k})")

    records = []
    best_indices = np.zeros(n, dtype=np.int64)
    best_pred_actions = np.zeros_like(sampled_actions[:, 0])
    sample_mpjpe = np.full(n, np.nan, dtype=np.float32)
    sample_fde = np.full(n, np.nan, dtype=np.float32)
    sample_wrist_mpjpe = np.full(n, np.nan, dtype=np.float32)
    sample_finger_relative_mpjpe = np.full(n, np.nan, dtype=np.float32)
    left_mpjpe = np.full(n, np.nan, dtype=np.float32)
    right_mpjpe = np.full(n, np.nan, dtype=np.float32)
    left_wrist_mpjpe = np.full(n, np.nan, dtype=np.float32)
    right_wrist_mpjpe = np.full(n, np.nan, dtype=np.float32)
    left_finger_relative_mpjpe = np.full(n, np.nan, dtype=np.float32)
    right_finger_relative_mpjpe = np.full(n, np.nan, dtype=np.float32)
    if reused_metrics is not None and "candidate_mpjpe" in reused_metrics:
        candidate_mpjpe = np.asarray(reused_metrics["candidate_mpjpe"], dtype=np.float32)[:n].copy()
    else:
        candidate_mpjpe = np.full((n, k), np.nan, dtype=np.float32)
    if reused_metrics is not None:
        reused_metrics.close()

    episode_ids = data["episode_ids"][:n] if "episode_ids" in data else np.asarray([""] * n)
    frame_ids = data["frame_ids"][:n] if "frame_ids" in data else np.full(n, -1)

    for i in tqdm(range(n), desc="Computing EgoDex 3D metrics"):
        if i % args.num_shards != args.shard_id:
            continue  # window owned by another shard (num_shards=1 -> condition never true)
        mask = expand_entity_mask(action_masks[i])
        gt = reconstruct_hand_joints(gt_actions[i], current_states[i], normalizer, mano_layer, args.device)
        best_record = None
        best_score = np.inf
        best_k = 0

        candidate_indices = (
            (int(reused_best_indices[i]),) if reused_best_indices is not None else range(k)
        )
        for j in candidate_indices:
            pred = reconstruct_hand_joints(sampled_actions[i, j], current_states[i], normalizer, mano_layer, args.device)
            record = side_errors(pred, gt, mask)
            candidate_mpjpe[i, j] = record["mpjpe"]
            score = record["mpjpe"] if not np.isnan(record["mpjpe"]) else np.inf
            if best_record is None or score < best_score:
                best_score = score
                best_record = record
                best_k = j

        if best_record is None:
            best_record = {
                "left_mpjpe": np.nan,
                "left_fde": np.nan,
                "left_frames": 0,
                "right_mpjpe": np.nan,
                "right_fde": np.nan,
                "right_wrist_mpjpe": np.nan,
                "right_finger_relative_mpjpe": np.nan,
                "left_wrist_mpjpe": np.nan,
                "left_finger_relative_mpjpe": np.nan,
                "right_frames": 0,
                "mpjpe": np.nan,
                "fde": np.nan,
                "wrist_mpjpe": np.nan,
                "finger_relative_mpjpe": np.nan,
                "frames": 0,
            }

        best_indices[i] = best_k
        best_pred_actions[i] = sampled_actions[i, best_k]
        sample_mpjpe[i] = best_record["mpjpe"]
        sample_fde[i] = best_record["fde"]
        sample_wrist_mpjpe[i] = best_record["wrist_mpjpe"]
        sample_finger_relative_mpjpe[i] = best_record["finger_relative_mpjpe"]
        left_mpjpe[i] = best_record["left_mpjpe"]
        right_mpjpe[i] = best_record["right_mpjpe"]
        left_wrist_mpjpe[i] = best_record["left_wrist_mpjpe"]
        right_wrist_mpjpe[i] = best_record["right_wrist_mpjpe"]
        left_finger_relative_mpjpe[i] = best_record["left_finger_relative_mpjpe"]
        right_finger_relative_mpjpe[i] = best_record["right_finger_relative_mpjpe"]

        records.append(
            {
                "sample_index": i,
                "episode_id": str(episode_ids[i]),
                "frame_id": int(frame_ids[i]),
                "best_k": int(best_k),
                **best_record,
                "mpjpe_mm": None if np.isnan(best_record["mpjpe"]) else best_record["mpjpe"] * 1000.0,
                "fde_mm": None if np.isnan(best_record["fde"]) else best_record["fde"] * 1000.0,
                "wrist_mpjpe_mm": (
                    None
                    if np.isnan(best_record["wrist_mpjpe"])
                    else best_record["wrist_mpjpe"] * 1000.0
                ),
                "finger_relative_mpjpe_mm": (
                    None
                    if np.isnan(best_record["finger_relative_mpjpe"])
                    else best_record["finger_relative_mpjpe"] * 1000.0
                ),
            }
        )

    output = {
        "eval_npz": str(args.eval_npz),
        "statistics_path": str(args.statistics_path),
        "mano_model_path": str(args.mano_model_path),
        "best_of_k": int(k),
        "summary": summarize(records),
        "records": records,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    if args.output_npz is not None:
        Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_npz,
            best_indices=best_indices,
            best_pred_actions=best_pred_actions,
            sample_mpjpe=sample_mpjpe,
            sample_fde=sample_fde,
            sample_wrist_mpjpe=sample_wrist_mpjpe,
            sample_finger_relative_mpjpe=sample_finger_relative_mpjpe,
            left_mpjpe=left_mpjpe,
            right_mpjpe=right_mpjpe,
            left_wrist_mpjpe=left_wrist_mpjpe,
            right_wrist_mpjpe=right_wrist_mpjpe,
            left_finger_relative_mpjpe=left_finger_relative_mpjpe,
            right_finger_relative_mpjpe=right_finger_relative_mpjpe,
            candidate_mpjpe=candidate_mpjpe,
        )

    summary = output["summary"]
    print(f"Best-of-{k} EgoDex MPJPE: {summary['overall_mpjpe_mm']:.2f} mm")
    print(f"Best-of-{k} EgoDex FDE:   {summary['overall_fde_mm']:.2f} mm")
    print(f"Best-of-{k} wrist MPJPE:  {summary['overall_wrist_mpjpe_mm']:.2f} mm")
    print(
        f"Best-of-{k} finger-relative MPJPE: "
        f"{summary['overall_finger_relative_mpjpe_mm']:.2f} mm"
    )
    print(f"Wrote metrics to {args.output_json}")
    if args.output_npz is not None:
        print(f"Wrote compact arrays to {args.output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

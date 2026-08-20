"""Compute normalization statistics for one configured hand dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from empire.datasets.dataset import FrameDataset
from empire.utils.config_utils import load_config


def _mean_std(values: list[list[float]], name: str) -> tuple[list[float], list[float]]:
    if not values:
        raise ValueError(f"No valid {name} values were found")
    array = np.asarray(values)
    return np.mean(array, axis=0).tolist(), np.std(array, axis=0).tolist()


def compute_statistics(
    dataset: FrameDataset,
    output: str | Path,
    *,
    num_workers: int = 16,
    batch_size: int = 128,
    use_obj: bool = False,
) -> dict:
    """Compute state/action mean and standard deviation and write JSON."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    values = {
        "state_left": [],
        "state_right": [],
        "action_left": [],
        "action_right": [],
        "state_obj": [],
        "action_obj": [],
    }
    trajectory_count = 0

    for batch in tqdm(loader, desc="Processing dataset"):
        current_state = batch["current_state"].numpy()
        current_state_mask = batch["current_state_mask"].numpy()
        actions = batch["action_list"].numpy()
        action_mask = batch["action_mask"].numpy()
        entity_count = action_mask.shape[2]
        if entity_count not in (2, 3):
            raise ValueError(f"Expected two or three entities, found {entity_count}")

        values["action_left"].extend(actions[action_mask[:, :, 0], :51].tolist())
        values["action_right"].extend(actions[action_mask[:, :, 1], 51:102].tolist())
        values["state_left"].extend(current_state[current_state_mask[:, 0], :61].tolist())
        values["state_right"].extend(current_state[current_state_mask[:, 1], 61:122].tolist())
        if entity_count == 3:
            values["action_obj"].extend(actions[action_mask[:, :, 2], 102:108].tolist())
            values["state_obj"].extend(current_state[current_state_mask[:, 2], 122:128].tolist())
        trajectory_count += int(current_state.shape[0])

    result = {
        "dataset_name": dataset.dataset_name,
        "action_type": dataset.action_type,
        "num_traj": trajectory_count,
        "use_obj": use_obj,
    }
    for key in ("state_left", "action_left", "state_right", "action_right"):
        mean, std = _mean_std(values[key], key)
        result[key] = {"mean": mean, "std": std}
    if use_obj:
        for key in ("state_obj", "action_obj"):
            mean, std = _mean_std(values[key], key)
            result[key] = {"mean": mean, "std": std}

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Statistics written to {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate hand-dataset statistics")
    parser.add_argument("--config", help="Training config supplying data root and dataset name")
    parser.add_argument("--data-root", help="Root containing <dataset-name>/episode_frame_index.npz")
    parser.add_argument("--dataset-name", help="Dataset directory name")
    parser.add_argument("--output", help="Required output JSON path unless supplied by config")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--flip-augmentation", type=float, default=1.0)
    parser.add_argument("--action-future-window-size", type=int, default=0)
    parser.add_argument("--action-type", choices=("angle", "keypoints"), default="angle")
    parser.add_argument("--rel-mode", choices=("step", "anchor"), default="step")
    parser.add_argument("--use-rel", action="store_true")
    parser.add_argument("--use-obj", action="store_true")
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--source-video-fps", type=float, default=30.0)
    parser.add_argument("--no-filter-samples-by-fps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    train_config = config.get("train_dataset", {})
    data_root = args.data_root or train_config.get("data_root_dir")
    dataset_name = args.dataset_name or train_config.get("data_mix")
    output = args.output or config.get("statistics_output")
    missing = [
        flag
        for flag, value in (
            ("--data-root", data_root),
            ("--dataset-name", dataset_name),
            ("--output", output),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required value(s): " + ", ".join(missing) + ". Pass them explicitly or provide them in --config."
        )

    dataset = FrameDataset(
        dataset_folder=data_root,
        dataset_name=dataset_name,
        action_past_window_size=0,
        action_future_window_size=args.action_future_window_size,
        augmentation=args.augmentation,
        flip_augmentation=args.flip_augmentation,
        set_none_ratio=0.0,
        action_type=args.action_type,
        use_rel=args.use_rel,
        use_obj=args.use_obj,
        rel_mode=args.rel_mode,
        load_images=False,
        state_mask_prob=0.0,
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        source_video_fps=args.source_video_fps,
        filter_samples_by_fps=not args.no_filter_samples_by_fps,
    )
    compute_statistics(
        dataset,
        output,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        use_obj=args.use_obj,
    )


if __name__ == "__main__":
    main()

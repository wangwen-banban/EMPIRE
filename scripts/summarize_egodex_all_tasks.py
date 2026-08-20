#!/usr/bin/env python3
"""Summarize EgoDex metrics by task and select diverse qualitative cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from empire.datasets.schema import load_episode_metadata, resolve_dataset_layout


def nanmean(records: list[dict], key: str) -> float:
    values = np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def summarize(records: list[dict]) -> dict:
    return {
        "num_samples": len(records),
        "num_episodes": len({record["episode_id"] for record in records}),
        "overall_mpjpe_mm": nanmean(records, "mpjpe") * 1000.0,
        "overall_fde_mm": nanmean(records, "fde") * 1000.0,
        "overall_wrist_mpjpe_mm": nanmean(records, "wrist_mpjpe") * 1000.0,
        "overall_finger_relative_mpjpe_mm": nanmean(
            records, "finger_relative_mpjpe"
        )
        * 1000.0,
        "left_mpjpe_mm": nanmean(records, "left_mpjpe") * 1000.0,
        "right_mpjpe_mm": nanmean(records, "right_mpjpe") * 1000.0,
        "left_wrist_mpjpe_mm": nanmean(records, "left_wrist_mpjpe") * 1000.0,
        "right_wrist_mpjpe_mm": nanmean(records, "right_wrist_mpjpe") * 1000.0,
        "left_finger_relative_mpjpe_mm": nanmean(
            records, "left_finger_relative_mpjpe"
        )
        * 1000.0,
        "right_finger_relative_mpjpe_mm": nanmean(
            records, "right_finger_relative_mpjpe"
        )
        * 1000.0,
    }


def training_task_counts(data_root: Path, dataset_name: str) -> Counter:
    layout = resolve_dataset_layout(data_root, dataset_name)
    counts = Counter()
    pattern = re.compile(r"^egodex_(.+)_\d+_ep_\d+$")
    for path in sorted(layout.annotations_dir.glob("*.npy")):
        match = pattern.match(path.stem)
        if match:
            counts[match.group(1)] += 1
            continue
        episode = load_episode_metadata(layout, path.stem)
        counts[str(episode.get("obj_class", "unknown"))] += 1
    return counts


def test_episode_tasks(data_root: Path, dataset_name: str) -> tuple[dict[str, str], Counter]:
    layout = resolve_dataset_layout(data_root, dataset_name)
    mapping = {}
    counts = Counter()
    pattern = re.compile(r"^egodex_test-(.+)-\d+_ep_\d+$")
    for path in sorted(layout.annotations_dir.glob("*.npy")):
        match = pattern.match(path.stem)
        if match:
            task = match.group(1)
        else:
            episode = load_episode_metadata(layout, path.stem)
            video_parts = Path(str(episode.get("video_name", ""))).parts
            if len(video_parts) >= 3 and video_parts[0] == "test":
                task = video_parts[1]
            else:
                task = str(episode.get("obj_class", "unknown"))
        mapping[path.stem] = task
        counts[task] += 1
    return mapping, counts


def raw_test_counts(raw_test_root: Path) -> Counter:
    return Counter(
        {
            task_dir.name: len(list(task_dir.glob("*.hdf5")))
            for task_dir in sorted(raw_test_root.iterdir())
            if task_dir.is_dir()
        }
    )


def task_fingerprint(counts: Counter) -> str:
    payload = "\n".join(f"{task}\t{int(count)}" for task, count in sorted(counts.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def diverse_group(records: list[dict], count: int, seen_tasks: set[str]) -> list[dict]:
    valid = [record for record in records if record.get("mpjpe_mm") is not None]
    valid.sort(key=lambda record: float(record["mpjpe_mm"]))
    if not valid or count <= 0:
        return []

    groups = {
        "easy": valid[: max(count * 4, count)],
        "medium": valid[
            max(0, len(valid) // 2 - count * 2) : min(len(valid), len(valid) // 2 + count * 2)
        ],
        "hard": list(reversed(valid[-max(count * 4, count) :])),
    }
    selected = []
    selected_tasks = set()
    per_group = max(1, count // 3)
    for difficulty in ("easy", "medium", "hard"):
        candidates = groups[difficulty]
        added = 0
        for record in candidates:
            if record["task"] in selected_tasks:
                continue
            selected.append(
                {
                    "group": difficulty,
                    "domain": "seen" if record["task"] in seen_tasks else "unseen",
                    "task": record["task"],
                    "sample_index": int(record["sample_index"]),
                    "episode_id": record["episode_id"],
                    "frame_id": int(record["frame_id"]),
                    "key": f"{record['episode_id']}_{int(record['frame_id']):06d}",
                }
            )
            selected_tasks.add(record["task"])
            added += 1
            if added >= per_group:
                break

    for record in valid:
        if len(selected) >= count:
            break
        if record["task"] in selected_tasks:
            continue
        selected.append(
            {
                "group": "medium",
                "domain": "seen" if record["task"] in seen_tasks else "unseen",
                "task": record["task"],
                "sample_index": int(record["sample_index"]),
                "episode_id": record["episode_id"],
                "frame_id": int(record["frame_id"]),
                "key": f"{record['episode_id']}_{int(record['frame_id']):06d}",
            }
        )
        selected_tasks.add(record["task"])
    return selected[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_json", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_data_root", type=Path, required=True)
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--raw_test_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--selection_json", type=Path, required=True)
    parser.add_argument("--seen_cases", type=int, default=6)
    parser.add_argument("--unseen_cases", type=int, default=6)
    args = parser.parse_args()

    with args.metrics_json.open() as file:
        metrics = json.load(file)

    train_counts = training_task_counts(args.train_data_root, args.train_dataset_name)
    seen_tasks = set(train_counts)
    episode_to_task, test_episode_counts = test_episode_tasks(args.data_root, args.dataset_name)
    raw_counts = raw_test_counts(args.raw_test_root)

    records = []
    records_by_task = defaultdict(list)
    for record in metrics.get("records", []):
        enriched = dict(record)
        task = episode_to_task.get(str(record["episode_id"]), "unknown")
        enriched["task"] = task
        enriched["seen_in_training"] = task in seen_tasks
        records.append(enriched)
        records_by_task[task].append(enriched)

    per_task = []
    for task in sorted(set(raw_counts) | set(records_by_task) | seen_tasks):
        task_records = records_by_task.get(task, [])
        task_summary = summarize(task_records) if task_records else {
            "num_samples": 0,
            "num_episodes": 0,
            "overall_mpjpe_mm": None,
            "overall_fde_mm": None,
            "overall_wrist_mpjpe_mm": None,
            "overall_finger_relative_mpjpe_mm": None,
            "left_mpjpe_mm": None,
            "right_mpjpe_mm": None,
            "left_wrist_mpjpe_mm": None,
            "right_wrist_mpjpe_mm": None,
            "left_finger_relative_mpjpe_mm": None,
            "right_finger_relative_mpjpe_mm": None,
        }
        per_task.append(
            {
                "task": task,
                "seen_in_training": task in seen_tasks,
                "training_episodes": int(train_counts.get(task, 0)),
                "raw_test_files": int(raw_counts.get(task, 0)),
                "prepared_test_episodes": int(test_episode_counts.get(task, 0)),
                "evaluable": bool(task_records),
                **task_summary,
            }
        )

    seen_records = [record for record in records if record["seen_in_training"]]
    unseen_records = [record for record in records if not record["seen_in_training"]]
    output = {
        "metrics_json": str(args.metrics_json),
        "data_root": str(args.data_root),
        "dataset_name": args.dataset_name,
        "train_data_root": str(args.train_data_root),
        "train_dataset_name": args.train_dataset_name,
        "raw_test_root": str(args.raw_test_root),
        "seen_unseen_definition": {
            "rule": (
                "A test task is seen iff the task appears in the training "
                "annotation snapshot supplied by --train_annotation_root."
            ),
            "training_task_fingerprint_sha256": task_fingerprint(train_counts),
        },
        "training_task_count": len(seen_tasks),
        "raw_test_task_count": len(raw_counts),
        "evaluable_task_count": len(records_by_task),
        "training_tasks": [
            {"task": task, "episodes": int(count)} for task, count in sorted(train_counts.items())
        ],
        "overall": metrics.get("summary", {}),
        "domain_summary": {
            "seen": summarize(seen_records),
            "unseen": summarize(unseen_records),
        },
        "per_task": per_task,
    }

    selection = {
        "selection_policy": "Diverse tasks across seen/unseen and easy/medium/hard best-of-K MPJPE.",
        "items": diverse_group(seen_records, args.seen_cases, seen_tasks)
        + diverse_group(unseen_records, args.unseen_cases, seen_tasks),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as file:
        json.dump(output, file, indent=2)
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    with args.selection_json.open("w") as file:
        json.dump(selection, file, indent=2)

    print(
        f"Wrote {args.output_json}: {len(records_by_task)} evaluable tasks, "
        f"{len(seen_tasks)} training tasks, {len(raw_counts)} raw test tasks"
    )
    print(f"Wrote {args.selection_json}: {len(selection['items'])} qualitative cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

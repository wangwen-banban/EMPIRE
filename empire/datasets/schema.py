"""Public VITRA-1M-compatible dataset layout helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DatasetLayout:
    dataset_dir: Path
    index_file: Path
    annotations_dir: Path
    statistics_file: Path
    videos_dir: Path
    layout_format: str = "hf"


def _layout_candidates(
    root: Path,
    dataset_name: str,
    configured_video_root: str | os.PathLike[str] | None,
) -> list[DatasetLayout]:
    dataset_dir = root / dataset_name
    preferred_videos = (
        Path(configured_video_root).expanduser() / dataset_name
        if configured_video_root
        else dataset_dir / "videos"
    )
    candidates = [
        DatasetLayout(
            dataset_dir=dataset_dir,
            index_file=dataset_dir / "episode_frame_index.npz",
            annotations_dir=dataset_dir / "episodic_annotations",
            statistics_file=dataset_dir / "statistics.json",
            videos_dir=preferred_videos,
            layout_format="hf",
        )
    ]

    legacy_roots = (dataset_dir, root / "Annotation" / dataset_name)
    for legacy_root in legacy_roots:
        legacy_videos = (
            Path(configured_video_root).expanduser() / dataset_name
            if configured_video_root
            else root / "Video" / dataset_name
        )
        statistics_file = (
            legacy_root / "statistics.json"
            if legacy_root == dataset_dir
            else root / "Annotation" / "statistics" / f"{dataset_name}_angle_statistics.json"
        )
        for index_name in ("episode_frame_index.npy", "episode_frame_index.npz"):
            for annotations_name in ("processed_episodes_vitra_format", "episodic_annotations"):
                candidates.append(
                    DatasetLayout(
                        dataset_dir=legacy_root,
                        index_file=legacy_root / index_name,
                        annotations_dir=legacy_root / annotations_name,
                        statistics_file=statistics_file,
                        videos_dir=legacy_videos,
                        layout_format="legacy",
                    )
                )
    return candidates


def resolve_dataset_layout(
    data_root: str | os.PathLike[str],
    dataset_name: str,
    video_root: str | os.PathLike[str] | None = None,
    *,
    require: bool = True,
) -> DatasetLayout:
    """Resolve the preferred or legacy runtime layout without path traversal.

    The preferred Hugging Face layout is checked first. Legacy lookup is
    read-only and supports the established VITRA ``Annotation`` tree and
    dictionary-backed ``.npy`` index.
    """
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("dataset_name must be one non-empty path component")
    if not data_root:
        raise ValueError("data_root is required")
    root = Path(data_root).expanduser()
    configured_video_root = video_root if video_root is not None else os.environ.get("VIDEO_ROOT")
    candidates = _layout_candidates(root, dataset_name, configured_video_root)
    for layout in candidates:
        if layout.index_file.is_file() and layout.annotations_dir.is_dir():
            return layout
    if not require:
        return candidates[0]

    attempted_indexes = sorted({str(layout.index_file) for layout in candidates})
    attempted_annotations = sorted({str(layout.annotations_dir) for layout in candidates})
    raise FileNotFoundError(
        "No complete dataset layout found. Expected an index in "
        + ", ".join(attempted_indexes)
        + " and a matching annotation directory in "
        + ", ".join(attempted_annotations)
    )


def load_dataset_index(layout: DatasetLayout) -> dict[str, np.ndarray]:
    """Load and validate preferred ``.npz`` or legacy dictionary ``.npy`` indices."""
    loaded = np.load(layout.index_file, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            payload = {key: loaded[key] for key in loaded.files}
        finally:
            loaded.close()
    else:
        try:
            payload = loaded.item()
        except (ValueError, AttributeError) as exc:
            raise ValueError("Legacy index must contain one dictionary") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Legacy index must contain one dictionary")

    required = {"index_frame_pair", "index_to_episode_id"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("Index is missing required array(s): " + ", ".join(missing))
    frame_pairs = np.asarray(payload["index_frame_pair"])
    episode_ids = np.asarray(payload["index_to_episode_id"], dtype=object)
    if frame_pairs.ndim != 2 or frame_pairs.shape[1] != 2:
        raise ValueError("index_frame_pair must have shape [N, 2]")
    if episode_ids.ndim != 1:
        raise ValueError("index_to_episode_id must be one-dimensional")
    return {"index_frame_pair": frame_pairs, "index_to_episode_id": episode_ids}


def load_episode_metadata(layout: DatasetLayout, episode_id: str) -> dict[str, Any]:
    """Load one episode dictionary and expose public caption/plan aliases."""
    if not episode_id or Path(episode_id).name != episode_id:
        raise ValueError("episode_id must be one non-empty path component")
    path = layout.annotations_dir / f"{episode_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Episode annotation does not exist: {path}")
    value = np.load(path, allow_pickle=True)
    try:
        metadata = value.item()
    except ValueError as exc:
        raise ValueError(f"Episode annotation must contain one dictionary: {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Episode annotation must contain one dictionary: {path}")
    result = dict(metadata)
    if "plan" in result and "cot" not in result:
        result["cot"] = result["plan"]
    if "caption" in result and "instruction" not in result:
        result["instruction"] = result["caption"]
    return result


SUPPORTED_PREDICTED_PLAN_SCHEMA_VERSIONS = frozenset({"1.0"})


def validate_sample_id(sample_id: str) -> str:
    """Validate and return canonical ``<episode_id>:<frame_id>`` syntax."""
    if not isinstance(sample_id, str):
        raise ValueError("Predicted-plan sample IDs must be strings")
    if sample_id.count(":") != 1:
        raise ValueError(
            f"Predicted-plan sample ID {sample_id!r} must match <episode_id>:<frame_id>"
        )
    episode_id, frame_text = sample_id.split(":")
    if (
        not episode_id
        or episode_id in {".", ".."}
        or Path(episode_id).name != episode_id
        or "/" in episode_id
        or "\\" in episode_id
        or any(character.isspace() for character in episode_id)
    ):
        raise ValueError(f"Invalid episode_id in predicted-plan sample ID: {sample_id!r}")
    try:
        frame_id = int(frame_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid frame_id in predicted-plan sample ID: {sample_id!r}") from exc
    if frame_id < 0 or frame_text != str(frame_id):
        raise ValueError(f"Invalid canonical frame_id in predicted-plan sample ID: {sample_id!r}")
    return sample_id


def canonical_sample_id(episode_id: str, frame_id: Integral) -> str:
    """Build the exact sample ID used by indices, datasets, and sidecars."""
    if isinstance(frame_id, bool) or not isinstance(frame_id, Integral):
        raise ValueError("frame_id must be a non-negative integer")
    return validate_sample_id(f"{episode_id}:{int(frame_id)}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON object key in predicted-plan sidecar: {key!r}")
        value[key] = item
    return value


def load_predicted_plan_sidecar(path: str | os.PathLike[str]) -> dict[str, str]:
    """Validate versioned ABI 1.0 or the explicit legacy raw-map branch.

    Versioned sidecars require both ``schema_version`` and a non-empty ``plans``
    object. Legacy sidecars are raw canonical-ID-to-string objects. Both forms
    reject duplicate JSON keys and non-canonical sample IDs.
    """
    sidecar_path = Path(path).expanduser()
    with sidecar_path.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_object_without_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("Predicted-plan sidecar must be a JSON object")

    is_versioned = "schema_version" in payload or "plans" in payload
    if is_versioned:
        if "schema_version" not in payload:
            raise ValueError("Versioned predicted-plan sidecar requires schema_version")
        schema_version = payload["schema_version"]
        if (
            not isinstance(schema_version, str)
            or schema_version not in SUPPORTED_PREDICTED_PLAN_SCHEMA_VERSIONS
        ):
            raise ValueError(f"Unsupported predicted-plan schema_version: {schema_version!r}")
        if "plans" not in payload:
            raise ValueError("Versioned predicted-plan sidecar requires a plans object")
        plans = payload["plans"]
        if not isinstance(plans, dict):
            raise ValueError("Versioned predicted-plan plans must be a JSON object")
    else:
        plans = payload

    if not plans:
        raise ValueError("Predicted-plan sidecar must contain at least one plan")

    normalized: dict[str, str] = {}
    for sample_id, entry in plans.items():
        validate_sample_id(sample_id)
        if is_versioned:
            if isinstance(entry, str):
                plan = entry
            elif isinstance(entry, dict):
                if "plan" not in entry:
                    raise ValueError(f"Predicted-plan entry for {sample_id!r} requires a plan field")
                plan = entry["plan"]
            else:
                raise ValueError(
                    f"Predicted-plan entry for {sample_id!r} must be a string or object"
                )
        else:
            if not isinstance(entry, str):
                raise ValueError(
                    f"Legacy predicted-plan value for {sample_id!r} must be a string"
                )
            plan = entry
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError(f"Predicted plan for {sample_id!r} must be a non-empty string")
        normalized[sample_id] = plan
    return normalized

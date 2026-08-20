# VITRA-1M-compatible data schema

The dataset itself and preprocessing programs are not distributed here. The preferred runtime layout is:

```text
${DATA_ROOT}/<dataset_name>/
├── episode_frame_index.npz
├── episodic_annotations/
│   ├── <episode_id>.npy
│   └── ...
├── statistics.json                 # required for normalized training
└── predicted_plans.json            # optional Stage II sidecar

${VIDEO_ROOT}/<dataset_name>/
└── <video_name>[.mp4]
```

`dataset_name` must be one path component. The release recipes use `empire_train`; another name can be supplied from the command line.

## Legacy runtime compatibility

For existing VITRA-format exports, the resolver also accepts either a direct dataset directory or this generic legacy layout:

```text
${DATA_ROOT}/Annotation/<dataset_name>/
├── episode_frame_index.npy         # dictionary containing the two index arrays
└── processed_episodes_vitra_format/
    ├── <episode_id>.npy
    └── ...

${DATA_ROOT}/Annotation/statistics/
└── <dataset_name>_angle_statistics.json
```

A legacy `.npz` index and either annotation-directory name are also accepted. `${VIDEO_ROOT}/<dataset_name>` takes precedence when `VIDEO_ROOT` is supplied; otherwise the resolver uses the corresponding standard or legacy video directory. Dataset names are configuration values rather than entries in a closed registry, so any safe one-component name works when one of these layouts is complete. FrameDataset, statistics, task summarization, and evaluation episode loading all use this resolver. This is read-only runtime compatibility, not a converter or preprocessing implementation.

## Frame index

`episode_frame_index.npz` contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `index_frame_pair` | `[N, 2]` | Each row is `[episode_index, frame_index]` |
| `index_to_episode_id` | `[E]` | String episode ID for each episode index |

The legacy `.npy` form stores the same arrays in one dictionary. The loader uses `allow_pickle=True` for compatibility with the established NumPy ABI. Load only annotations from a trusted publisher.

## Episode metadata

Each `episodic_annotations/<episode_id>.npy` stores one Python dictionary. The motion loader consumes:

- `video_name`, `video_decode_frame`, `intrinsics`, and per-frame `extrinsics`;
- `left` and `right` dictionaries with `beta`, `kept_frames`, `global_orient_worldspace`, `transl_worldspace`, `hand_pose`, and `joints_worldspace`;
- `text` and `anno_type` for hand activity labels;
- optional `fps` (or `annotation_fps`) for temporal sampling;
- optional `caption` and `plan` fields.

`caption` is accepted as an alias for `instruction`. `plan` is accepted as an alias for the established runtime field. A plan is a list of text/range entries such as:

```python
[[["reach toward the object", "close the fingers"], [0, 59]]]
```

Object trajectories are optional and use `obj_rt` plus `obj_mesh_path`. Mesh files and MANO model assets are not part of this release.

## Predicted-plan sidecar ABI

Stage II predicted-plan training uses a UTF-8 JSON sidecar keyed by stable sample ID. ABI version `1.0` defines the canonical ID as `<episode_id>:<frame_index>`; the frame index is the second value in `index_frame_pair`. The frame index is canonical non-negative decimal text (for example `4`, not `04`), and the episode ID is one non-empty path component without whitespace or `:`.

```json
{
  "schema_version": "1.0",
  "plans": {
    "episode_0001:24": {
      "plan": "reach, align the wrist, then close the fingers"
    }
  }
}
```

A versioned plan value may also be a string directly. A versioned sidecar must be an object containing supported string `schema_version` and non-empty object `plans`; neither field is inferred. Non-canonical IDs, duplicate JSON keys, empty plans, missing `plan` fields, wrong value types, and unsupported versions are rejected. The explicit legacy branch accepts only a non-empty raw canonical-ID-to-string object—nested entries and text-keyed lookup are not part of that ABI. The `predicted_plan` ablation reads `${DATA_ROOT}/empire_train/predicted_plans.json`.

## Synthetic validation

`tests/test_data_schema.py` creates preferred and generic legacy indices and episode dictionaries in temporary directories, then crosses FrameDataset metadata parsing, task summarization, and evaluation loading for both layouts. It also exercises positive and negative sidecar ABI cases. `tests/test_predicted_plan_source.py` validates canonical runtime IDs and fail-closed sidecar selection. No real videos or dataset files are needed.

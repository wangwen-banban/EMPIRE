# EMPIRE

EMPIRE is a two-stage system for egocentric hand-motion prediction. Stage I uses a PaliGemma 2 planner with RGB, Depth Anything 3 features, caption, and field-of-view conditioning. Stage II freezes the planner and condition encoders, then trains a DiT-B actor with flow matching to predict 60 future frames at 12 FPS.

This public research release contains the final Stage I and Stage II recipes, paper-table ablations, runtime loaders, training code, and evaluation utilities. Model weights, MANO assets, dataset files, and preprocessing code are not included.

> **License note:** DiT-derived files are under Creative Commons Attribution-NonCommercial 4.0 and restrict commercial use. The root MIT license does not override third-party terms. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before use or redistribution.

## Paper

- ArXiv: Coming Soon

## Dataset

Dataset: **Coming Soon**.

The loader accepts the documented VITRA-1M-compatible layout and its generic legacy runtime layout. See [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) for index arrays, episode metadata, video layout, and the versioned predicted-plan sidecar ABI. Synthetic Hugging Face-style and legacy fixtures exercise the complete FrameDataset, statistics-summary, and evaluation-loading paths without downloading a dataset.

## Installation

The base wheel has a lightweight import. The supported public environment, including the release tests, is installed in one resolver pass:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[public,test]'
```

Granular `model`, `data`, `depth`, `train`, and `eval` extras remain available. The `data` and aggregate `public` extras include the pinned EasternJournalist `utils3d` distribution (version `0.0.2`, used by camera/intrinsics augmentation) and `trimesh>=4.11,<5` (used by object-mesh loading and sampling). The similarly named PyPI `utils3d` project exposes a different API, so the compatible distribution is pinned to its verified VCS revision in `pyproject.toml`; no follow-up manual install is required.

Some evaluation paths also require a separately installed `manopth` implementation. MANO model assets have their own terms and must be obtained by the user; set `MANO_MODEL_PATH` rather than placing assets in this repository.

## Required environment

The public JSON recipes use these variables:

| Variable | Meaning |
| --- | --- |
| `DATA_ROOT` | Root containing one directory per dataset name |
| `VIDEO_ROOT` | Root containing the matching video directories |
| `OUTPUT_ROOT` | Writable root for logs and checkpoints |
| `STAGE1_CHECKPOINT` | Stage I checkpoint used to initialize Stage II |

Unset variables produce a clear error when a config is loaded. Model backbones use the public Hugging Face IDs `google/paligemma2-3b-mix-224` and `depth-anything/DA3METRIC-LARGE`, not local snapshot paths.

## Training

Stage I:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config empire/configs/train_stage1_planner.json
```

Stage II:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config empire/configs/train_stage2_actor.json
```

Both recipes are fresh-start recipes (`resume=false`). Stage I runs for 10,171 optimizer steps at planner learning rate `1e-5`; Stage II runs for 40,684 steps at actor learning rate `1e-4`. The global batch size is 64 in both stages.

See [docs/stage2_training_checkpoint_and_freeze_guide.md](docs/stage2_training_checkpoint_and_freeze_guide.md), [docs/lr_schedule_config.md](docs/lr_schedule_config.md), and [docs/flow_matching_dit_usage.md](docs/flow_matching_dit_usage.md).

## Paper-table configurations

The files under `empire/configs/ablations/` correspond only to reported comparisons:

- Table 4: `baseline_caption_only`, `plan_conditioning`, `depth_cross_attention`, `depth_concatenation`
- Table 5: `single_stage_joint_training`, `trainable_planner`, `frozen_planner`
- Table 6: `dit_small`, `dit_base`, `dit_large`
- Supplementary plan comparison: `ground_truth_plan`, `predicted_plan`

Configuration evidence for the two plan comparisons is explicit:

| Reported comparison | Recipe evidence |
| --- | --- |
| Table 4 `+ motion plan` | `plan_conditioning` is the no-depth pilot (`use_depth=false`) with plan-span cross-attention (`train_dataset.use_cot=true`, `cot_condition_span=true`, `condition_mode=vlm_cross_attention`), DiT-B, batch `8` / global batch `64`, and `20,000` steps. It is therefore not a renamed full-data GT-plan recipe. |
| Supplementary GT plan | `ground_truth_plan` uses the full Stage II conditions: DiT-B flow matching, depth concatenation, 60 frames, batch `4` / global batch `64`, four epochs / `40,684` steps, and annotation plans (no `cot_source` override). |
| Supplementary predicted plan | `predicted_plan` has the same architecture, data, checkpoint, depth, batch, and schedule as `ground_truth_plan`; only `task_name`, `train_dataset.cot_source=predicted`, and `train_dataset.predicted_plan_sidecar` differ. |

`tests/test_public_configs.py` recursively enforces the supplementary pair equality after removing those three plan-source ABI fields.

## Statistics and evaluation

```bash
bash scripts/compute_statistic.sh
bash scripts/run_full_eval.sh
```

Both entry points are environment-driven and fail before work begins when required paths are absent. `scripts/run_full_eval.sh` lists all required variables at its start. Evaluation and training task assignment use `DATA_ROOT` / `DATASET_NAME` and `TRAIN_DATA_ROOT` / `TRAIN_DATASET_NAME` respectively (`TRAIN_DATA_ROOT` defaults to `DATA_ROOT`); both pairs are resolved through the same preferred/legacy schema resolver before any GPU work starts.

## Acknowledgements

We sincerely thank the authors of [VITRA](https://github.com/microsoft/VITRA) for releasing their code and the VITRA-1M data format, which provided an important foundation for this work. We also thank the authors of [Being-H0](https://github.com/BeingBeyond/Being-H0) for releasing their models and code, which supported the comparative evaluation in EMPIRE.

## Licensing

`LICENSE` preserves the MIT license supplied with the VITRA-derived source. Bundled and adapted third-party files have additional terms. Complete notices and redistributed license material are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).

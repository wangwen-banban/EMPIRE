# EMPIRE

**Explicit Manipulation Planning as a Learnable Intermediate Representation for Egocentric Hand-Motion Forecasting**

Forecasting dexterous hand motions from egocentric observations is fundamental to intelligent interactive systems. Existing vision-language approaches typically map observations directly to future motions, overlooking the manipulation process that governs hand-object interactions. EMPIRE addresses this limitation by introducing explicit manipulation planning as a learnable intermediate representation for egocentric hand-motion forecasting.

EMPIRE follows a two-stage **plan-then-act** framework:

- **Stage I — Learn to Plan:** the Planner predicts explicit, temporally ordered manipulation plans that describe the progression of per-hand interactions from multimodal egocentric context.
- **Stage II — Learn to Act:** a flow-matching Actor synthesizes future bimanual hand motions conditioned on the frozen Planner representations, decoupling high-level manipulation understanding from low-level motion generation.

Under a unified training and evaluation protocol, EMPIRE achieves an MPJPE of **84.53 mm** and a finger-relative error of **38.97 mm** on the held-out EgoDex benchmark. This repository provides the official implementation, training recipes, and evaluation code for EMPIRE.

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

## Compute dataset statistics

Dataset statistics must be generated **before Stage I training**. For the documented Hugging Face-style layout, the loader expects them at `<DATA_ROOT>/<DATASET_NAME>/statistics.json`.

```bash
export DATA_ROOT=/path/to/data
export DATASET_NAME=my_dataset
bash scripts/compute_statistic.sh
```

To write to a different location, set `STATISTICS_OUTPUT` explicitly and make the corresponding loader configuration point to that file. Object-aware statistics can be computed with `scripts/compute_statistic_with_obj.sh`.

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

## Evaluation

```bash
bash scripts/run_full_eval.sh
```

The evaluation entry point is environment-driven and fails before work begins when required paths are absent. `scripts/run_full_eval.sh` lists all required variables at its start. Evaluation and training task assignment use `DATA_ROOT` / `DATASET_NAME` and `TRAIN_DATA_ROOT` / `TRAIN_DATASET_NAME` respectively (`TRAIN_DATA_ROOT` defaults to `DATA_ROOT`); both pairs are resolved through the same preferred/legacy schema resolver before any GPU work starts.

## Acknowledgements

We sincerely thank the authors of [VITRA](https://github.com/microsoft/VITRA) for releasing their code and the VITRA-1M data format, which provided an important foundation for this work. We also thank the authors of [Being-H0](https://github.com/BeingBeyond/Being-H0) for releasing their models and code, which supported the comparative evaluation in EMPIRE.

## Licensing

`LICENSE` preserves the MIT license supplied with the VITRA-derived source. Bundled and adapted third-party files have additional terms. Complete notices and redistributed license material are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).

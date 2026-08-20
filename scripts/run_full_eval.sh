#!/usr/bin/env bash
# Run sharded prediction, best-of-K metrics, and task-level summaries.
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the public-schema dataset root}"
: "${VIDEO_ROOT:?Set VIDEO_ROOT to the video root}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a writable output root}"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT for config expansion}"
: "${MODEL_PATH:?Set MODEL_PATH to the Stage II weights file}"
: "${CONFIG_PATH:?Set CONFIG_PATH to the evaluation config}"
: "${STATISTICS_PATH:?Set STATISTICS_PATH to normalization statistics}"
: "${MANO_MODEL_PATH:?Set MANO_MODEL_PATH to separately licensed MANO assets}"
: "${DATASET_NAME:?Set DATASET_NAME to the evaluation dataset name}"
: "${TRAIN_DATASET_NAME:?Set TRAIN_DATASET_NAME for seen-task assignment}"
: "${RAW_TEST_ROOT:?Set RAW_TEST_ROOT for task metadata}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${DATA_ROOT}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
REPEAT_TIMES="${REPEAT_TIMES:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ONLINE_PLAN="${ONLINE_PLAN:-1}"
FORCE="${FORCE:-0}"
OUTPUT_DIR="${EVALUATION_OUTPUT:-${OUTPUT_ROOT}/evaluation/${DATASET_NAME}}"

for required_file in "${MODEL_PATH}" "${CONFIG_PATH}" "${STATISTICS_PATH}"; do
  [[ -f "${required_file}" ]] || { echo "Required file missing: ${required_file}" >&2; exit 2; }
done
[[ -e "${MANO_MODEL_PATH}" ]] || { echo "MANO assets missing: ${MANO_MODEL_PATH}" >&2; exit 2; }
"${PYTHON}" -c 'import sys; from empire.datasets.schema import resolve_dataset_layout; resolve_dataset_layout(sys.argv[1], sys.argv[2])' "${DATA_ROOT}" "${DATASET_NAME}"
"${PYTHON}" -c 'import sys; from empire.datasets.schema import resolve_dataset_layout; resolve_dataset_layout(sys.argv[1], sys.argv[2])' "${TRAIN_DATA_ROOT}" "${TRAIN_DATASET_NAME}"

read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"
[[ "${NUM_SHARDS}" -gt 0 ]] || { echo "GPUS must contain at least one device" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}/logs"

shards=()
pids=()
for index in "${!GPU_ARRAY[@]}"; do
  shard="${OUTPUT_DIR}/predictions_shard${index}.npz"
  shards+=("${shard}")
  if [[ -f "${shard}" && "${FORCE}" != "1" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$index]}" \
  COT_GENERATE_THEN_CONDITION="${ONLINE_PLAN}" \
  "${PYTHON}" scripts/eval_mse.py \
    --config "${CONFIG_PATH}" \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_ROOT}" \
    --dataset_name "${DATASET_NAME}" \
    --output_path "${shard}" \
    --repeated_diffusion_steps "${REPEAT_TIMES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --save_repeated_samples \
    --num_shards "${NUM_SHARDS}" \
    --shard_id "${index}" \
    >"${OUTPUT_DIR}/logs/prediction_shard${index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
[[ "${failed}" == "0" ]] || { echo "At least one prediction shard failed" >&2; exit 1; }
for shard in "${shards[@]}"; do
  [[ -f "${shard}" ]] || { echo "Prediction shard missing: ${shard}" >&2; exit 1; }
done

merged="${OUTPUT_DIR}/predictions.npz"
metrics_json="${OUTPUT_DIR}/metrics.json"
metrics_npz="${OUTPUT_DIR}/metrics.npz"
summary_json="${OUTPUT_DIR}/summary.json"
selection_json="${OUTPUT_DIR}/selection.json"

if [[ ! -f "${merged}" || "${FORCE}" == "1" ]]; then
  "${PYTHON}" scripts/merge_eval_shards.py --output_path "${merged}" --shards "${shards[@]}"
fi
if [[ ! -f "${metrics_json}" || "${FORCE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "${PYTHON}" scripts/compute_egodex_metrics.py \
    --eval_npz "${merged}" \
    --config "${CONFIG_PATH}" \
    --statistics_path "${STATISTICS_PATH}" \
    --mano_model_path "${MANO_MODEL_PATH}" \
    --output_json "${metrics_json}" \
    --output_npz "${metrics_npz}" \
    --device cuda
fi
"${PYTHON}" scripts/summarize_egodex_all_tasks.py \
  --metrics_json "${metrics_json}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --train_data_root "${TRAIN_DATA_ROOT}" \
  --train_dataset_name "${TRAIN_DATASET_NAME}" \
  --raw_test_root "${RAW_TEST_ROOT}" \
  --output_json "${summary_json}" \
  --selection_json "${selection_json}"

echo "Evaluation complete: ${summary_json}"

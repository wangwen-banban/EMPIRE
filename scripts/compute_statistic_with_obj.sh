#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
DATASET_NAME="${DATASET_NAME:-empire_train}"
"${PYTHON:-python}" -m empire.datasets.calculate_statistics \
  --data-root "${DATA_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --output "${STATISTICS_OUTPUT:-${DATA_ROOT}/${DATASET_NAME}/statistics_with_object.json}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-16}" \
  --use-obj

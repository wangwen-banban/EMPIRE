#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a writable output root}"
DATASET_NAME="${DATASET_NAME:-empire_train}"
"${PYTHON:-python}" -m empire.datasets.calculate_statistics \
  --data-root "${DATA_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --output "${OUTPUT_ROOT}/statistics/${DATASET_NAME}_angle.json" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-16}"

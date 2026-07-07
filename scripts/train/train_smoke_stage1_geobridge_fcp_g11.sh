#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

export OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/smoke_stage1_geobridge_fcp_g11_feature_knn"}
export LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/smoke_stage1_geobridge_fcp_g11_feature_knn"}
export MAX_STEPS=${MAX_STEPS:-1}
export SAVE_STEPS=${SAVE_STEPS:-1}
export LOGGING_STEPS=${LOGGING_STEPS:-1}
export MAX_GROUPS=${MAX_GROUPS:-8}
export AUTO_RESUME=${AUTO_RESUME:-"False"}
export NUM_WORKERS=${NUM_WORKERS:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

exec bash "${SCRIPT_DIR}/train_stage1_geobridge_fcp_g11.sh"

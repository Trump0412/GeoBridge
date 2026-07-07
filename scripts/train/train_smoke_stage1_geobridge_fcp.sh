#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-"$(cd "${SCRIPT_DIR}/../.." && pwd)/outputs/smoke_stage1_geobridge_fcp_feature_knn"}
LOG_DIR=${LOG_DIR:-"$(cd "${SCRIPT_DIR}/../.." && pwd)/logs/smoke_stage1_geobridge_fcp_feature_knn"}
MAX_STEPS=${MAX_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1}
LOGGING_STEPS=${LOGGING_STEPS:-1}
MAX_GROUPS=${MAX_GROUPS:-8}
AUTO_RESUME=${AUTO_RESUME:-"False"}
NUM_WORKERS=${NUM_WORKERS:-0}

export OUTPUT_DIR LOG_DIR MAX_STEPS SAVE_STEPS LOGGING_STEPS MAX_GROUPS AUTO_RESUME NUM_WORKERS
exec bash "${SCRIPT_DIR}/train_stage1_geobridge_fcp.sh"

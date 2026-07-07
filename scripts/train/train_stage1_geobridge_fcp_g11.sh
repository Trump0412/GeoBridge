#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

export FEATURE_LAYERS=${FEATURE_LAYERS:-"g11"}
export OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/stage1_geobridge_fcp_g11_feature_knn"}
export LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/stage1_geobridge_fcp_g11_feature_knn"}
export MANIFEST_PATH=${MANIFEST_PATH:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow_corrgraph_feature_knn_g11/manifest.jsonl"}

exec bash "${SCRIPT_DIR}/train_stage1_geobridge_fcp.sh"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

export INPUT_MANIFEST_PATH=${INPUT_MANIFEST_PATH:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow_corrgraph_feature_knn_g11_packed/manifest.jsonl"}
export WINDOW_READY_CACHE_DIR=${WINDOW_READY_CACHE_DIR:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker_zenview_vggt_bank/cache/stage1_geobridge_window_ready_joint_fp16_g11"}
export OUTPUT_MANIFEST_PATH=${OUTPUT_MANIFEST_PATH:-"${WINDOW_READY_CACHE_DIR}/manifest.jsonl"}

exec bash "${SCRIPT_DIR}/build_stage1_window_ready_joint_cache.sh"

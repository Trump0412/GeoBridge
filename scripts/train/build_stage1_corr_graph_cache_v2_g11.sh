#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

export FEATURE_LAYERS=${FEATURE_LAYERS:-"g11"}
export CORR_CACHE_DIR=${CORR_CACHE_DIR:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow_corrgraph_feature_knn_g11"}
export OUTPUT_MANIFEST_PATH=${OUTPUT_MANIFEST_PATH:-"${CORR_CACHE_DIR}/manifest.jsonl"}

exec bash "${SCRIPT_DIR}/build_stage1_corr_graph_cache_v2.sh"

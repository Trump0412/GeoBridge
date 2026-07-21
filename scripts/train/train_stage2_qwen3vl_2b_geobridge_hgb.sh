#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
if [[ -f "${PROJECT_ROOT}/configs/geobridge_paths.env" ]]; then
  source "${PROJECT_ROOT}/configs/geobridge_paths.env"
fi

export GEOBRIDGE_STAGE2_MODEL_PATH="${MODEL_PATH:-${QWEN3VL_2B_PATH:-models/Qwen3-VL-2B-Instruct}}"
export MODEL_PATH="${GEOBRIDGE_STAGE2_MODEL_PATH}"
export VGGT_BANK_FUSION_LAYER_INDICES="${VGGT_BANK_FUSION_LAYER_INDICES:-0,1,2}"
export VARIANT_NAME="${VARIANT_NAME:-qwen3vl_2b_spatialfit_hgb_layers012}"
export QWEN_ATTN_IMPLEMENTATION="${QWEN_ATTN_IMPLEMENTATION:-sdpa}"

exec bash "${SCRIPT_DIR}/train_stage2_qwen25vl_7b_geobridge_hgb.sh"

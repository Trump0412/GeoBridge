#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
MODEL_PATH=${MODEL_PATH:-"/data3/yeyuanhao/sp_re_cbp/thirdparty/models/Qwen2.5-VL-7B-Instruct"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker/models/VGGT-1B"}
CACHE_WINDOW_MODE=${CACHE_WINDOW_MODE:-"fixed8"}
if [ -z "${CACHE_DIR:-}" ]; then
  if [ "${CACHE_WINDOW_MODE}" = "multi_window" ]; then
    CACHE_DIR="${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow"
  else
    CACHE_DIR="${PROJECT_ROOT}/cache/zenview_continuity_bank_v2"
  fi
fi
MANIFEST_PATH=${MANIFEST_PATH:-"${CACHE_DIR}/manifest.jsonl"}
DATASETS=${DATASETS:-"llava_hound_64k,spar_234k"}
WRITE_FEATURES=${WRITE_FEATURES:-"False"}
MAX_GROUPS=${MAX_GROUPS:-"-1"}
NUM_WINDOWS_PER_SAMPLE=${NUM_WINDOWS_PER_SAMPLE:-4}
NUM_FRAMES=${NUM_FRAMES:-8}
STRIDE_MIN=${STRIDE_MIN:-4}
STRIDE_MAX=${STRIDE_MAX:-12}
PYTHON_BIN=${PYTHON_BIN:-"python"}

mkdir -p "${CACHE_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/qwen_vl/train/build_vggt_feature_cache.py" \
  --dataset_use "${DATASETS}" \
  --model_name_or_path "${MODEL_PATH}" \
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
  --cache_dir "${CACHE_DIR}" \
  --manifest_path "${MANIFEST_PATH}" \
  --write_features "${WRITE_FEATURES}" \
  --max_groups "${MAX_GROUPS}" \
  --cache_window_mode "${CACHE_WINDOW_MODE}" \
  --num_windows_per_sample "${NUM_WINDOWS_PER_SAMPLE}" \
  --num_frames "${NUM_FRAMES}" \
  --stride_min "${STRIDE_MIN}" \
  --stride_max "${STRIDE_MAX}"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

INPUT_MANIFEST_PATH=${INPUT_MANIFEST_PATH:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow_corrgraph_feature_knn_packed/manifest.jsonl"}
WINDOW_READY_CACHE_DIR=${WINDOW_READY_CACHE_DIR:-"${PROJECT_ROOT}/cache/stage1_window_ready_joint_fp16"}
OUTPUT_MANIFEST_PATH=${OUTPUT_MANIFEST_PATH:-"${WINDOW_READY_CACHE_DIR}/manifest.jsonl"}

FEATURE_DTYPE=${FEATURE_DTYPE:-"float16"}
SCORE_DTYPE=${SCORE_DTYPE:-"float16"}
MAX_SOURCES=${MAX_SOURCES:--1}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_RANK=${SHARD_RANK:-0}
OVERWRITE=${OVERWRITE:-"False"}
PYTHON_BIN=${PYTHON_BIN:-"python"}

mkdir -p "${WINDOW_READY_CACHE_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" src/qwen_vl/train/build_stage1_window_ready_joint_cache.py \
  --input_manifest_path "${INPUT_MANIFEST_PATH}" \
  --output_cache_dir "${WINDOW_READY_CACHE_DIR}" \
  --output_manifest_path "${OUTPUT_MANIFEST_PATH}" \
  --feature_dtype "${FEATURE_DTYPE}" \
  --score_dtype "${SCORE_DTYPE}" \
  --max_sources "${MAX_SOURCES}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_rank "${SHARD_RANK}" \
  --overwrite "${OVERWRITE}"

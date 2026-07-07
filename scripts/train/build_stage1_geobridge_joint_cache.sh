#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

MODEL_PATH=${MODEL_PATH:-"/data3/yeyuanhao/sp_re_cbp/thirdparty/models/Qwen2.5-VL-7B-Instruct"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker/models/VGGT-1B"}
PROJECTOR_CHECKPOINT_PATH=${PROJECTOR_CHECKPOINT_PATH:-"${PROJECT_ROOT}/outputs/stage1_continuity_v2_corrgraph_feature_knn/best.pt"}

INPUT_MANIFEST_PATH=${INPUT_MANIFEST_PATH:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow_corrgraph_feature_knn/manifest.jsonl"}
JOINT_CACHE_DIR=${JOINT_CACHE_DIR:-"/data2/yeyuanhao/sp_re_cbp/GeoThinker_zenview_vggt_bank/cache/stage1_geobridge_joint_compact_projected_int8"}
OUTPUT_MANIFEST_PATH=${OUTPUT_MANIFEST_PATH:-"${JOINT_CACHE_DIR}/manifest.jsonl"}

FEATURE_LAYERS=${FEATURE_LAYERS:-"g11,g17,g23"}
MAX_SOURCES=${MAX_SOURCES:--1}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_RANK=${SHARD_RANK:-0}
OVERWRITE=${OVERWRITE:-"False"}
PYTHON_BIN=${PYTHON_BIN:-"python"}

mkdir -p "${JOINT_CACHE_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" src/qwen_vl/train/build_stage1_joint_compact_cache.py \
  --input_manifest_path "${INPUT_MANIFEST_PATH}" \
  --output_cache_dir "${JOINT_CACHE_DIR}" \
  --output_manifest_path "${OUTPUT_MANIFEST_PATH}" \
  --projector_checkpoint_path "${PROJECTOR_CHECKPOINT_PATH}" \
  --model_name_or_path "${MODEL_PATH}" \
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
  --feature_layers "${FEATURE_LAYERS}" \
  --max_sources "${MAX_SOURCES}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_rank "${SHARD_RANK}" \
  --overwrite "${OVERWRITE}"

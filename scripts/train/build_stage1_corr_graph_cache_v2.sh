#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

BASE_CACHE_DIR=${BASE_CACHE_DIR:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow"}
BASE_MANIFEST_PATH=${BASE_MANIFEST_PATH:-"${BASE_CACHE_DIR}/manifest.jsonl"}
CORR_CACHE_DIR=${CORR_CACHE_DIR:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow_corrgraph_feature_knn"}
OUTPUT_MANIFEST_PATH=${OUTPUT_MANIFEST_PATH:-"${CORR_CACHE_DIR}/manifest.jsonl"}
PROJECTOR_CHECKPOINT_PATH=${PROJECTOR_CHECKPOINT_PATH:-"${PROJECT_ROOT}/outputs/stage1_continuity_bank_v2_multiwindow_truebatch_fix0502/best.pt"}
MODEL_PATH=${MODEL_PATH:-"/data3/yeyuanhao/sp_re_cbp/thirdparty/models/Qwen2.5-VL-7B-Instruct"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker/models/VGGT-1B"}
TMP_ROOT=${TMP_ROOT:-"${PROJECT_ROOT}/tmp/stage1_v2_corr_cache"}

PYTHON_BIN=${PYTHON_BIN:-"python"}
MAX_GROUPS=${MAX_GROUPS:-"-1"}
CORR_GRAPH_METHOD=${CORR_GRAPH_METHOD:-"feature_knn"}
TEMPORAL_RADIUS=${TEMPORAL_RADIUS:-2}
TOPK_NEIGHBORS=${TOPK_NEIGHBORS:-8}
FEATURE_NORM=${FEATURE_NORM:-"True"}
D_GEOM=${D_GEOM:-1024}
FEATURE_LAYERS=${FEATURE_LAYERS:-"g11,g17,g23"}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_RANK=${SHARD_RANK:-0}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-1}

mkdir -p "${CORR_CACHE_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/qwen_vl/train/build_stage1_corr_graph_cache.py" \
  --base_manifest_path "${BASE_MANIFEST_PATH}" \
  --base_cache_dir "${BASE_CACHE_DIR}" \
  --output_manifest_path "${OUTPUT_MANIFEST_PATH}" \
  --corr_cache_dir "${CORR_CACHE_DIR}" \
  --projector_checkpoint_path "${PROJECTOR_CHECKPOINT_PATH}" \
  --model_name_or_path "${MODEL_PATH}" \
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
  --d_geom "${D_GEOM}" \
  --feature_layers "${FEATURE_LAYERS}" \
  --method "${CORR_GRAPH_METHOD}" \
  --temporal_radius "${TEMPORAL_RADIUS}" \
  --topk_neighbors "${TOPK_NEIGHBORS}" \
  --feature_norm "${FEATURE_NORM}" \
  --max_groups "${MAX_GROUPS}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_rank "${SHARD_RANK}" \
  --feature_batch_size "${FEATURE_BATCH_SIZE}"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
VARIANT_NAME=${VARIANT_NAME:-"vanilla_qwen25vl_7b_continuity_bank_v2"}
MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-VL-7B-Instruct"}
GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE:-"vggt"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"models/VGGT-1B"}
CACHE_WINDOW_MODE=${CACHE_WINDOW_MODE:-"fixed8"}
if [ -z "${CACHE_DIR:-}" ]; then
  if [ "${CACHE_WINDOW_MODE}" = "multi_window" ]; then
    CACHE_DIR="${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow"
  else
    CACHE_DIR="${PROJECT_ROOT}/cache/zenview_continuity_bank_v2"
  fi
fi
MANIFEST_PATH=${MANIFEST_PATH:-"${CACHE_DIR}/manifest.jsonl"}
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/stage1_continuity_bank_v2"}
DEFAULT_STAGE1_CHECKPOINT_PATH="${STAGE1_OUTPUT_DIR}/latest.pt"
if [ -f "${STAGE1_OUTPUT_DIR}/best.pt" ]; then
  DEFAULT_STAGE1_CHECKPOINT_PATH="${STAGE1_OUTPUT_DIR}/best.pt"
fi
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-"${DEFAULT_STAGE1_CHECKPOINT_PATH}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/${VARIANT_NAME}"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/${VARIANT_NAME}"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
GEO_IMPORTANCE_GATE=${GEO_IMPORTANCE_GATE:-"False"}
GEOMETRY_ENCODER_FREEZE=${GEOMETRY_ENCODER_FREEZE:-"True"}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-"False"}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-48}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6"}
TMP_ROOT=${TMP_ROOT:-"${PROJECT_ROOT}/tmp/stage2"}

mkdir -p "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

BASE_EXTRA_ARGS="--geo_inject_version zenview_continuity_bank_v2 --geometry_encoder_freeze True --vggt_bank_layers 11,17,23 --vggt_bank_d_geom 1024 --vggt_bank_topk 2 --vggt_bank_num_layers 20 --vggt_bank_fusion_layer_indices 0,2,4,6 --vggt_bank_use_layer_embedding False --use_continuity True --continuity_radius 1 --continuity_use_spatial_neighbors False --continuity_mlp_hidden_ratio 2.0 --continuity_attention_heads 4 --bank_gate_mode scalar --bank_debug False --cache_vggt_features False --stage1_checkpoint_path ${STAGE1_CHECKPOINT_PATH} --freeze_projector True --freeze_base_geometry_fusion True --freeze_continuity_builder True --freeze_geometry_decoder True --normalize_query True --normalize_bank True --bank_temperature 0.07 --candidate_dropout_enabled False --g11_drop_prob 0.0 --g17_drop_prob 0.0 --g23_drop_prob 0.0 --continuity_drop_prob 0.0 --geometry_cache_dir ${CACHE_DIR} --geometry_cache_manifest ${MANIFEST_PATH} --geometry_cache_use True --geometry_cache_required False"

if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
  EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS} ${EXTRA_TRAIN_ARGS}"
else
  EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS}"
fi

export PROJECT_ROOT VARIANT_NAME MODEL_PATH GEOMETRY_ENCODER_TYPE GEOMETRY_ENCODER_PATH
export GEOMETRY_ENCODER_FREEZE GEO_IMPORTANCE_GATE OUTPUT_DIR LOG_DIR TRAIN_LOG
export EXTRA_TRAIN_ARGS NPROC_PER_NODE TOTAL_BATCH_SIZE CUDA_VISIBLE_DEVICES GRADIENT_CHECKPOINTING
export GEO_INJECT_VERSION=zenview_continuity_bank_v2

if [[ "${VARIANT_NAME}" == *"fix0502"* || "${VARIANT_NAME}" == *"fix0503"* || -n "${STAGE2_APPROVAL_FLAG:-}" ]]; then
  STAGE2_APPROVAL_FLAG=${STAGE2_APPROVAL_FLAG:-"${PROJECT_ROOT}/logs/.allow_stage2_fix0502"}
  while [ ! -f "${STAGE2_APPROVAL_FLAG}" ]; do
    echo "[stage2-gate] waiting for approval flag: ${STAGE2_APPROVAL_FLAG}"
    sleep 120
  done
  echo "[stage2-gate] approval flag found, continuing Stage 2"
fi

exec bash "${SCRIPT_DIR}/train_vanilla_qwen25vl_variant.sh"

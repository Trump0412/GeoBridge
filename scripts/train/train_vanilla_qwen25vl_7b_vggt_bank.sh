#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
VARIANT_NAME=${VARIANT_NAME:-"vanilla_qwen25vl_7b_vggt_bank"}
MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-VL-7B-Instruct"}
GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE:-"vggt"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"models/VGGT-1B"}
GEO_INJECT_VERSION=${GEO_INJECT_VERSION:-"zenview_vggt_bank"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/${VARIANT_NAME}"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/${VARIANT_NAME}"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
GEO_IMPORTANCE_GATE=${GEO_IMPORTANCE_GATE:-"False"}
GEOMETRY_ENCODER_FREEZE=${GEOMETRY_ENCODER_FREEZE:-"False"}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-"False"}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-48}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6"}

BASE_EXTRA_ARGS="--geo_inject_version zenview_vggt_bank --geometry_encoder_freeze False --vggt_bank_layers 11,17,23 --vggt_bank_d_geom 1024 --vggt_bank_topk 2 --vggt_bank_num_layers 20 --use_continuity True --continuity_radius 1 --continuity_use_spatial_neighbors False --continuity_mlp_hidden_ratio 2.0 --continuity_attention_heads 4 --bank_gate_mode scalar --bank_debug False --cache_vggt_features False"

if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS} ${EXTRA_TRAIN_ARGS}"
else
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS}"
fi

export PROJECT_ROOT
export VARIANT_NAME
export MODEL_PATH
export GEOMETRY_ENCODER_TYPE
export GEOMETRY_ENCODER_PATH
export GEOMETRY_ENCODER_FREEZE
export GEO_INJECT_VERSION
export GEO_IMPORTANCE_GATE
export OUTPUT_DIR
export LOG_DIR
export TRAIN_LOG
export EXTRA_TRAIN_ARGS
export NPROC_PER_NODE
export TOTAL_BATCH_SIZE
export CUDA_VISIBLE_DEVICES
export GRADIENT_CHECKPOINTING

exec bash "${SCRIPT_DIR}/train_vanilla_qwen25vl_variant.sh"

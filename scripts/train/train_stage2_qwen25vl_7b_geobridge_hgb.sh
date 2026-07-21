#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
if [[ -f "${PROJECT_ROOT}/configs/geobridge_paths.env" ]]; then
  source "${PROJECT_ROOT}/configs/geobridge_paths.env"
fi
GEOBRIDGE_WORK_ROOT=${GEOBRIDGE_WORK_ROOT:-"${PROJECT_ROOT}/.local"}
VARIANT_NAME=${VARIANT_NAME:-"qwen25vl_7b_geobridge_hgb_pdegu_sparse4"}
MODEL_PATH=${MODEL_PATH:-"${GEOBRIDGE_STAGE2_MODEL_PATH:-${QWEN25VL_7B_PATH:-${GEOBRIDGE_WORK_ROOT}/models/Qwen/Qwen2.5-VL-7B-Instruct}}"}
GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE:-"vggt"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"${VGGT_1B_PATH:-${GEOBRIDGE_WORK_ROOT}/models/VGGT-1B}"}
CACHE_DIR=${CACHE_DIR:-"${STAGE1_WINDOW_READY_CACHE:-${PROJECT_ROOT}/cache/stage1_geobridge_window_ready_joint_fp16_g11}"}
MANIFEST_PATH=${MANIFEST_PATH:-"${CACHE_DIR}/manifest.jsonl"}
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/stage1_geobridge_fcp_g11_windowready_fp16_b8_resume_from1000"}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-"${STAGE1_FCP_CKPT:-}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/${VARIANT_NAME}"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/${VARIANT_NAME}"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
DATASETS=${DATASETS:-"llava_hound_64k,spar_234k,vsi_590k,vlm3r_vsi_205k,vlm3r_vst_132k,mindcube_10k,joyai_openspatial_100k"}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-48}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6"}
TMP_ROOT=${TMP_ROOT:-"${PROJECT_ROOT}/tmp/stage2_geobridge_hgb"}

mkdir -p "${TMP_ROOT}"
mkdir -p "${LOG_DIR}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

if [ -z "${STAGE1_CHECKPOINT_PATH}" ]; then
  echo "[SpatialFit-HGB] STAGE1_CHECKPOINT_PATH must be explicitly provided. Example:" >&2
  echo "  STAGE1_CHECKPOINT_PATH=${STAGE1_OUTPUT_DIR}/best.pt bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi
if [ ! -f "${STAGE1_CHECKPOINT_PATH}" ]; then
  echo "[SpatialFit-HGB] Stage1 checkpoint not found: ${STAGE1_CHECKPOINT_PATH}" >&2
  exit 1
fi
if [ ! -d "${CACHE_DIR}" ]; then
  echo "[SpatialFit-HGB] Geometry cache dir not found: ${CACHE_DIR}" >&2
  exit 1
fi
GEOMETRY_CACHE_USE=${GEOMETRY_CACHE_USE:-"True"}
GEOMETRY_CACHE_REQUIRED=${GEOMETRY_CACHE_REQUIRED:-"False"}
if [ "${GEOMETRY_CACHE_USE}" = "True" ] && [ ! -f "${MANIFEST_PATH}" ]; then
  echo "[SpatialFit-HGB] Geometry cache manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

export ZENVIEW_ROUTER_STATS_ENABLE=${ZENVIEW_ROUTER_STATS_ENABLE:-1}
export ZENVIEW_ROUTER_STATS_DIR=${ZENVIEW_ROUTER_STATS_DIR:-"${LOG_DIR}"}
export ZENVIEW_ROUTER_STATS_TAG=${ZENVIEW_ROUTER_STATS_TAG:-"${VARIANT_NAME}"}
export ZENVIEW_ROUTER_STATS_FLUSH_EVERY=${ZENVIEW_ROUTER_STATS_FLUSH_EVERY:-4}

VGGT_BANK_FUSION_LAYER_INDICES=${VGGT_BANK_FUSION_LAYER_INDICES:-"0,2,4,6"}
# Current SpatialFit sparse4 schedule.
BASE_EXTRA_ARGS="--geo_inject_version geobridge_hgb --geometry_encoder_freeze True --vggt_bank_layers 11,17,23 --vggt_bank_d_geom 1024 --vggt_bank_num_layers 8 --vggt_bank_fusion_layer_indices ${VGGT_BANK_FUSION_LAYER_INDICES} --vggt_bank_use_layer_embedding False --use_continuity True --continuity_radius 2 --continuity_use_spatial_neighbors False --continuity_mlp_hidden_ratio 2.0 --continuity_attention_heads 4 --bank_gate_mode scalar --bank_debug False --cache_vggt_features False --stage1_checkpoint_path ${STAGE1_CHECKPOINT_PATH} --freeze_projector True --freeze_base_geometry_fusion True --freeze_continuity_builder True --freeze_geometry_decoder True --freeze_continuity_selector True --freeze_activated_corr_graph True --normalize_query True --normalize_bank True --bank_temperature 0.07 --hgb_use_saliency_prior True --hgb_local_topk 2 --hgb_corr_topk_neighbors 8 --hgb_temporal_radius 2 --hgb_layer_scale_init 0.05 --hgb_gate_none_bias 0.0 --hgb_gate_local_bias 0.4 --hgb_gate_cont_bias 0.6 --hgb_use_gate_bias_init True --hgb_layer0_g11_logit_bias 2.0 --hgb_strict_alignment True --hgb_allow_layout_fallback False --hgb_alignment_audit_only False --hgb_min_overlap_ratio 1.0 --geometry_cache_dir ${CACHE_DIR} --geometry_cache_manifest ${MANIFEST_PATH} --geometry_cache_use ${GEOMETRY_CACHE_USE} --geometry_cache_required ${GEOMETRY_CACHE_REQUIRED}"

if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
  EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS} ${EXTRA_TRAIN_ARGS}"
else
  EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS}"
fi

export PROJECT_ROOT VARIANT_NAME MODEL_PATH GEOMETRY_ENCODER_TYPE GEOMETRY_ENCODER_PATH
export OUTPUT_DIR LOG_DIR TRAIN_LOG EXTRA_TRAIN_ARGS DATASETS NPROC_PER_NODE TOTAL_BATCH_SIZE CUDA_VISIBLE_DEVICES
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-"False"}
export GEO_INJECT_VERSION=geobridge_hgb
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

exec bash "${SCRIPT_DIR}/train_vanilla_qwen25vl_variant.sh"

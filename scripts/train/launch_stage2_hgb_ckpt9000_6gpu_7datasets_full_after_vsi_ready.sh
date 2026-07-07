#!/bin/bash
set -euo pipefail

# Full 7-dataset Stage2-HGB launcher. Run only after
# scripts/data/validate_vsi590k_paths.py reports missing_image_refs=0.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
cd "${ROOT}"
if [[ -f "${ROOT}/configs/geobridge_paths.env" ]]; then
  source "${ROOT}/configs/geobridge_paths.env"
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
export TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE))}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}

SAMPLE_COUNT=${SAMPLE_COUNT:-1336718}
MAX_STEPS=${MAX_STEPS:-$(((SAMPLE_COUNT + TOTAL_BATCH_SIZE - 1) / TOTAL_BATCH_SIZE))}
BATCH_TAG="bs${PER_DEVICE_BATCH_SIZE}_acc$((TOTAL_BATCH_SIZE / (NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE)))"
VARIANT_NAME=${VARIANT_NAME:-stage2_hgb_ckpt9000_6gpu_7datasets_${BATCH_TAG}_layers0246}

export OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/outputs/${VARIANT_NAME}"}
export LOG_DIR=${LOG_DIR:-"${ROOT}/logs/${VARIANT_NAME}"}
export TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
export TMP_ROOT=${TMP_ROOT:-"/tmp/geobridge_stage2_tmp_7d_${BATCH_TAG}"}
export TMPDIR=${TMP_ROOT}
export TMP=${TMP_ROOT}
export TEMP=${TMP_ROOT}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TMP_ROOT}"

export DATASETS=${DATASETS:-${STAGE2_DATASETS:-llava_hound_64k,spar_234k,vsi_590k,vlm3r_vsi_205k,vlm3r_vst_132k,mindcube_10k,joyai_openspatial_100k}}
export STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-"${ROOT}/outputs/stage1_geobridge_fcp_g11_windowready_fp16_b8_resume_from1000/checkpoint-9000.pt"}
export CACHE_DIR=${CACHE_DIR:-"${ROOT}/cache/stage1_geobridge_window_ready_joint_fp16_g11"}
export MANIFEST_PATH=${MANIFEST_PATH:-"${CACHE_DIR}/manifest.jsonl"}
export GEOMETRY_CACHE_USE=${GEOMETRY_CACHE_USE:-True}
export GEOMETRY_CACHE_REQUIRED=${GEOMETRY_CACHE_REQUIRED:-False}
export VGGT_BANK_FUSION_LAYER_INDICES=${VGGT_BANK_FUSION_LAYER_INDICES:-0,2,4,6}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
HGB_LAYER_SCALE_INIT=${HGB_LAYER_SCALE_INIT:-0.05}
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-} --max_steps ${MAX_STEPS} --warmup_ratio ${WARMUP_RATIO} --hgb_layer_scale_init ${HGB_LAYER_SCALE_INIT}"

echo "[INFO] variant=${VARIANT_NAME}"
echo "[INFO] datasets=${DATASETS}"
echo "[INFO] layers=${VGGT_BANK_FUSION_LAYER_INDICES}"
echo "[INFO] total_batch=${TOTAL_BATCH_SIZE} per_device=${PER_DEVICE_BATCH_SIZE} max_steps=${MAX_STEPS}"
echo "[INFO] warmup_ratio=${WARMUP_RATIO} hgb_layer_scale_init=${HGB_LAYER_SCALE_INIT}"
echo "[INFO] output=${OUTPUT_DIR}"

bash scripts/train/train_stage2_qwen25vl_7b_geobridge_hgb.sh

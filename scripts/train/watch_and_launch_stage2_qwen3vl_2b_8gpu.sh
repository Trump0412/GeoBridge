#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
if [[ -f "${PROJECT_ROOT}/configs/geobridge_paths.env" ]]; then
  source "${PROJECT_ROOT}/configs/geobridge_paths.env"
fi

GEOBRIDGE_WORK_ROOT=${GEOBRIDGE_WORK_ROOT:-/mnt/guojh/lq/new}
GPU_LIST=${GPU_LIST:-"0,1,2,3,4,5,6,7"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"${GPU_LIST}"}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE))}
GPU_MEM_USED_MAX_MB=${GPU_MEM_USED_MAX_MB:-2048}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-600}

RUN_ID=${RUN_ID:-"qwen3vl2b_stage2_hgb_7data_b16_8gpu_$(date +%Y%m%d_%H%M%S)"}
OUTPUT_DIR=${OUTPUT_DIR:-"${GEOBRIDGE_WORK_ROOT}/checkpoints/GeoBridge/${RUN_ID}"}
LOG_DIR=${LOG_DIR:-"${GEOBRIDGE_WORK_ROOT}/logs/${RUN_ID}"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
SUPERVISOR_LOG=${SUPERVISOR_LOG:-"${LOG_DIR}/supervisor.log"}

MODEL_PATH=${MODEL_PATH:-"${QWEN3VL_2B_PATH:-${GEOBRIDGE_WORK_ROOT}/models/Qwen/Qwen3-VL-2B-Instruct}"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"${VGGT_1B_PATH:-${GEOBRIDGE_WORK_ROOT}/weights/base_models/VGGT-1B}"}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-"${STAGE1_FCP_CKPT:-${GEOBRIDGE_WORK_ROOT}/weights/stage1/checkpoint-9000.pt}"}
CACHE_DIR=${CACHE_DIR:-"${STAGE1_WINDOW_READY_CACHE:-${GEOBRIDGE_WORK_ROOT}/cache/GeoBridge/stage1_geobridge_window_ready_joint_fp16_g11}"}
MANIFEST_PATH=${MANIFEST_PATH:-"${CACHE_DIR}/manifest.jsonl"}
DATASETS=${DATASETS:-"${STAGE2_DATASETS:-llava_hound_64k,spar_234k,vsi_590k,vlm3r_vsi_205k,vlm3r_vst_132k,mindcube_10k,joyai_openspatial_100k}"}
LEGACY_GEOBRIDGE_DATA_ROOT=${LEGACY_GEOBRIDGE_DATA_ROOT:-"${GEOBRIDGE_WORK_ROOT}/code/spatial4nips/data"}

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${CACHE_DIR}"
exec > >(tee -a "${SUPERVISOR_LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

prepare_data_links() {
  mkdir -p "${PROJECT_ROOT}/data"
  for name in train media evaluation; do
    if [[ ! -e "${PROJECT_ROOT}/data/${name}" && -e "${LEGACY_GEOBRIDGE_DATA_ROOT}/${name}" ]]; then
      ln -s "${LEGACY_GEOBRIDGE_DATA_ROOT}/${name}" "${PROJECT_ROOT}/data/${name}"
      log "linked data/${name} -> ${LEGACY_GEOBRIDGE_DATA_ROOT}/${name}"
    fi
  done
}

preflight() {
  [[ -d "${MODEL_PATH}" ]] || { log "missing MODEL_PATH=${MODEL_PATH}"; exit 1; }
  [[ -d "${GEOMETRY_ENCODER_PATH}" ]] || { log "missing GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH}"; exit 1; }
  [[ -f "${STAGE1_CHECKPOINT_PATH}" ]] || { log "missing STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH}"; exit 1; }
  [[ -x "${PYTHON_BIN:-}" || -x "${GEOBRIDGE_WORK_ROOT}/conda/envs/geothinker/bin/python" ]] || \
    { log "missing geothinker python; set PYTHON_BIN"; exit 1; }

  local train_root="${PROJECT_ROOT}/data/train"
  local missing=0
  for file in \
    llava_hound_64k.json \
    spar_234k.json \
    vsi_590k.json \
    vlm3r_vsi_205k.json \
    vlm3r_vst_132k.json \
    mindcube_10k.json \
    joyai_openspatial_100k.jsonl; do
    if [[ ! -s "${train_root}/${file}" ]]; then
      log "missing annotation ${train_root}/${file}"
      missing=1
    fi
  done
  [[ "${missing}" -eq 0 ]] || exit 1

  if [[ -f "${MANIFEST_PATH}" ]]; then
    export GEOMETRY_CACHE_USE=${GEOMETRY_CACHE_USE:-True}
    export GEOMETRY_CACHE_REQUIRED=${GEOMETRY_CACHE_REQUIRED:-True}
    log "using geometry cache manifest ${MANIFEST_PATH}"
  else
    export GEOMETRY_CACHE_USE=${GEOMETRY_CACHE_USE:-False}
    export GEOMETRY_CACHE_REQUIRED=${GEOMETRY_CACHE_REQUIRED:-False}
    log "geometry cache manifest not found; starting with GEOMETRY_CACHE_USE=${GEOMETRY_CACHE_USE}"
  fi
}

gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

wait_for_gpus() {
  log "waiting for GPUs ${GPU_LIST}; threshold memory.used <= ${GPU_MEM_USED_MAX_MB} MiB"
  while true; do
    local busy=0
    IFS=',' read -r -a gpu_ids <<< "${GPU_LIST}"
    for gpu in "${gpu_ids[@]}"; do
      local used
      used=$(gpu_used_mb "${gpu}")
      if [[ "${used}" -gt "${GPU_MEM_USED_MAX_MB}" ]]; then
        busy=1
      fi
      log "gpu=${gpu} memory.used=${used}MiB"
    done
    if [[ "${busy}" -eq 0 ]]; then
      log "all requested GPUs are free enough; launching Stage2"
      break
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

launch_stage2() {
  export PROJECT_ROOT MODEL_PATH GEOMETRY_ENCODER_PATH STAGE1_CHECKPOINT_PATH
  export CACHE_DIR MANIFEST_PATH OUTPUT_DIR LOG_DIR TRAIN_LOG
  export DATASETS CUDA_VISIBLE_DEVICES NPROC_PER_NODE PER_DEVICE_BATCH_SIZE TOTAL_BATCH_SIZE
  export QWEN_ATTN_IMPLEMENTATION=${QWEN_ATTN_IMPLEMENTATION:-sdpa}
  export VGGT_BANK_FUSION_LAYER_INDICES=${VGGT_BANK_FUSION_LAYER_INDICES:-0,1,2}
  export VARIANT_NAME=${VARIANT_NAME:-"${RUN_ID}"}
  export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  export HF_HOME=${HF_HOME:-"${GEOBRIDGE_WORK_ROOT}/hf_home"}
  export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"${HF_HOME}/hub"}
  export TMPDIR=${TMPDIR:-"${PROJECT_ROOT}/tmp/${RUN_ID}"}
  export TMP=${TMPDIR}
  export TEMP=${TMPDIR}
  mkdir -p "${TMPDIR}"

  PYTHON_BIN=${PYTHON_BIN:-"${GEOBRIDGE_WORK_ROOT}/conda/envs/geothinker/bin/python"}
  TORCHRUN_BIN=${TORCHRUN_BIN:-"${GEOBRIDGE_WORK_ROOT}/conda/envs/geothinker/bin/torchrun"}
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
  export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
  export TORCHRUN_BIN

  log "launching with MODEL_PATH=${MODEL_PATH}"
  log "launching with GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH}"
  log "launching with STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH}"
  log "launching with DATASETS=${DATASETS}"
  log "launching with per_device=${PER_DEVICE_BATCH_SIZE} total_batch=${TOTAL_BATCH_SIZE} gpus=${CUDA_VISIBLE_DEVICES}"
  exec bash "${PROJECT_ROOT}/scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh"
}

prepare_data_links
preflight
wait_for_gpus
launch_stage2

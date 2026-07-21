#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-VL-7B-Instruct"}
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
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/stage1_continuity_bank_v2"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/stage1_continuity_bank_v2"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOGGING_STEPS=${LOGGING_STEPS:-50}
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.10}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MASKED_RATIO=${MASKED_RATIO:-0.20}
GEOMETRY_CACHE_REQUIRED=${GEOMETRY_CACHE_REQUIRED:-"True"}
ONLINE_FALLBACK=${ONLINE_FALLBACK:-"False"}
GROUP_WINDOWS_BY_SOURCE=${GROUP_WINDOWS_BY_SOURCE:-"True"}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_GROUPS=${MAX_GROUPS:-"-1"}
PYTHON_BIN=${PYTHON_BIN:-"python"}
TORCHRUN_BIN=${TORCHRUN_BIN:-""}
EVAL_ONLY=${EVAL_ONLY:-"False"}
EVAL_OUTPUT_PATH=${EVAL_OUTPUT_PATH:-"${LOG_DIR}/stage1_eval.json"}
EVAL_CHECKPOINT_PATH=${EVAL_CHECKPOINT_PATH:-""}
EVAL_MAX_GROUPS=${EVAL_MAX_GROUPS:-"-1"}
AUTO_RESUME=${AUTO_RESUME:-"True"}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-""}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-"False"}
TMP_ROOT=${TMP_ROOT:-"${PROJECT_ROOT}/tmp/stage1"}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TMP_ROOT}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

if [ -z "${TORCHRUN_BIN}" ]; then
    TORCHRUN_BIN=$(command -v torchrun || true)
fi
if [ -z "${TORCHRUN_BIN}" ]; then
    for candidate in \
        "torchrun" \
        "${HOME}/.conda/envs/geothinker/bin/torchrun"; do
        if [ -x "${candidate}" ]; then
            TORCHRUN_BIN="${candidate}"
            break
        fi
    done
fi
if [ -z "${TORCHRUN_BIN}" ]; then
    echo "[ERROR] torchrun not found"
    exit 1
fi

cd "${PROJECT_ROOT}"
if [ "${EVAL_ONLY}" != "True" ] && [ -z "${RESUME_CHECKPOINT_PATH}" ] && [ "${AUTO_RESUME}" = "True" ] && [ -f "${OUTPUT_DIR}/latest.pt" ]; then
  RESUME_CHECKPOINT_PATH="${OUTPUT_DIR}/latest.pt"
fi
if [ "${EVAL_ONLY}" = "True" ]; then
  exec "${PYTHON_BIN}" src/qwen_vl/train/train_stage1_continuity.py \
    --model_name_or_path "${MODEL_PATH}" \
    --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
    --geometry_cache_manifest "${MANIFEST_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --max_groups "${MAX_GROUPS}" \
    --masked_ratio "${MASKED_RATIO}" \
    --geometry_cache_required "${GEOMETRY_CACHE_REQUIRED}" \
    --online_fallback "${ONLINE_FALLBACK}" \
    --group_windows_by_source "${GROUP_WINDOWS_BY_SOURCE}" \
    --num_workers "${NUM_WORKERS}" \
    --eval_only "${EVAL_ONLY}" \
    --eval_output_path "${EVAL_OUTPUT_PATH}" \
    --eval_checkpoint_path "${EVAL_CHECKPOINT_PATH}" \
    --eval_max_groups "${EVAL_MAX_GROUPS}" \
    > "${TRAIN_LOG}" 2>&1
fi

exec "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/qwen_vl/train/train_stage1_continuity.py \
  --model_name_or_path "${MODEL_PATH}" \
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
  --geometry_cache_manifest "${MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --max_groups "${MAX_GROUPS}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --min_lr_ratio "${MIN_LR_RATIO}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --masked_ratio "${MASKED_RATIO}" \
  --geometry_cache_required "${GEOMETRY_CACHE_REQUIRED}" \
  --online_fallback "${ONLINE_FALLBACK}" \
  --group_windows_by_source "${GROUP_WINDOWS_BY_SOURCE}" \
  --num_workers "${NUM_WORKERS}" \
  --resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}" \
  --resume_model_only "${RESUME_MODEL_ONLY}" \
  > "${TRAIN_LOG}" 2>&1

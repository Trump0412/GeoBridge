#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/smoke_vanilla_qwen25vl_7b_continuity_bank_v2"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/smoke_vanilla_qwen25vl_7b_continuity_bank_v2"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
BASE_EXTRA_ARGS="--max_steps 1 --save_steps 1 --logging_steps 1 --dataloader_num_workers 0 --bank_debug True"

if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
  EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS} ${EXTRA_TRAIN_ARGS}"
else
  EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS}"
fi

export OUTPUT_DIR LOG_DIR TRAIN_LOG EXTRA_TRAIN_ARGS
exec bash "${SCRIPT_DIR}/train_stage2_vanilla_qwen25vl_7b_continuity_bank_v2.sh"

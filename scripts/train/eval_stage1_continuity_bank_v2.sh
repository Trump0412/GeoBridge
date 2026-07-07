#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/stage1_continuity_bank_v2_eval"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/eval.log"}
EVAL_ONLY=True
EVAL_MAX_GROUPS=${EVAL_MAX_GROUPS:-256}

export LOG_DIR TRAIN_LOG EVAL_ONLY EVAL_MAX_GROUPS
exec bash "${SCRIPT_DIR}/train_stage1_continuity_bank_v2.sh"

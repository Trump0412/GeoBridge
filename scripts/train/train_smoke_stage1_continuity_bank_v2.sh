#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-"$(cd "${SCRIPT_DIR}/../.." && pwd)/outputs/smoke_stage1_continuity_bank_v2"}
LOG_DIR=${LOG_DIR:-"$(cd "${SCRIPT_DIR}/../.." && pwd)/logs/smoke_stage1_continuity_bank_v2"}
MAX_STEPS=${MAX_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1}
LOGGING_STEPS=${LOGGING_STEPS:-1}
MAX_GROUPS=${MAX_GROUPS:-8}
AUTO_RESUME=${AUTO_RESUME:-"False"}

export OUTPUT_DIR LOG_DIR MAX_STEPS SAVE_STEPS LOGGING_STEPS MAX_GROUPS AUTO_RESUME
exec bash "${SCRIPT_DIR}/train_stage1_continuity_bank_v2.sh"

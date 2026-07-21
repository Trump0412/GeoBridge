#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/train_stage2_qwen25vl_7b_geobridge_hgb.sh" "$@"

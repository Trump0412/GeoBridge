#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/train_smoke_stage1_geobridge_fcp_g11.sh" "$@"

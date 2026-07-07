#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MAX_STEPS=${MAX_STEPS:-3}
export VARIANT_NAME=${VARIANT_NAME:-stage2_hgb_ckpt9000_4gpu_7datasets_bs32_acc1_layers0246_smoke_20260601}

exec bash "${SCRIPT_DIR}/launch_stage2_hgb_ckpt9000_4gpu_7datasets_bs32_layers0246_after_vsi_ready.sh"

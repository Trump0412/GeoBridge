#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

export FEATURE_LAYERS=${FEATURE_LAYERS:-"g11"}
export JOINT_CACHE_DIR=${JOINT_CACHE_DIR:-"${PROJECT_ROOT:-.}/cache/stage1_geobridge_joint_compact_projected_int8_g11"}
export OUTPUT_MANIFEST_PATH=${OUTPUT_MANIFEST_PATH:-"${JOINT_CACHE_DIR}/manifest.jsonl"}

exec bash "${SCRIPT_DIR}/build_stage1_geobridge_joint_cache.sh"

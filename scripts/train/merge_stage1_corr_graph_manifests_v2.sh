#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

CORR_CACHE_DIR=${CORR_CACHE_DIR:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow_corrgraph_feature_knn"}
SHARD_MANIFEST_DIR=${SHARD_MANIFEST_DIR:-"${PROJECT_ROOT}/shards/stage1_v2_corr6"}
OUTPUT_MANIFEST_PATH=${OUTPUT_MANIFEST_PATH:-"${CORR_CACHE_DIR}/manifest.jsonl"}
COMPLETE_FLAG=${COMPLETE_FLAG:-"${CORR_CACHE_DIR}/.complete"}
PYTHON_BIN=${PYTHON_BIN:-"python3"}

mkdir -p "${CORR_CACHE_DIR}"

exec "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

corr_cache_dir = Path(os.environ["CORR_CACHE_DIR"])
shard_manifest_dir = Path(os.environ["SHARD_MANIFEST_DIR"])
output_manifest_path = Path(os.environ["OUTPUT_MANIFEST_PATH"])
complete_flag = Path(os.environ["COMPLETE_FLAG"])

paths = sorted(shard_manifest_dir.glob("manifest.s*.jsonl"))
if not paths:
    raise SystemExit(f"No shard manifests found under {shard_manifest_dir}")

seen = set()
rows = []
for path in paths:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            group_id = row["group_id"]
            if group_id in seen:
                continue
            seen.add(group_id)
            rows.append(row)

output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
with output_manifest_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

with complete_flag.open("w", encoding="utf-8") as handle:
    handle.write("done\n")

print(f"[corr-cache-merge] merged_rows={len(rows)}")
print(f"[corr-cache-merge] output_manifest_path={output_manifest_path}")
print(f"[corr-cache-merge] complete_flag={complete_flag}")
PY

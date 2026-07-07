#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
GEOBRIDGE_ROOT=${GEOBRIDGE_ROOT:-/mnt/guojh/lq/new}
MINDCUBE_SOURCE_DIR=${MINDCUBE_SOURCE_DIR:-"${GEOBRIDGE_ROOT}/datasets/MindCube"}
MINDCUBE_URL=${MINDCUBE_URL:-"https://hf-mirror.com/datasets/MLL-Lab/MindCube/resolve/main/data.zip"}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-"${REPO_ROOT}/tmp/mindcube_download"}
ZIP_PATH=${ZIP_PATH:-"${DOWNLOAD_DIR}/MindCube_data.zip"}
UNPACK_DIR=${UNPACK_DIR:-"${DOWNLOAD_DIR}/unpacked"}
MEDIA_DIR=${MEDIA_DIR:-"${REPO_ROOT}/data/media/MindCube"}
RAW_DIR=${RAW_DIR:-"${REPO_ROOT}/data/evaluation/MindCube/raw"}

mkdir -p "${DOWNLOAD_DIR}"

echo "[mindcube] repo_root=${REPO_ROOT}"
echo "[mindcube] source_dir=${MINDCUBE_SOURCE_DIR}"
echo "[mindcube] media_dir=${MEDIA_DIR}"

if [ -d "${MINDCUBE_SOURCE_DIR}/data/other_all_image" ] && [ -d "${MINDCUBE_SOURCE_DIR}/data/raw" ]; then
  echo "[mindcube] reuse existing source tree: ${MINDCUBE_SOURCE_DIR}"
  if [ ! -e "${MEDIA_DIR}" ]; then
    mkdir -p "$(dirname "${MEDIA_DIR}")"
    ln -s "${MINDCUBE_SOURCE_DIR}" "${MEDIA_DIR}"
  fi
  if [ ! -e "$(dirname "${RAW_DIR}")" ]; then
    mkdir -p "$(dirname "${RAW_DIR}")"
  fi
  echo "[mindcube] done"
  exit 0
fi

mkdir -p "${MEDIA_DIR}" "${RAW_DIR}"
if command -v aria2c >/dev/null 2>&1 && [ "$(readlink -f "$(command -v aria2c)")" != "/usr/bin/snap" ]; then
  aria2c -x 8 -s 8 -c --allow-overwrite=false -d "${DOWNLOAD_DIR}" -o "$(basename "${ZIP_PATH}")" "${MINDCUBE_URL}"
else
  wget -c --tries=20 --timeout=120 -O "${ZIP_PATH}" "${MINDCUBE_URL}"
fi

python3 - <<'PY' "${ZIP_PATH}" "${UNPACK_DIR}"
import os
import sys
import zipfile
zip_path = sys.argv[1]
unpack_dir = sys.argv[2]
os.makedirs(unpack_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(unpack_dir)
print(f"[mindcube] extracted -> {unpack_dir}")
PY

if [ -d "${UNPACK_DIR}/data/other_all_image" ]; then
  mkdir -p "${MEDIA_DIR}/other_all_image"
  cp -an "${UNPACK_DIR}/data/other_all_image/." "${MEDIA_DIR}/other_all_image/"
fi

if [ -d "${UNPACK_DIR}/data/raw" ]; then
  cp -an "${UNPACK_DIR}/data/raw/." "${RAW_DIR}/"
fi

echo "[mindcube] done"

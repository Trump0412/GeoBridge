#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/yeyuanhao/sp_re_cbp/GeoThinker_zenview_vggt_bank

source ~/.bashrc >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" >/dev/null 2>&1
elif [ -x "${HOME}/.conda/bin/conda" ]; then
  eval "$("${HOME}/.conda/bin/conda" shell.bash hook)" >/dev/null 2>&1
fi
conda activate geothinker >/dev/null 2>&1 || exit 127

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python scripts/analysis/visualize_stage1_vggt_similarity_heatmaps.py \
  --project_root /data3/yeyuanhao/sp_re_cbp/GeoThinker_zenview_vggt_bank \
  --manifest eval_inputs/revsi_32_selection_dev_window8_stride8_v1/manifest.jsonl \
  --sample_index 264 \
  --query_frame_index 4 \
  --target_frame_indices 0,1,2,3,4,5,6,7 \
  --query_coord 7,9 \
  --query_mode coord \
  --stage1_checkpoint outputs/stage1_geobridge_fcp_g11_windowready_fp16_b8_resume_from1000/checkpoint-6000.pt \
  --vggt_model_path /data3/yeyuanhao/sp_re_cbp/GeoThinker/models/VGGT-1B \
  --layers g5,g11,g17,g23,stage1_c \
  --cmap viridis \
  --norm percentile \
  --percentile_low 2 \
  --percentile_high 98 \
  --temperature 0.05 \
  --upsample bicubic \
  --alpha 0.55 \
  --topk 5 \
  --output_dir analysis/stage1_vggt_similarity_heatmaps/revsi_sample264_frame4_coord_7_9_ckpt6000

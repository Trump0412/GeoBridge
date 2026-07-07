#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
MODEL_PATH=${MODEL_PATH:-"/data3/yeyuanhao/sp_re_cbp/thirdparty/models/Qwen2.5-VL-7B-Instruct"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker/models/VGGT-1B"}

CORR_CACHE_DIR=${CORR_CACHE_DIR:-"${PROJECT_ROOT}/cache/zenview_continuity_bank_v2_multiwindow_corrgraph_feature_knn"}
MANIFEST_PATH=${MANIFEST_PATH:-"${CORR_CACHE_DIR}/manifest.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/stage1_continuity_v2_corrgraph_feature_knn"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/stage1_continuity_v2_corrgraph_feature_knn"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
TMP_ROOT=${TMP_ROOT:-"${PROJECT_ROOT}/tmp/stage1_v2"}

NPROC_PER_NODE=${NPROC_PER_NODE:-6}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 30000-39999 -n 1)}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOGGING_STEPS=${LOGGING_STEPS:-50}
LR=${LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.10}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MASKED_RATIO=${MASKED_RATIO:-0.20}

INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:-"${PROJECT_ROOT}/outputs/stage1_continuity_bank_v2_multiwindow_truebatch_fix0502/best.pt"}
AUTO_RESUME=${AUTO_RESUME:-"True"}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-""}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-"False"}

CONTINUITY_MODE=${CONTINUITY_MODE:-"corr_graph"}
FEATURE_LAYERS=${FEATURE_LAYERS:-"g11,g17,g23"}
CONTINUITY_RADIUS=${CONTINUITY_RADIUS:-2}
CONTINUITY_USE_SPATIAL_NEIGHBORS=${CONTINUITY_USE_SPATIAL_NEIGHBORS:-"False"}
CONTINUITY_MLP_HIDDEN_RATIO=${CONTINUITY_MLP_HIDDEN_RATIO:-2.0}
CONTINUITY_ATTENTION_HEADS=${CONTINUITY_ATTENTION_HEADS:-4}
CORR_SCORE_BETA=${CORR_SCORE_BETA:-1.0}
TIME_BIAS_INIT=${TIME_BIAS_INIT:--0.10}
USE_CONTINUITY_SELECTOR=${USE_CONTINUITY_SELECTOR:-"False"}
USE_ACTIVATED_CORR_GRAPH=${USE_ACTIVATED_CORR_GRAPH:-"False"}
CUS_LOSS_WEIGHT=${CUS_LOSS_WEIGHT:-0.0}
CUS_BUDGET_RATIO=${CUS_BUDGET_RATIO:-0.20}
CUS_BUDGET_WEIGHT=${CUS_BUDGET_WEIGHT:-0.10}
CUS_TARGET_TEMPERATURE=${CUS_TARGET_TEMPERATURE:-0.10}

MGC_COSINE_WEIGHT=${MGC_COSINE_WEIGHT:-1.0}
MGC_L1_WEIGHT=${MGC_L1_WEIGHT:-0.2}
CORR_NCE_WEIGHT=${CORR_NCE_WEIGHT:-0.5}
ATTN_WEIGHT=${ATTN_WEIGHT:-0.05}
LOV_GLOBAL_WEIGHT=${LOV_GLOBAL_WEIGHT:-0.2}
VAR_WEIGHT=${VAR_WEIGHT:-0.01}
VARIANCE_GAMMA=${VARIANCE_GAMMA:-1.0}
TEMPERATURE=${TEMPERATURE:-0.07}
NUM_NEGATIVES=${NUM_NEGATIVES:-64}
POSITIVE_TOPK=${POSITIVE_TOPK:-3}
MAX_CONTRASTIVE_ANCHORS=${MAX_CONTRASTIVE_ANCHORS:-256}

RANDOM_PATCH_PROB=${RANDOM_PATCH_PROB:-0.20}
CORR_TUBE_PROB=${CORR_TUBE_PROB:-0.50}
FRAME_BLOCK_PROB=${FRAME_BLOCK_PROB:-0.30}
RANDOM_PATCH_PROB_PHASE3=${RANDOM_PATCH_PROB_PHASE3:-0.10}
CORR_TUBE_PROB_PHASE3=${CORR_TUBE_PROB_PHASE3:-0.55}
FRAME_BLOCK_PROB_PHASE3=${FRAME_BLOCK_PROB_PHASE3:-0.35}
PHASE0_STEPS=${PHASE0_STEPS:-1000}
PHASE1_STEPS=${PHASE1_STEPS:-5000}
PHASE3_START_STEP=${PHASE3_START_STEP:-15000}

GEOMETRY_CACHE_REQUIRED=${GEOMETRY_CACHE_REQUIRED:-"False"}
CORR_CACHE_REQUIRED=${CORR_CACHE_REQUIRED:-"True"}
ONLINE_FALLBACK=${ONLINE_FALLBACK:-"True"}
FREEZE_GEO_PROJECTOR=${FREEZE_GEO_PROJECTOR:-"False"}
GROUP_WINDOWS_BY_SOURCE=${GROUP_WINDOWS_BY_SOURCE:-"True"}
NUM_WORKERS=${NUM_WORKERS:-4}
MEMORY_CACHE_SIZE=${MEMORY_CACHE_SIZE:-8}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-"True"}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
MAX_GROUPS=${MAX_GROUPS:-"-1"}

PYTHON_BIN=${PYTHON_BIN:-"python"}
TORCHRUN_BIN=${TORCHRUN_BIN:-""}
EVAL_ONLY=${EVAL_ONLY:-"False"}
EVAL_OUTPUT_PATH=${EVAL_OUTPUT_PATH:-"${LOG_DIR}/stage1_v2_eval.json"}
EVAL_CHECKPOINT_PATH=${EVAL_CHECKPOINT_PATH:-""}
EVAL_MAX_GROUPS=${EVAL_MAX_GROUPS:-"-1"}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TMP_ROOT}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

if [ -z "${TORCHRUN_BIN}" ]; then
  TORCHRUN_BIN=$(command -v torchrun || true)
fi
if [ -z "${TORCHRUN_BIN}" ]; then
  for candidate in \
    "/data3/yeyuanhao/.conda/envs/geothinker/bin/torchrun" \
    "${HOME}/.conda/envs/geothinker/bin/torchrun"; do
    if [ -x "${candidate}" ]; then
      TORCHRUN_BIN="${candidate}"
      break
    fi
  done
fi
if [ -z "${TORCHRUN_BIN}" ]; then
  echo "[ERROR] torchrun not found"
  exit 1
fi

cd "${PROJECT_ROOT}"
if [ "${EVAL_ONLY}" != "True" ] && [ -z "${RESUME_CHECKPOINT_PATH}" ] && [ "${AUTO_RESUME}" = "True" ] && [ -f "${OUTPUT_DIR}/latest.pt" ]; then
  RESUME_CHECKPOINT_PATH="${OUTPUT_DIR}/latest.pt"
fi

if [ "${EVAL_ONLY}" = "True" ]; then
  exec "${PYTHON_BIN}" src/qwen_vl/train/train_stage1_continuity_v2.py \
    --model_name_or_path "${MODEL_PATH}" \
    --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
    --geometry_cache_manifest "${MANIFEST_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --max_groups "${MAX_GROUPS}" \
    --masked_ratio "${MASKED_RATIO}" \
    --feature_layers "${FEATURE_LAYERS}" \
    --positive_topk "${POSITIVE_TOPK}" \
    --continuity_mode "${CONTINUITY_MODE}" \
    --use_continuity_selector "${USE_CONTINUITY_SELECTOR}" \
    --use_activated_corr_graph "${USE_ACTIVATED_CORR_GRAPH}" \
    --geometry_cache_required "${GEOMETRY_CACHE_REQUIRED}" \
    --corr_cache_required "${CORR_CACHE_REQUIRED}" \
    --online_fallback "${ONLINE_FALLBACK}" \
    --freeze_geo_projector "${FREEZE_GEO_PROJECTOR}" \
    --group_windows_by_source "${GROUP_WINDOWS_BY_SOURCE}" \
    --num_workers "${NUM_WORKERS}" \
    --memory_cache_size "${MEMORY_CACHE_SIZE}" \
    --persistent_workers "${PERSISTENT_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    --eval_only "${EVAL_ONLY}" \
    --eval_output_path "${EVAL_OUTPUT_PATH}" \
    --eval_checkpoint_path "${EVAL_CHECKPOINT_PATH}" \
    --eval_max_groups "${EVAL_MAX_GROUPS}" \
    > "${TRAIN_LOG}" 2>&1
fi

exec "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/qwen_vl/train/train_stage1_continuity_v2.py \
  --model_name_or_path "${MODEL_PATH}" \
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
  --geometry_cache_manifest "${MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --max_groups "${MAX_GROUPS}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --min_lr_ratio "${MIN_LR_RATIO}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --masked_ratio "${MASKED_RATIO}" \
  --feature_layers "${FEATURE_LAYERS}" \
  --continuity_mode "${CONTINUITY_MODE}" \
  --continuity_radius "${CONTINUITY_RADIUS}" \
  --continuity_use_spatial_neighbors "${CONTINUITY_USE_SPATIAL_NEIGHBORS}" \
  --continuity_mlp_hidden_ratio "${CONTINUITY_MLP_HIDDEN_RATIO}" \
  --continuity_attention_heads "${CONTINUITY_ATTENTION_HEADS}" \
  --corr_score_beta "${CORR_SCORE_BETA}" \
  --time_bias_init "${TIME_BIAS_INIT}" \
  --use_continuity_selector "${USE_CONTINUITY_SELECTOR}" \
  --use_activated_corr_graph "${USE_ACTIVATED_CORR_GRAPH}" \
  --cus_loss_weight "${CUS_LOSS_WEIGHT}" \
  --cus_budget_ratio "${CUS_BUDGET_RATIO}" \
  --cus_budget_weight "${CUS_BUDGET_WEIGHT}" \
  --cus_target_temperature "${CUS_TARGET_TEMPERATURE}" \
  --mgc_cosine_weight "${MGC_COSINE_WEIGHT}" \
  --mgc_l1_weight "${MGC_L1_WEIGHT}" \
  --corr_nce_weight "${CORR_NCE_WEIGHT}" \
  --attn_weight "${ATTN_WEIGHT}" \
  --lov_global_weight "${LOV_GLOBAL_WEIGHT}" \
  --var_weight "${VAR_WEIGHT}" \
  --variance_gamma "${VARIANCE_GAMMA}" \
  --temperature "${TEMPERATURE}" \
  --num_negatives "${NUM_NEGATIVES}" \
  --positive_topk "${POSITIVE_TOPK}" \
  --max_contrastive_anchors "${MAX_CONTRASTIVE_ANCHORS}" \
  --random_patch_prob "${RANDOM_PATCH_PROB}" \
  --corr_tube_prob "${CORR_TUBE_PROB}" \
  --frame_block_prob "${FRAME_BLOCK_PROB}" \
  --random_patch_prob_phase3 "${RANDOM_PATCH_PROB_PHASE3}" \
  --corr_tube_prob_phase3 "${CORR_TUBE_PROB_PHASE3}" \
  --frame_block_prob_phase3 "${FRAME_BLOCK_PROB_PHASE3}" \
  --phase0_steps "${PHASE0_STEPS}" \
  --phase1_steps "${PHASE1_STEPS}" \
  --phase3_start_step "${PHASE3_START_STEP}" \
  --geometry_cache_required "${GEOMETRY_CACHE_REQUIRED}" \
  --corr_cache_required "${CORR_CACHE_REQUIRED}" \
  --online_fallback "${ONLINE_FALLBACK}" \
  --freeze_geo_projector "${FREEZE_GEO_PROJECTOR}" \
  --group_windows_by_source "${GROUP_WINDOWS_BY_SOURCE}" \
  --num_workers "${NUM_WORKERS}" \
  --memory_cache_size "${MEMORY_CACHE_SIZE}" \
  --persistent_workers "${PERSISTENT_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}" \
  --resume_model_only "${RESUME_MODEL_ONLY}" \
  --init_checkpoint_path "${INIT_CHECKPOINT_PATH}" \
  > "${TRAIN_LOG}" 2>&1

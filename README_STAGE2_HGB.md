# Stage2 HGB Runbook

This document records the current GeoBridge Stage2 path after removing Stage3 RL and benchmark construction from this repository.

## Required Inputs

```text
MODEL_PATH
GEOMETRY_ENCODER_PATH
STAGE1_CHECKPOINT_PATH
CACHE_DIR
MANIFEST_PATH
DATASETS
```

Shared defaults:

```bash
source configs/geobridge_paths.env
```

## Qwen2.5 Baseline

The historical baseline uses Qwen2.5-VL-7B with HGB:

```bash
bash scripts/train/launch_stage2_hgb_ckpt9000_4gpu_7datasets_bs32_layers0246_after_vsi_ready.sh
```

Important settings:

```text
datasets: llava_hound_64k,spar_234k,vsi_590k,vlm3r_vsi_205k,vlm3r_vst_132k,mindcube_10k,joyai_openspatial_100k
fusion layers: 0,2,4,6
warmup ratio: 0.05
hgb layer scale init: 0.05
precision: bf16
deepspeed: scripts/zero2_opt.json
```

## Qwen3-VL-2B Adaptation

The Qwen3 entrypoint is:

```bash
bash scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh
```

Default model:

```text
/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct
```

The code path still shares the Qwen2.5-style GeoBridge training stack, but `train_qwen.py` dispatches to the Qwen3 wrapper when `MODEL_PATH` contains `qwen3`.

Qwen3 runs should start from Stage1/HGB initialization, not from Qwen2.5 Stage2 checkpoints.

## Smoke

```bash
MAX_STEPS=3 \
STAGE1_CHECKPOINT_PATH=/path/to/checkpoint-9000.pt \
CACHE_DIR=/path/to/stage1_cache \
bash scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh
```

## Full Run

```bash
PROJECT_ROOT=/mnt/guojh/lq/new/code/spatial4nips \
STAGE1_CHECKPOINT_PATH=/path/to/checkpoint-9000.pt \
CACHE_DIR=/path/to/stage1_cache \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
PER_DEVICE_BATCH_SIZE=8 \
TOTAL_BATCH_SIZE=48 \
bash scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh
```

The launcher writes outputs and logs under:

```text
outputs/${VARIANT_NAME}
logs/${VARIANT_NAME}
```

These paths are ignored by Git.

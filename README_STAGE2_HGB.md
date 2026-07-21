# Stage 2 Heterogeneous Geometry Bridging Runbook

This runbook describes the SpatialFit Stage 2 training path.

## Required Inputs

```text
MODEL_PATH
GEOMETRY_ENCODER_PATH
STAGE1_CHECKPOINT_PATH
CACHE_DIR
MANIFEST_PATH
DATASETS
```

The repository provides portable defaults in `configs/spatialfit_paths.env`, but
real experiments should set explicit local paths.

## Qwen2.5-VL Baseline

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
STAGE1_CHECKPOINT_PATH=/path/to/stage1/checkpoint.pt \
CACHE_DIR=/path/to/stage1/cache \
bash scripts/train/train_stage2_qwen25vl_7b_spatialfit_hgb.sh
```

Typical settings:

```text
datasets: llava_hound_64k,spar_234k,vsi_590k,vlm3r_vsi_205k,vlm3r_vst_132k,mindcube_10k,joyai_openspatial_100k
fusion layers: 0,2,4,6
warmup ratio: 0.05
precision: bf16
deepspeed: scripts/zero2_opt.json
```

## Qwen3-VL Adaptation

```bash
MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
STAGE1_CHECKPOINT_PATH=/path/to/stage1/checkpoint.pt \
CACHE_DIR=/path/to/stage1/cache \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
TOTAL_BATCH_SIZE=32 \
bash scripts/train/train_stage2_qwen3vl_2b_spatialfit_hgb.sh
```

Qwen3 runs should start from Stage 1 initialization rather than Qwen2.5 Stage 2
checkpoints.

## Smoke

```bash
MAX_STEPS=3 \
STAGE1_CHECKPOINT_PATH=/path/to/stage1/checkpoint.pt \
CACHE_DIR=/path/to/stage1/cache \
bash scripts/train/train_stage2_qwen3vl_2b_spatialfit_hgb.sh
```

Outputs and logs are written under ignored `outputs/` and `logs/` directories
unless overridden.

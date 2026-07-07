# GeoBridge

GeoBridge bridges discrete Qwen-VL visual tokens with continuous VGGT geometry for spatial reasoning.

This repository is the cleaned GeoBridge codebase for the paper line that now keeps only:

```text
Stage 1: FCP - Foreground-aware Correspondence Pretraining
Stage 2: HGB - Heterogeneous Geometry Bridging
```

The previous Stage3 RL/PSRO work has moved to the separate GeoPSRO repository. Benchmark construction and lmms-eval benchmark runners are intentionally not part of this repo.

## Repository Layout

```text
src/qwen_vl/model/geometry_bank/   FCP/HGB geometry bank modules
src/qwen_vl/model/stage2/          HGB gate and local-router components
src/qwen_vl/model/qwenvl3/         Qwen3-VL compatibility wrapper
src/qwen_vl/train/                 Stage1/Stage2 training and cache builders
scripts/train/                     Server launchers and data-prep helpers
configs/geobridge_paths.env        Shared server paths
tests/                             Lightweight Stage1/Stage2 unit tests
```

Generated data, model weights, checkpoints, caches, logs, and evaluation artifacts are ignored by Git.

## Server Defaults

Current shared server root:

```text
/mnt/guojh/lq/new
```

Important defaults are collected in `configs/geobridge_paths.env`:

```text
Qwen2.5-VL-7B: /mnt/guojh/lq/new/models/Qwen/Qwen2.5-VL-7B-Instruct
Qwen3-VL-2B:   /mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct
VGGT-1B:       /mnt/guojh/lq/new/models/VGGT-1B
HF mirror:     https://hf-mirror.com
```

## Main Commands

Stage1 FCP smoke:

```bash
bash scripts/train/train_smoke_stage1_geobridge_fcp_g11.sh
```

Stage2 HGB smoke with the Qwen2.5-compatible path:

```bash
bash scripts/train/launch_stage2_hgb_ckpt9000_4gpu_7datasets_bs32_layers0246_smoke.sh
```

Stage2 HGB using the Qwen3-VL-2B local model:

```bash
bash scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh
```

Use environment overrides for real runs:

```bash
PROJECT_ROOT=/mnt/guojh/lq/new/code/spatial4nips \
STAGE1_CHECKPOINT_PATH=/path/to/checkpoint-9000.pt \
CACHE_DIR=/path/to/stage1_cache \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
TOTAL_BATCH_SIZE=48 \
bash scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh
```

## Documents

- `GeoBridge_CORE_DESIGN.md`: method boundary and paper story.
- `GeoBridge_ENGINEERING_MANUAL.md`: server paths, training commands, and assets.
- `GeoBridge_PROGRESS.md`: current migration and implementation status.
- `README_STAGE2_HGB.md`: Stage2-HGB focused runbook.

## Current Boundary

GeoBridge remains a Qwen2.5-style codebase with Qwen3-VL compatibility added at the launcher/model-loading layer. The first Qwen3 target is:

```text
/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct
```

Full retraining on Qwen3 should be treated as a new Stage2 run because Qwen2.5 checkpoints are not weight-compatible with Qwen3-VL.

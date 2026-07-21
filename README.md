# SpatialFit

SpatialFit is a geometry-aware training codebase for spatial reasoning with
vision-language models. It connects discrete visual tokens with frozen geometry
features through two training stages:

```text
Stage 1: Foreground-aware Correspondence Pretraining
Stage 2: Heterogeneous Geometry Bridging
```

Generated datasets, model weights, checkpoints, caches, logs, and evaluation
artifacts are intentionally excluded from Git.

## Repository Layout

```text
src/qwen_vl/model/geometry_bank/   Stage 1 geometry-bank modules
src/qwen_vl/model/stage2/          Stage 2 gate and local-router components
src/qwen_vl/model/qwenvl3/         Qwen3-VL compatibility wrapper
src/qwen_vl/train/                 training and cache-building code
scripts/train/                     launchers and data-preparation helpers
configs/                           portable runtime defaults
tests/                             lightweight unit tests
```

Some internal Python symbols keep their historical names for backward
compatibility with checkpoints and launcher arguments. Public documentation and
package metadata use the SpatialFit name.

## Installation

```bash
python -m pip install -e .
python -m pytest -q
```

## Configure Paths

Set local paths before real training:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export SPATIALFIT_WORK_ROOT=/path/to/workdir
export QWEN25VL_7B_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export QWEN3VL_2B_PATH=/path/to/Qwen3-VL-2B-Instruct
export VGGT_1B_PATH=/path/to/VGGT-1B
export STAGE1_FCP_CKPT=/path/to/stage1/checkpoint.pt
export STAGE1_WINDOW_READY_CACHE=/path/to/stage1/cache
```

`configs/spatialfit_paths.env` provides portable defaults that can be overridden
by environment variables.

## Main Commands

Stage 1 smoke:

```bash
bash scripts/train/train_smoke_stage1_spatialfit_fcp_g11.sh
```

Stage 2 with Qwen3-VL:

```bash
MODEL_PATH="${QWEN3VL_2B_PATH}" \
STAGE1_CHECKPOINT_PATH="${STAGE1_FCP_CKPT}" \
CACHE_DIR="${STAGE1_WINDOW_READY_CACHE}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
TOTAL_BATCH_SIZE=32 \
bash scripts/train/train_stage2_qwen3vl_2b_spatialfit_hgb.sh
```

Stage 2 with Qwen2.5-VL:

```bash
MODEL_PATH="${QWEN25VL_7B_PATH}" \
STAGE1_CHECKPOINT_PATH="${STAGE1_FCP_CKPT}" \
CACHE_DIR="${STAGE1_WINDOW_READY_CACHE}" \
bash scripts/train/train_stage2_qwen25vl_7b_spatialfit_hgb.sh
```

## Notes

- Qwen3-VL runs use early decoder fusion layers by default.
- Qwen2.5-VL runs keep the sparse historical fusion-layer schedule.
- Full training should set all model, cache, dataset, and output paths
  explicitly.

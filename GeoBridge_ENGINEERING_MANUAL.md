# GeoBridge Engineering Manual

Last updated: 2026-07-07

## Local And Server Roots

Local working repo:

```text
/home/chenbp/GeoBridge
```

Shared server work root:

```text
/mnt/guojh/lq/new
```

The server code mirror currently used for training is expected at:

```text
/mnt/guojh/lq/new/code/spatial4nips
```

Shared path defaults live in:

```text
configs/geobridge_paths.env
```

Source it before manual runs, or let the launcher scripts source it automatically.

## Assets Not Stored In Git

The following are external assets and must stay outside Git:

```text
model weights
VGGT weights
training media
stage1 geometry cache
stage1 checkpoints
stage2 checkpoints
logs and generated outputs
```

Current Qwen3 target:

```text
/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct
```

## Stage1 FCP

Primary scripts:

```text
scripts/train/build_stage1_window_ready_joint_cache_g11.sh
scripts/train/train_stage1_geobridge_fcp_g11.sh
scripts/train/train_smoke_stage1_geobridge_fcp_g11.sh
```

Stage1 output consumed by Stage2:

```text
STAGE1_CHECKPOINT_PATH=/path/to/checkpoint-9000.pt
CACHE_DIR=/path/to/stage1_geobridge_window_ready_joint_fp16_g11
MANIFEST_PATH=${CACHE_DIR}/manifest.jsonl
```

## Stage2 HGB

Historical Qwen2.5-compatible launcher:

```bash
bash scripts/train/launch_stage2_hgb_ckpt9000_4gpu_7datasets_bs32_layers0246_after_vsi_ready.sh
```

Qwen3-VL-2B launcher:

```bash
bash scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh
```

The Qwen3 launcher sets:

```text
MODEL_PATH=/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct
QWEN_ATTN_IMPLEMENTATION=sdpa
```

Use `flash_attention_2` only after the environment has a compatible flash-attn build.

## Mirrors

Use Hugging Face mirror by default:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

The project should not rely on public HF endpoints for large dataset/model pulls unless explicitly requested.

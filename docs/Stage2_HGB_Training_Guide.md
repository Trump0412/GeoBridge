# Stage2 HGB Training Guide

GeoBridge Stage2 trains the Heterogeneous Geometry Bridge on top of a Qwen-VL backbone.

## Active Boundary

This guide covers only Stage2 HGB. Stage3 RL/PSRO and benchmark construction are out of scope for this repository.

## Inputs

Stage2 requires:

- a Qwen-VL model path,
- a VGGT model path,
- a Stage1 FCP checkpoint,
- a Stage1 geometry cache,
- a cache manifest,
- the Stage2 training data mixture.

The standard seven-source mixture is:

```text
llava_hound_64k
spar_234k
vsi_590k
vlm3r_vsi_205k
vlm3r_vst_132k
mindcube_10k
joyai_openspatial_100k
```

## Launchers

Qwen2.5-compatible path:

```bash
bash scripts/train/train_stage2_qwen25vl_7b_geobridge_hgb.sh
```

Qwen3-VL-2B path:

```bash
bash scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh
```

The Qwen3 wrapper only changes the model path and attention default. All HGB arguments are shared.

## Core HGB Arguments

```text
geo_inject_version=geobridge_hgb
vggt_bank_layers=11,17,23
vggt_bank_num_layers=8
vggt_bank_fusion_layer_indices=0,2,4,6
use_continuity=True
continuity_radius=2
hgb_local_topk=2
hgb_corr_topk_neighbors=8
hgb_layer_scale_init=0.05
hgb_strict_alignment=True
hgb_allow_layout_fallback=False
```

## Repro Notes

- `MANIFEST_PATH=${CACHE_DIR}/manifest.jsonl` must exist for cache-backed training.
- `GEOMETRY_CACHE_REQUIRED=True` is recommended for strict reproduction.
- `QWEN_ATTN_IMPLEMENTATION=sdpa` is the safe default for Qwen3 unless flash-attn is confirmed compatible.
- Large assets belong under `/mnt/guojh/lq/new`, not in this Git repository.

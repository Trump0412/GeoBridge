# GeoBridge Core Design

Last updated: 2026-07-07

## Scope

GeoBridge now covers two stages:

- `Stage1 FCP`: learns correspondence-aware continuity tokens from VGGT features.
- `Stage2 HGB`: injects local geometry and continuity geometry into Qwen-VL with a competitive bridge gate.

The old `Stage3 PSRO/RL` branch is no longer part of GeoBridge. It is maintained as GeoPSRO. Benchmark construction is also removed from this repository so GeoBridge stays focused on model training and geometry bridging.

## Claim

Spatial VLMs operate on discrete 2D visual tokens, while spatial reasoning often depends on continuous cross-frame geometry. GeoBridge bridges that mismatch by projecting VGGT local and continuity geometry back into Qwen-VL's visual-token space.

## Stage1: FCP

Stage1 trains a continuity token over VGGT features. The current stable engineering path is the `g11` window-ready cache:

- frozen VGGT feature extraction,
- correspondence graph / packed cache construction,
- continuity builder training,
- checkpoint consumed by Stage2 HGB.

The important artifact is the Stage1 checkpoint plus its geometry cache manifest. Stage2 should not silently fall back to online geometry if strict reproduction is desired.

## Stage2: HGB

Stage2 reads four geometry candidates per visual token:

```text
g11, g17, g23, continuity
```

The active HGB configuration is:

```text
fusion layers: 0,2,4,6
local top-k: 2
continuity radius: 2
hgb layer scale init: 0.05
strict alignment: true
```

The final historical Qwen2.5 run used seven data sources:

```text
llava_hound_64k
spar_234k
vsi_590k
vlm3r_vsi_205k
vlm3r_vst_132k
mindcube_10k
joyai_openspatial_100k
```

## Qwen3 Compatibility

The code keeps the Qwen2.5-VL implementation as the stable baseline. A Qwen3-VL wrapper and launcher are included for the next Stage2 run:

```text
/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct
```

This is a model-family migration, not a checkpoint resume. Use it for new Stage2 runs after Stage1 cache/checkpoint paths are verified.

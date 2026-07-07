# GeoBridge Progress

Last updated: 2026-07-07

## Completed In This Cleanup

- Created the local Git repository at `/home/chenbp/GeoBridge`.
- Imported the current server GeoBridge code from `spatial4nips`.
- Kept Stage1 FCP and Stage2 HGB code paths.
- Removed Stage3 RL/PSRO code and VERL configuration from GeoBridge.
- Removed lmms-eval benchmark code and benchmark runner scripts from the repository scope.
- Added `configs/geobridge_paths.env` for the `/mnt/guojh/lq/new` server layout.
- Added Qwen3-VL-2B Stage2 launcher:
  `scripts/train/train_stage2_qwen3vl_2b_geobridge_hgb.sh`.
- Updated package metadata from the inherited GeoThinker naming to GeoBridge.

## Current Data State

Stage2 seven-source media audit on the server has passed with `missing_refs=0` for:

```text
llava_hound_64k
spar_234k
vsi_590k
vlm3r_vsi_205k
vlm3r_vst_132k
mindcube_10k
joyai_openspatial_100k
```

Ready marker:

```text
/mnt/guojh/lq/new/tmp/stage2_7source_ready.ok
```

## Next Work

- Run a Qwen3-VL-2B Stage2 smoke once the target environment is selected.
- Confirm `VGGT_1B_PATH` on the server before full Qwen3 Stage2.
- Decide whether the new Qwen3 Stage2 run should reuse the historical seven-source recipe unchanged or downscale batch/sequence length for the 2B model first.

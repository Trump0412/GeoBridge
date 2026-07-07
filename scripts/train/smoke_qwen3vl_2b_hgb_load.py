#!/usr/bin/env python
"""Load-only smoke test for Qwen3-VL-2B GeoBridge HGB Stage2."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--geometry-encoder-path", required=True)
    parser.add_argument("--stage1-checkpoint-path", required=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--fusion-layers", default="0,1,2")
    return parser.parse_args()


def set_config_attr(config, name: str, value) -> None:
    setattr(config, name, value)
    if hasattr(config, "text_config"):
        setattr(config.text_config, name, value)


def load_stage1_checkpoint(model, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    load_targets = {
        "geo_projector": getattr(model.model, "geo_projector", None),
        "base_geometry_fusion": getattr(model.model, "base_geometry_fusion", None),
        "continuity_builder": getattr(model.model, "continuity_builder", None),
        "geometry_decoder": getattr(model.model, "geometry_decoder", None),
        "continuity_selector": getattr(model.model, "continuity_selector", None),
        "activated_corr_graph": getattr(model.model, "activated_corr_graph", None),
    }
    for prefix, module in load_targets.items():
        if module is None:
            print(f"stage1_prefix={prefix} status=missing_module")
            continue
        submodule_state = {
            key[len(prefix) + 1 :]: value
            for key, value in state_dict.items()
            if key.startswith(f"{prefix}.")
        }
        if not submodule_state:
            print(f"stage1_prefix={prefix} status=no_state")
            continue
        module_state = module.state_dict()
        compatible_state = {}
        skipped = []
        for key, value in submodule_state.items():
            target = module_state.get(key)
            if target is None or not torch.is_tensor(value) or value.shape == target.shape:
                compatible_state[key] = value
            elif value.numel() == target.numel():
                compatible_state[key] = value.reshape(target.shape)
            else:
                skipped.append((key, tuple(value.shape), tuple(target.shape)))
        if skipped:
            print(f"stage1_prefix={prefix} skipped_incompatible={skipped}")
        submodule_state = compatible_state
        if not submodule_state:
            print(f"stage1_prefix={prefix} status=no_compatible_state")
            continue
        missing, unexpected = module.load_state_dict(submodule_state, strict=False)
        print(
            f"stage1_prefix={prefix} loaded_keys={len(submodule_state)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))

    for label, path in [
        ("model", args.model_path),
        ("geometry_encoder", args.geometry_encoder_path),
        ("stage1_checkpoint", args.stage1_checkpoint_path),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} path not found: {path}")

    from qwen_vl.model.qwenvl3.configuration_qwen3_vl import Qwen3VLConfig
    from qwen_vl.model.qwenvl3.modeling_qwen3_vl import Qwen3VLForConditionalGenerationWithVGGT

    config = Qwen3VLConfig.from_pretrained(args.model_path)
    hgb_defaults = {
        "use_geometry_encoder": True,
        "geometry_encoder_type": "vggt",
        "geometry_encoder_path": args.geometry_encoder_path,
        "geometry_encoder_freeze": True,
        "reference_frame": "first",
        "feature_fusion_method": "zero",
        "geo_cross_attn": True,
        "geo_inject_version": "geobridge_hgb",
        "geo_importance_gate": True,
        "vggt_bank_layers": "11,17,23",
        "vggt_bank_d_geom": 1024,
        "vggt_bank_num_layers": 8,
        "vggt_bank_fusion_layer_indices": args.fusion_layers,
        "vggt_bank_use_layer_embedding": False,
        "use_continuity": True,
        "continuity_radius": 2,
        "continuity_use_spatial_neighbors": False,
        "continuity_mlp_hidden_ratio": 2.0,
        "continuity_attention_heads": 4,
        "bank_gate_mode": "scalar",
        "bank_debug": False,
        "cache_vggt_features": False,
        "stage1_checkpoint_path": args.stage1_checkpoint_path,
        "freeze_projector": True,
        "freeze_base_geometry_fusion": True,
        "freeze_continuity_builder": True,
        "freeze_geometry_decoder": True,
        "freeze_continuity_selector": True,
        "freeze_activated_corr_graph": True,
        "normalize_query": True,
        "normalize_bank": True,
        "bank_temperature": 0.07,
        "hgb_use_saliency_prior": True,
        "hgb_local_topk": 2,
        "hgb_corr_topk_neighbors": 8,
        "hgb_temporal_radius": 2,
        "hgb_layer_scale_init": 0.05,
        "hgb_gate_none_bias": 0.0,
        "hgb_gate_local_bias": 0.4,
        "hgb_gate_cont_bias": 0.6,
        "hgb_use_gate_bias_init": True,
        "hgb_layer0_g11_logit_bias": 2.0,
        "hgb_strict_alignment": True,
        "hgb_allow_layout_fallback": False,
        "hgb_alignment_audit_only": False,
        "hgb_min_overlap_ratio": 1.0,
        "depart_smi_token": True,
        "smi_image_num": 8,
        "smi_downsample_rate": 2,
    }
    for key, value in hgb_defaults.items():
        set_config_attr(config, key, value)
    config.text_config.vision_config = config.vision_config

    print(
        "qwen3_config "
        f"text_hidden={config.text_config.hidden_size} "
        f"text_layers={config.text_config.num_hidden_layers} "
        f"vision_depth={config.vision_config.depth} "
        f"deepstack={config.vision_config.deepstack_visual_indexes}"
    )
    model = Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        geometry_encoder_path=args.geometry_encoder_path,
    )
    load_stage1_checkpoint(model, args.stage1_checkpoint_path)
    active_layers = [
        idx
        for idx, layer in enumerate(model.model.language_model.layers)
        if bool(getattr(layer, "cross_flag", False))
    ]
    print(f"hgb_active_layers={active_layers}")
    if active_layers != [0, 1, 2]:
        raise RuntimeError(f"Unexpected Qwen3 HGB active layers: {active_layers}")
    print("SMOKE_OK qwen3vl_2b_hgb_load")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate multi-frame Stage1/VGGT similarity heatmaps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from stage1_vggt_similarity_common import (
    build_grid_image,
    cosine_similarity_map,
    delta_stem,
    describe_array,
    draw_patch_boxes,
    draw_query_box,
    ensure_unit_interval,
    extract_stage1_vggt_feature_bundle,
    git_commit_hash,
    layer_display_name,
    layer_file_stem,
    load_manifest_row,
    maybe_smooth_map,
    normalize_similarity_map,
    parse_int_csv,
    parse_layer_names,
    parse_query_coord,
    resolve_project_path,
    resolve_query_coord,
    save_heatmap_only,
    save_overlay,
    save_overlay_image,
    stage1_checkpoint_label,
    topk_patch_entries,
    validate_feature_alignment,
    validate_frame_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--query_frame_index", type=int, required=True)
    parser.add_argument("--target_frame_indices", type=str, required=True)
    parser.add_argument("--query_coord", type=str, default=None)
    parser.add_argument("--query_mode", type=str, default="center", choices=("coord", "center", "max_cus", "max_contrast"))
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--vggt_model_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--layers", type=str, default="g5,g11,g17,g23,stage1_c")
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--delta_cmap", type=str, default="viridis")
    parser.add_argument("--norm", type=str, default="percentile", choices=("raw", "minmax", "percentile", "softmax"))
    parser.add_argument("--percentile_low", type=float, default=2.0)
    parser.add_argument("--percentile_high", type=float, default=98.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--upsample", type=str, default="bicubic", choices=("nearest", "bilinear", "bicubic"))
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shared_norm_per_panel_group", action="store_true")
    parser.add_argument("--d_geom", type=int, default=1024)
    parser.add_argument("--temporal_radius", type=int, default=2)
    parser.add_argument("--topk_neighbors", type=int, default=8)
    parser.add_argument("--continuity_mlp_hidden_ratio", type=float, default=2.0)
    parser.add_argument("--continuity_attention_heads", type=int, default=4)
    parser.add_argument("--corr_score_beta", type=float, default=1.0)
    parser.add_argument("--time_bias_init", type=float, default=-0.10)
    return parser.parse_args()


def build_shared_reference(raw_maps: Sequence[object]) -> object | None:
    arrays = []
    for raw_map in raw_maps:
        if raw_map is None:
            continue
        arrays.append(raw_map)
    if not arrays:
        return None
    import numpy as np

    return np.concatenate([np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays], axis=0)


def normalize_for_visualization(
    raw_map,
    *,
    norm_mode: str,
    p_low: float,
    p_high: float,
    temperature: float,
    smooth_sigma: float,
    shared_reference=None,
):
    sim_vis, norm_info = normalize_similarity_map(
        raw_map,
        mode=norm_mode,
        p_low=p_low,
        p_high=p_high,
        temperature=temperature,
        reference_values=shared_reference,
    )
    if smooth_sigma > 0:
        sim_vis = maybe_smooth_map(sim_vis, smooth_sigma)
    sim_vis, clipped = ensure_unit_interval(sim_vis)
    norm_info["clipped_to_unit_interval"] = bool(clipped)
    norm_info["smoothed"] = bool(smooth_sigma > 0)
    norm_info["sim_vis_min"] = float(sim_vis.min())
    norm_info["sim_vis_max"] = float(sim_vis.max())
    return sim_vis, norm_info


def main() -> None:
    args = parse_args()
    project_root = resolve_project_path(Path.cwd(), args.project_root, required=True)
    manifest_path = resolve_project_path(project_root, args.manifest, required=True)
    stage1_checkpoint = resolve_project_path(project_root, args.stage1_checkpoint, required=True)
    vggt_model_path = resolve_project_path(project_root, args.vggt_model_path, required=True)
    output_dir = resolve_project_path(project_root, args.output_dir, required=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_layers = parse_layer_names(args.layers)
    target_frame_indices = parse_int_csv(args.target_frame_indices)
    query_coord_arg = parse_query_coord(args.query_coord)

    sample = load_manifest_row(manifest_path, args.sample_index)
    frame_paths = list(sample["frame_paths"])
    validate_frame_indices(len(frame_paths), args.query_frame_index, target_frame_indices)

    hidden_layer_names = set(requested_layers)
    hidden_layer_names.add("g11")
    hidden_layer_names.add("stage1_c")
    bundle = extract_stage1_vggt_feature_bundle(
        frame_paths=frame_paths,
        project_root=project_root,
        vggt_model_path=str(vggt_model_path),
        stage1_checkpoint=str(stage1_checkpoint),
        layer_names=sorted(hidden_layer_names),
        processor_path=args.model_name_or_path,
        device=args.device,
        d_geom=args.d_geom,
        temporal_radius=args.temporal_radius,
        topk_neighbors=args.topk_neighbors,
        continuity_mlp_hidden_ratio=args.continuity_mlp_hidden_ratio,
        continuity_attention_heads=args.continuity_attention_heads,
        corr_score_beta=args.corr_score_beta,
        time_bias_init=args.time_bias_init,
    )
    validate_feature_alignment(bundle)

    query_frame_shape = bundle.frame_shapes[args.query_frame_index]
    query_coord = resolve_query_coord(
        query_frame_shape,
        query_mode=args.query_mode,
        query_coord=query_coord_arg,
        reference_tokens=bundle.feature_maps["g11"][args.query_frame_index],
        selector_probs=None if bundle.selector_probs is None else bundle.selector_probs[args.query_frame_index],
        selector_stats=None if bundle.selector_stats is None else bundle.selector_stats[args.query_frame_index],
    )

    stage1_label = stage1_checkpoint_label(stage1_checkpoint)
    visible_layers = list(requested_layers)
    frame_key_to_index = {f"frame_{frame_idx:02d}": frame_idx for frame_idx in target_frame_indices}
    feature_shapes = {
        layer_name: {
            "tensor_shape": list(bundle.feature_maps[layer_name].shape),
            "frame_shapes": [list(shape) for shape in bundle.frame_shapes],
            "token_counts": list(bundle.token_counts),
        }
        for layer_name in sorted(bundle.feature_maps)
    }

    raw_similarity_maps: Dict[str, Dict[str, object]] = {}
    raw_similarity_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    topk_metadata: Dict[str, Dict[str, List[Dict[str, float | int]]]] = {}
    delta_raw_stats: Dict[str, Dict[str, float]] = {}
    normalization_info: Dict[str, Dict[str, object]] = {}

    for frame_idx in target_frame_indices:
        frame_key = f"frame_{frame_idx:02d}"
        raw_similarity_maps[frame_key] = {}
        raw_similarity_stats[frame_key] = {}
        topk_metadata[frame_key] = {}
        for layer_name in visible_layers:
            raw_map = cosine_similarity_map(
                bundle.feature_maps[layer_name][args.query_frame_index],
                bundle.feature_maps[layer_name][frame_idx],
                query_coord,
            )
            raw_similarity_maps[frame_key][layer_name] = raw_map
            raw_similarity_stats[frame_key][layer_name] = describe_array(raw_map)
            topk_metadata[frame_key][layer_name] = topk_patch_entries(raw_map, args.topk)
        delta_raw = (
            cosine_similarity_map(
                bundle.feature_maps["stage1_c"][args.query_frame_index],
                bundle.feature_maps["stage1_c"][frame_idx],
                query_coord,
            )
            - cosine_similarity_map(
                bundle.feature_maps["g11"][args.query_frame_index],
                bundle.feature_maps["g11"][frame_idx],
                query_coord,
            )
        )
        raw_similarity_maps[frame_key][delta_stem()] = delta_raw
        delta_raw_stats[frame_key] = describe_array(delta_raw)
        raw_similarity_stats[frame_key][delta_stem()] = delta_raw_stats[frame_key]
        topk_metadata[frame_key][delta_stem()] = topk_patch_entries(delta_raw, args.topk)

    shared_norm_reference: Dict[str, object | None] = {}
    if args.shared_norm_per_panel_group:
        for layer_name in visible_layers + [delta_stem()]:
            shared_norm_reference[layer_name] = build_shared_reference(
                [raw_similarity_maps[f"frame_{frame_idx:02d}"][layer_name] for frame_idx in target_frame_indices]
            )
    else:
        for layer_name in visible_layers + [delta_stem()]:
            shared_norm_reference[layer_name] = None

    overview_rows = []
    expected_files = []
    for frame_idx in target_frame_indices:
        frame_key = f"frame_{frame_idx:02d}"
        heatmap_dir = output_dir / "heatmaps" / frame_key
        overlay_dir = output_dir / "overlays" / frame_key
        topk_dir = output_dir / "topk"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        topk_dir.mkdir(parents=True, exist_ok=True)

        rgb_frame = bundle.rgb_frames[frame_idx]
        rgb_panel = rgb_frame if frame_idx != args.query_frame_index else draw_query_box(rgb_frame, bundle.frame_shapes[frame_idx], query_coord)
        row_panels = [(f"Frame {frame_idx} / RGB", rgb_panel)]
        normalization_info[frame_key] = {}

        for layer_name in visible_layers:
            file_stem = layer_file_stem(layer_name)
            raw_map = raw_similarity_maps[frame_key][layer_name]
            sim_vis, norm_info = normalize_for_visualization(
                raw_map,
                norm_mode=args.norm,
                p_low=args.percentile_low,
                p_high=args.percentile_high,
                temperature=args.temperature,
                smooth_sigma=args.smooth_sigma,
                shared_reference=shared_norm_reference[layer_name],
            )
            normalization_info[frame_key][layer_name] = norm_info
            heatmap_path = heatmap_dir / f"{file_stem}_heatmap.png"
            overlay_path = overlay_dir / f"{file_stem}_overlay.png"
            save_heatmap_only(
                sim_vis,
                heatmap_path,
                cmap=args.cmap,
                output_size=rgb_frame.size,
                upsample=args.upsample,
            )
            save_overlay(
                rgb_frame,
                sim_vis,
                overlay_path,
                cmap=args.cmap,
                alpha=args.alpha,
                upsample=args.upsample,
                topk_entries=topk_metadata[frame_key][layer_name],
                grid_shape=bundle.frame_shapes[frame_idx],
            )
            expected_files.extend([heatmap_path, overlay_path])
            panel_image = save_overlay_image(
                rgb_frame,
                sim_vis,
                cmap=args.cmap,
                alpha=args.alpha,
                upsample=args.upsample,
                topk_entries=topk_metadata[frame_key][layer_name][:1],
                grid_shape=bundle.frame_shapes[frame_idx],
            )
            row_panels.append((f"Frame {frame_idx} / {layer_name}", panel_image))

        delta_vis, delta_norm_info = normalize_for_visualization(
            raw_similarity_maps[frame_key][delta_stem()],
            norm_mode="percentile",
            p_low=args.percentile_low,
            p_high=args.percentile_high,
            temperature=args.temperature,
            smooth_sigma=args.smooth_sigma,
            shared_reference=shared_norm_reference[delta_stem()],
        )
        normalization_info[frame_key][delta_stem()] = delta_norm_info
        delta_heatmap_path = heatmap_dir / f"{delta_stem()}_heatmap.png"
        delta_overlay_path = overlay_dir / f"{delta_stem()}_overlay.png"
        save_heatmap_only(
            delta_vis,
            delta_heatmap_path,
            cmap=args.delta_cmap,
            output_size=rgb_frame.size,
            upsample=args.upsample,
        )
        save_overlay(
            rgb_frame,
            delta_vis,
            delta_overlay_path,
            cmap=args.delta_cmap,
            alpha=args.alpha,
            upsample=args.upsample,
            topk_entries=topk_metadata[frame_key][delta_stem()],
            grid_shape=bundle.frame_shapes[frame_idx],
        )
        expected_files.extend([delta_heatmap_path, delta_overlay_path])
        row_panels.append(
            (
                f"Frame {frame_idx} / delta",
                save_overlay_image(
                    rgb_frame,
                    delta_vis,
                    cmap=args.delta_cmap,
                    alpha=args.alpha,
                    upsample=args.upsample,
                    topk_entries=topk_metadata[frame_key][delta_stem()][:1],
                    grid_shape=bundle.frame_shapes[frame_idx],
                ),
            )
        )
        (topk_dir / f"{frame_key}_topk.json").write_text(
            json.dumps(
                {
                    "frame_index": frame_idx,
                    "grid_shape": list(bundle.frame_shapes[frame_idx]),
                    "query_frame_index": args.query_frame_index,
                    "query_coord": list(query_coord),
                    "layers": topk_metadata[frame_key],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        expected_files.append(topk_dir / f"{frame_key}_topk.json")
        overview_rows.append(row_panels)

    grid_overview_path = output_dir / "grid_overview.png"
    build_grid_image(overview_rows).save(grid_overview_path)
    expected_files.append(grid_overview_path)

    for file_path in expected_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Expected output file is missing: {file_path}")
    for frame_idx in target_frame_indices:
        frame_key = f"frame_{frame_idx:02d}"
        for layer_name, norm_info in normalization_info[frame_key].items():
            if not (0.0 <= float(norm_info["sim_vis_min"]) <= 1.0 and 0.0 <= float(norm_info["sim_vis_max"]) <= 1.0):
                raise ValueError(f"{frame_key}/{layer_name} sim_vis is outside [0, 1]: {norm_info}")
        for layer_name, stats in raw_similarity_stats[frame_key].items():
            if any(value != value or value in (float("inf"), float("-inf")) for value in stats.values()):
                raise ValueError(f"{frame_key}/{layer_name} has non-finite raw similarity stats: {stats}")

    metadata = {
        "sample_index": args.sample_index,
        "source_dataset": sample.get("source_dataset"),
        "source_sample_id": sample.get("source_sample_id"),
        "group_id": sample.get("group_id"),
        "query_frame_index": args.query_frame_index,
        "target_frame_indices": target_frame_indices,
        "query_coord": list(query_coord),
        "query_mode": args.query_mode,
        "stage1_checkpoint": str(stage1_checkpoint),
        "vggt_model_path": str(vggt_model_path),
        "processor_path": bundle.processor_path,
        "layers": visible_layers,
        "feature_shapes": feature_shapes,
        "patch_grid": list(bundle.patch_grid),
        "merged_grid": list(bundle.merged_grid),
        "frame_paths": bundle.frame_paths,
        "sampled_frame_indices": sample.get("sampled_frame_indices"),
        "normalization": {
            "mode": args.norm,
            "percentile_low": args.percentile_low,
            "percentile_high": args.percentile_high,
            "temperature": args.temperature,
            "smooth_sigma": args.smooth_sigma,
            "shared_norm_per_panel_group": bool(args.shared_norm_per_panel_group),
            "per_frame_layer": normalization_info,
        },
        "raw_similarity_stats": raw_similarity_stats,
        "delta_raw_stats": delta_raw_stats,
        "topk": topk_metadata,
        "git_commit_hash": git_commit_hash(project_root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

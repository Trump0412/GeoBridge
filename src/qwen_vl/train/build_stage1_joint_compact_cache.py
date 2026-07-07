"""Build source-level compact joint cache for Stage 1 v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qwen_vl.data.utils import prepare_image_inputs
from qwen_vl.model.geometry_bank import GeoProjector, VGGTBankExtractor
from qwen_vl.train.stage1_compact_cache import (
    PROJECTED_SOURCE_CACHE_FORMAT,
    quantize_projected_tokenwise,
)


def parse_feature_layers(value: str) -> List[str]:
    layers = [item.strip() for item in str(value).split(",") if item.strip()]
    if not layers:
        raise ValueError("feature_layers must not be empty")
    valid = {"g11", "g17", "g23"}
    invalid = [layer for layer in layers if layer not in valid]
    if invalid:
        raise ValueError(f"Unsupported feature_layers: {invalid}")
    return layers


def raw_feature_name(layer_name: str) -> str:
    return f"{layer_name}_raw"


def resolve_frame_path(frame_path: str) -> str:
    if os.path.isabs(frame_path):
        return frame_path
    candidate = os.path.join(str(project_root), frame_path)
    if os.path.exists(candidate):
        return candidate
    return frame_path


def load_geometry_inputs(frame_paths: Sequence[str], image_processor) -> torch.Tensor:
    from PIL import Image

    resolved_paths = [resolve_frame_path(path) for path in frame_paths]
    with Image.open(resolved_paths[0]) as first_image:
        base_size = first_image.size
    prepared_inputs = []
    for frame_path in resolved_paths:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            if rgb.size != base_size:
                rgb = rgb.resize(base_size, Image.BILINEAR)
            prepared_inputs.append(prepare_image_inputs(rgb, image_processor))
    return torch.stack([prepared["geometry_encoder_inputs"] for prepared in prepared_inputs])


def load_projector(checkpoint_path: str, d_geom: int, device: torch.device, feature_layers: Sequence[str]) -> GeoProjector:
    projector = GeoProjector(
        input_dims={f"{name}_raw": 8192 for name in feature_layers},
        d_geom=d_geom,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = payload.get("model", payload)
    own_state = projector.state_dict()
    filtered_state = {
        key.removeprefix("geo_projector."): value.to(device)
        for key, value in state_dict.items()
        if key.startswith("geo_projector.")
        and key.removeprefix("geo_projector.") in own_state
        and own_state[key.removeprefix("geo_projector.")].shape == value.shape
    }
    projector.load_state_dict(filtered_state, strict=False)
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad = False
    return projector


def source_cache_path(cache_dir: str, source_sample_id: str) -> str:
    prefix = source_sample_id[:2] if source_sample_id else "xx"
    return os.path.join(cache_dir, "source_features", prefix, f"{source_sample_id}.pt")


def group_manifest_records(records: Sequence[Dict]) -> List[Tuple[str, List[Dict]]]:
    grouped = defaultdict(list)
    for record in records:
        source_sample_id = record.get("source_sample_id", record["group_id"])
        grouped[source_sample_id].append(record)
    return sorted(grouped.items(), key=lambda item: item[0])


def collect_source_frames(records: Sequence[Dict]) -> Tuple[List[int], List[str]]:
    frame_map: Dict[int, str] = {}
    for record in records:
        sampled_indices = record.get("sampled_frame_indices", list(range(len(record["frame_paths"]))))
        for source_index, frame_path in zip(sampled_indices, record["frame_paths"]):
            normalized_index = int(source_index)
            if normalized_index in frame_map and frame_map[normalized_index] != frame_path:
                raise ValueError(
                    f"Conflicting frame path for source index {normalized_index}: "
                    f"{frame_map[normalized_index]} vs {frame_path}"
                )
            frame_map[normalized_index] = frame_path
    ordered_indices = sorted(frame_map)
    ordered_paths = [frame_map[index] for index in ordered_indices]
    return ordered_indices, ordered_paths


def build_source_payload(
    source_sample_id: str,
    source_dataset: str,
    source_frame_indices: Sequence[int],
    source_frame_paths: Sequence[str],
    image_processor,
    extractor: VGGTBankExtractor,
    projector: GeoProjector,
    device: torch.device,
    feature_layers: Sequence[str],
) -> Dict:
    geometry_inputs = load_geometry_inputs(source_frame_paths, image_processor)
    if device.type == "cuda":
        geometry_inputs = geometry_inputs.to(device)
    with torch.no_grad():
        extracted = extractor.extract(geometry_inputs)
        projected = projector(
            {
                raw_feature_name(layer_name): extracted.layer_tokens[raw_feature_name(layer_name)]
                for layer_name in feature_layers
            }
        )

    payload = {
        "cache_format": PROJECTED_SOURCE_CACHE_FORMAT,
        "source_sample_id": source_sample_id,
        "source_dataset": source_dataset,
        "source_frame_indices": [int(value) for value in source_frame_indices],
        "source_frame_paths": list(source_frame_paths),
        "layer_names": list(feature_layers),
        "token_counts": [int(value) for value in extracted.frame_layout.token_counts],
        "frame_shapes": [list(shape) for shape in extracted.frame_layout.frame_shapes],
        "patch_grid": list(extracted.patch_grid),
        "merged_grid": list(extracted.merged_grid),
    }
    for layer_name in feature_layers:
        quantized, scales = quantize_projected_tokenwise(projected[layer_name])
        payload[f"{layer_name}_q"] = quantized
        payload[f"{layer_name}_scale"] = scales
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_path", type=str, required=True)
    parser.add_argument("--output_cache_dir", type=str, required=True)
    parser.add_argument("--output_manifest_path", type=str, required=True)
    parser.add_argument("--projector_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--geometry_encoder_path", type=str, required=True)
    parser.add_argument("--d_geom", type=int, default=1024)
    parser.add_argument("--feature_layers", type=str, default="g11,g17,g23")
    parser.add_argument("--max_sources", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--overwrite", type=str, default="False")
    args = parser.parse_args()

    overwrite = str(args.overwrite).lower() in {"1", "true", "yes", "y"}
    feature_layers = parse_feature_layers(args.feature_layers)
    os.makedirs(args.output_cache_dir, exist_ok=True)
    output_manifest_dir = os.path.dirname(args.output_manifest_path)
    if output_manifest_dir:
        os.makedirs(output_manifest_dir, exist_ok=True)

    with open(args.input_manifest_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    grouped_records = group_manifest_records(records)
    if args.max_sources > 0:
        grouped_records = grouped_records[: args.max_sources]
    if args.num_shards > 1:
        if not (0 <= args.shard_rank < args.num_shards):
            raise ValueError(f"Invalid shard_rank={args.shard_rank} for num_shards={args.num_shards}")
        grouped_records = [
            item for index, item in enumerate(grouped_records) if index % args.num_shards == args.shard_rank
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoProcessor.from_pretrained(args.model_name_or_path).image_processor
    layer_id_map = {"g11": 11, "g17": 17, "g23": 23}
    extractor = VGGTBankExtractor(
        model_path=args.geometry_encoder_path,
        layer_ids=tuple(layer_id_map[name] for name in feature_layers),
        spatial_merge_size=getattr(image_processor, "merge_size", 2),
        freeze_encoder=True,
    ).eval()
    if device.type == "cuda":
        extractor = extractor.to(device)
    projector = load_projector(
        args.projector_checkpoint_path,
        d_geom=args.d_geom,
        device=device,
        feature_layers=feature_layers,
    )

    with open(args.output_manifest_path, "w", encoding="utf-8") as manifest_handle:
        processed_sources = 0
        processed_rows = 0
        for source_sample_id, source_records in grouped_records:
            source_dataset = source_records[0].get("source_dataset", "unknown")
            source_frame_indices, source_frame_paths = collect_source_frames(source_records)
            cache_path = source_cache_path(args.output_cache_dir, source_sample_id)
            if overwrite or not os.path.exists(cache_path):
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                payload = build_source_payload(
                    source_sample_id,
                    source_dataset,
                    source_frame_indices,
                    source_frame_paths,
                    image_processor,
                    extractor,
                    projector,
                    device,
                    feature_layers,
                )
                torch.save(payload, cache_path)

            for record in source_records:
                updated = dict(record)
                updated["cache_path"] = cache_path
                updated["cache_format"] = PROJECTED_SOURCE_CACHE_FORMAT
                manifest_handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
                processed_rows += 1
            processed_sources += 1
            if processed_sources % 100 == 0:
                manifest_handle.flush()
                print(
                    f"[joint-cache] processed_sources={processed_sources} processed_rows={processed_rows}",
                    flush=True,
                )

    print(f"[joint-cache] output_manifest_path={args.output_manifest_path}")
    print(f"[joint-cache] processed_sources={processed_sources}")
    print(f"[joint-cache] processed_rows={processed_rows}")


if __name__ == "__main__":
    main()

"""Build correspondence-graph cache for Stage 1 v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qwen_vl.model.geometry_bank import GeoProjector
from qwen_vl.data.utils import prepare_image_inputs
from qwen_vl.model.geometry_bank import VGGTBankExtractor
from qwen_vl.train.stage1_compact_cache import (
    is_projected_source_cache,
    materialize_projected_source_cache,
)


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_feature_layers(value: str) -> List[str]:
    layers = [item.strip() for item in str(value).split(",") if item.strip()]
    if not layers:
        raise ValueError("feature_layers must not be empty")
    valid = {"g11", "g17", "g23"}
    invalid = [layer for layer in layers if layer not in valid]
    if invalid:
        raise ValueError(f"Unsupported feature_layers: {invalid}")
    return layers


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


def resolve_frame_path(frame_path: str) -> str:
    if os.path.isabs(frame_path):
        return frame_path
    candidate = os.path.join(str(project_root), frame_path)
    if os.path.exists(candidate):
        return candidate
    return frame_path


def load_geometry_inputs(frame_paths: List[str], image_processor) -> torch.Tensor:
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


def load_feature_extractor(
    model_name_or_path: str,
    geometry_encoder_path: str,
    device: torch.device,
    feature_layers: Sequence[str],
):
    image_processor = AutoProcessor.from_pretrained(model_name_or_path).image_processor
    layer_id_map = {"g11": 11, "g17": 17, "g23": 23}
    extractor = VGGTBankExtractor(
        model_path=geometry_encoder_path,
        layer_ids=tuple(layer_id_map[name] for name in feature_layers),
        spatial_merge_size=getattr(image_processor, "merge_size", 2),
        freeze_encoder=True,
    ).eval()
    if device.type == "cuda":
        extractor = extractor.to(device)
    return image_processor, extractor


def feature_output_to_cached_payload(record: Dict, extracted, feature_layers: Sequence[str]) -> Dict:
    payload = {
        "group_id": record["group_id"],
        "source_dataset": record.get("source_dataset", "unknown"),
        "source_sample_id": record.get("source_sample_id", ""),
        "window_id": record.get("window_id", 0),
        "cache_window_mode": record.get("cache_window_mode", "multi_window"),
        "frame_paths": record["frame_paths"],
        "sampled_frame_indices": record.get("sampled_frame_indices", []),
        "valid_frame_mask": record.get("valid_frame_mask", []),
        "layer_names": list(feature_layers),
        "token_counts": extracted.frame_layout.token_counts,
        "frame_shapes": extracted.frame_layout.frame_shapes,
        "patch_grid": extracted.patch_grid,
        "merged_grid": extracted.merged_grid,
    }
    for layer_name in feature_layers:
        payload[f"{layer_name}_raw"] = extracted.layer_tokens[f"{layer_name}_raw"].cpu().to(torch.bfloat16)
    return payload


def sanitize_manifest_record(record: Dict) -> Dict:
    sanitized = dict(record)
    sanitized.pop("_preloaded_geometry_inputs", None)
    return sanitized


def extract_cached_payload_from_record(
    record: Dict,
    image_processor,
    extractor: VGGTBankExtractor,
    device: torch.device,
    feature_layers: Sequence[str],
) -> Dict:
    geometry_inputs = load_geometry_inputs(record["frame_paths"], image_processor)
    if device.type == "cuda":
        geometry_inputs = geometry_inputs.to(device)
    with torch.no_grad():
        extracted = extractor.extract(geometry_inputs)
    return feature_output_to_cached_payload(record, extracted, feature_layers)


def extract_cached_payloads_from_records(
    records: Sequence[Dict],
    image_processor,
    extractor: VGGTBankExtractor,
    device: torch.device,
    feature_layers: Sequence[str],
) -> List[Dict]:
    if not records:
        return []
    geometry_inputs_list = []
    for record in records:
        preloaded = record.get("_preloaded_geometry_inputs")
        if preloaded is None:
            preloaded = load_geometry_inputs(record["frame_paths"], image_processor)
        geometry_inputs_list.append(preloaded)
    reference_shape = tuple(geometry_inputs_list[0].shape)
    if not all(tuple(item.shape) == reference_shape for item in geometry_inputs_list):
        return [
            extract_cached_payload_from_record(record, image_processor, extractor, device, feature_layers)
            for record in records
        ]
    image_batch = torch.stack(geometry_inputs_list, dim=0)
    if device.type == "cuda":
        image_batch = image_batch.to(device)
    with torch.no_grad():
        extracted_batch = extractor.extract_batch(image_batch)
    return [
        feature_output_to_cached_payload(record, extracted, feature_layers)
        for record, extracted in zip(records, extracted_batch)
    ]


def build_feature_knn_corr_graph(
    cached_payload: Dict,
    projector: GeoProjector,
    *,
    feature_layers: Sequence[str],
    temporal_radius: int,
    topk_neighbors: int,
    feature_norm: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if cached_payload.get("feature_space") == "projected":
        feature = torch.cat(
            [cached_payload[layer_name].to(device=device, dtype=torch.float32) for layer_name in feature_layers],
            dim=-1,
        )
        if feature_norm:
            feature = F.normalize(feature.float(), dim=-1)
    else:
        raw_tokens = {
            f"{layer_name}_raw": cached_payload[f"{layer_name}_raw"].to(device=device, dtype=torch.float32)
            for layer_name in feature_layers
        }
        with torch.no_grad():
            projected = projector(raw_tokens)
            feature = torch.cat([projected[layer_name] for layer_name in feature_layers], dim=-1)
            if feature_norm:
                feature = F.normalize(feature.float(), dim=-1)

    token_counts = [int(value) for value in cached_payload["token_counts"]]
    num_frames, max_patches = feature.shape[:2]
    neighbor_indices = torch.full((num_frames, max_patches, topk_neighbors, 2), -1, dtype=torch.long)
    neighbor_scores = torch.zeros((num_frames, max_patches, topk_neighbors), dtype=torch.float32)

    for frame_idx in range(num_frames):
        count_t = token_counts[frame_idx]
        if count_t <= 0:
            continue
        for patch_idx in range(count_t):
            anchor = feature[frame_idx, patch_idx]
            candidates: List[torch.Tensor] = []
            candidate_meta: List[tuple[int, int]] = []
            for other_frame in range(max(0, frame_idx - temporal_radius), min(num_frames, frame_idx + temporal_radius + 1)):
                if other_frame == frame_idx:
                    continue
                count_s = token_counts[other_frame]
                if count_s <= 0:
                    continue
                candidates.append(feature[other_frame, :count_s])
                candidate_meta.extend((other_frame, other_patch) for other_patch in range(count_s))
            if not candidates:
                continue
            candidate_tensor = torch.cat(candidates, dim=0)
            scores = torch.matmul(candidate_tensor, anchor)
            limit = min(topk_neighbors, scores.numel())
            top_scores, top_indices = torch.topk(scores, k=limit, largest=True, sorted=True)
            for rank, flat_idx in enumerate(top_indices.tolist()):
                other_frame, other_patch = candidate_meta[flat_idx]
                neighbor_indices[frame_idx, patch_idx, rank, 0] = int(other_frame)
                neighbor_indices[frame_idx, patch_idx, rank, 1] = int(other_patch)
                neighbor_scores[frame_idx, patch_idx, rank] = float(top_scores[rank].item())

    return {
        "neighbor_indices": neighbor_indices,
        "neighbor_scores": neighbor_scores,
        "method": "feature_knn",
        "feature_layers": list(feature_layers),
        "temporal_radius": int(temporal_radius),
        "topk_neighbors": int(topk_neighbors),
        "feature_norm": bool(feature_norm),
    }


def load_cached_payload_for_record(record: Dict, cache_path: str) -> Dict:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if is_projected_source_cache(payload):
        sampled_frame_indices = record.get("sampled_frame_indices", list(range(len(record.get("frame_paths", [])))))
        return materialize_projected_source_cache(payload, sampled_frame_indices)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_manifest_path", type=str, required=True)
    parser.add_argument("--base_cache_dir", type=str, default="")
    parser.add_argument("--output_manifest_path", type=str, required=True)
    parser.add_argument("--corr_cache_dir", type=str, required=True)
    parser.add_argument("--projector_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--geometry_encoder_path", type=str, required=True)
    parser.add_argument("--d_geom", type=int, default=1024)
    parser.add_argument("--feature_layers", type=str, default="g11,g17,g23")
    parser.add_argument("--method", type=str, default="feature_knn", choices=("feature_knn", "vggt_3d_knn"))
    parser.add_argument("--temporal_radius", type=int, default=2)
    parser.add_argument("--topk_neighbors", type=int, default=8)
    parser.add_argument("--feature_norm", type=str, default="True")
    parser.add_argument("--max_groups", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--feature_batch_size", type=int, default=1)
    args = parser.parse_args()

    if args.method != "feature_knn":
        raise NotImplementedError("vggt_3d_knn interface is reserved but not implemented in this round")
    feature_layers = parse_feature_layers(args.feature_layers)

    os.makedirs(args.corr_cache_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projector = load_projector(
        args.projector_checkpoint_path,
        d_geom=args.d_geom,
        device=device,
        feature_layers=feature_layers,
    )
    image_processor = None
    extractor = None
    feature_norm = parse_bool(args.feature_norm)
    feature_batch_size = max(int(args.feature_batch_size), 1)
    manifest_dir = os.path.dirname(args.output_manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    with open(args.base_manifest_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if args.max_groups > 0:
        records = records[: args.max_groups]
    if args.num_shards > 1:
        if not (0 <= args.shard_rank < args.num_shards):
            raise ValueError(f"Invalid shard_rank={args.shard_rank} for num_shards={args.num_shards}")
        records = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_rank]

    corr_graph_dir = os.path.join(args.corr_cache_dir, "corr_graph")
    os.makedirs(corr_graph_dir, exist_ok=True)

    def process_record(record: Dict) -> Tuple[Dict, Dict]:
        cache_path = record.get("cache_path", "")
        if not cache_path:
            base_cache_dir = args.base_cache_dir or os.path.dirname(args.base_manifest_path)
            cache_path = os.path.join(base_cache_dir, "features", f"{record['group_id']}.pt")
        corr_cache_path = os.path.join(corr_graph_dir, f"{record['group_id']}.pt")
        if cache_path and os.path.exists(cache_path):
            cached_payload = load_cached_payload_for_record(record, cache_path)
        else:
            raise FileNotFoundError("process_record should only be used for existing cache paths")
        if not os.path.exists(corr_cache_path):
            corr_payload = build_feature_knn_corr_graph(
                cached_payload,
                projector,
                feature_layers=feature_layers,
                temporal_radius=args.temporal_radius,
                topk_neighbors=args.topk_neighbors,
                feature_norm=feature_norm,
                device=device,
            )
            corr_payload["group_id"] = record["group_id"]
            torch.save(corr_payload, corr_cache_path)
        new_record = dict(record)
        new_record["cache_path"] = cache_path if cache_path and os.path.exists(cache_path) else ""
        if cached_payload.get("cache_format"):
            new_record["cache_format"] = cached_payload["cache_format"]
        new_record["corr_cache_path"] = corr_cache_path
        new_record["corr_graph_method"] = args.method
        new_record["corr_temporal_radius"] = args.temporal_radius
        new_record["corr_topk_neighbors"] = args.topk_neighbors
        new_record["corr_feature_norm"] = feature_norm
        return new_record, {"corr_cache_path": corr_cache_path}

    def flush_pending_batch(
        pending_records: List[Dict],
        manifest_handle,
    ) -> int:
        if not pending_records:
            return 0
        nonlocal image_processor, extractor
        if image_processor is None or extractor is None:
            image_processor, extractor = load_feature_extractor(
                args.model_name_or_path,
                args.geometry_encoder_path,
                device=device,
                feature_layers=feature_layers,
            )
        cached_payloads = extract_cached_payloads_from_records(
            pending_records,
            image_processor,
            extractor,
            device,
            feature_layers,
        )
        written = 0
        for record, cached_payload in zip(pending_records, cached_payloads):
            corr_cache_path = os.path.join(corr_graph_dir, f"{record['group_id']}.pt")
            if not os.path.exists(corr_cache_path):
                corr_payload = build_feature_knn_corr_graph(
                    cached_payload,
                    projector,
                    feature_layers=feature_layers,
                    temporal_radius=args.temporal_radius,
                    topk_neighbors=args.topk_neighbors,
                    feature_norm=feature_norm,
                    device=device,
                )
                corr_payload["group_id"] = record["group_id"]
                torch.save(corr_payload, corr_cache_path)
            new_record = sanitize_manifest_record(record)
            new_record["cache_path"] = ""
            new_record["corr_cache_path"] = corr_cache_path
            new_record["corr_graph_method"] = args.method
            new_record["corr_temporal_radius"] = args.temporal_radius
            new_record["corr_topk_neighbors"] = args.topk_neighbors
            new_record["corr_feature_norm"] = feature_norm
            manifest_handle.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            written += 1
        return written

    with open(args.output_manifest_path, "w", encoding="utf-8") as manifest_handle:
        processed_count = 0
        pending_missing: List[Dict] = []
        pending_shape: Tuple[int, ...] | None = None
        for record in records:
            cache_path = record.get("cache_path", "")
            if not cache_path:
                base_cache_dir = args.base_cache_dir or os.path.dirname(args.base_manifest_path)
                cache_path = os.path.join(base_cache_dir, "features", f"{record['group_id']}.pt")
            if cache_path and os.path.exists(cache_path):
                processed_count += flush_pending_batch(pending_missing, manifest_handle)
                pending_missing = []
                pending_shape = None
                new_record, _ = process_record(record)
                manifest_handle.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                processed_count += 1
            else:
                if image_processor is None or extractor is None:
                    image_processor, extractor = load_feature_extractor(
                        args.model_name_or_path,
                        args.geometry_encoder_path,
                        device=device,
                        feature_layers=feature_layers,
                    )
                sample_inputs = load_geometry_inputs(record["frame_paths"], image_processor)
                sample_shape = tuple(sample_inputs.shape)
                record = sanitize_manifest_record(record)
                record["_preloaded_geometry_inputs"] = sample_inputs
                if pending_missing and (pending_shape != sample_shape or len(pending_missing) >= feature_batch_size):
                    processed_count += flush_pending_batch(pending_missing, manifest_handle)
                    pending_missing = []
                    pending_shape = None
                pending_missing.append(record)
                pending_shape = sample_shape

            if processed_count and processed_count % 100 == 0:
                manifest_handle.flush()
                print(f"[corr-cache] processed_groups={processed_count}", flush=True)

        processed_count += flush_pending_batch(pending_missing, manifest_handle)
        if processed_count and processed_count % 100 != 0:
            manifest_handle.flush()
            print(f"[corr-cache] processed_groups={processed_count}", flush=True)

    complete_flag = os.path.join(args.corr_cache_dir, ".complete")
    with open(complete_flag, "w", encoding="utf-8") as handle:
        handle.write("done\n")
    print(f"[corr-cache] output_manifest_path={args.output_manifest_path}")
    print(f"[corr-cache] complete_flag={complete_flag}")


if __name__ == "__main__":
    main()

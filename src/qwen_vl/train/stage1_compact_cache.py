"""Compact and window-ready cache helpers for Stage 1 geometry training."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch


PROJECTED_SOURCE_CACHE_FORMAT = "projected_int8_source_v1"
WINDOW_READY_SOURCE_JOINT_PACK_FORMAT = "window_ready_source_joint_fp16_v1"


def quantize_projected_tokenwise(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-wise symmetric int8 quantization for projected geometry features."""

    features = features.detach().to(dtype=torch.float32)
    scales = features.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    quantized = torch.clamp(torch.round(features / scales.unsqueeze(-1)), -127, 127).to(torch.int8)
    return quantized.cpu(), scales.to(dtype=torch.float16).cpu()


def dequantize_projected_tokenwise(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return quantized.to(dtype=torch.float32) * scales.to(dtype=torch.float32).unsqueeze(-1)


def is_projected_source_cache(payload: Dict) -> bool:
    return payload.get("cache_format") == PROJECTED_SOURCE_CACHE_FORMAT


def is_window_ready_source_joint_pack(payload: Dict) -> bool:
    return payload.get("pack_format") == WINDOW_READY_SOURCE_JOINT_PACK_FORMAT


def build_source_frame_index_map(source_frame_indices: Sequence[int]) -> Dict[int, int]:
    return {int(frame_index): offset for offset, frame_index in enumerate(source_frame_indices)}


def _select_source_cache_frames(payload: Dict, sampled_frame_indices: Sequence[int]) -> tuple[List[int], Dict]:
    if not is_projected_source_cache(payload):
        raise ValueError("Payload is not a projected source cache")

    source_frame_indices = payload.get("source_frame_indices")
    if source_frame_indices is None:
        raise KeyError("projected source cache missing source_frame_indices")
    frame_index_map = build_source_frame_index_map(source_frame_indices)
    selected_positions: List[int] = []
    for frame_index in sampled_frame_indices:
        normalized_index = int(frame_index)
        if normalized_index not in frame_index_map:
            raise KeyError(
                f"sampled frame index {normalized_index} missing from projected source cache "
                f"for source_sample_id={payload.get('source_sample_id', 'unknown')}"
            )
        selected_positions.append(frame_index_map[normalized_index])
    frame_shapes = [tuple(int(value) for value in payload["frame_shapes"][position]) for position in selected_positions]
    token_counts = [int(payload["token_counts"][position]) for position in selected_positions]

    metadata = {
        "cache_format": PROJECTED_SOURCE_CACHE_FORMAT,
        "layer_names": list(payload.get("layer_names", ("g11", "g17", "g23"))),
        "token_counts": token_counts,
        "frame_shapes": frame_shapes,
        "patch_grid": tuple(int(value) for value in payload.get("patch_grid", frame_shapes[0])),
        "merged_grid": tuple(int(value) for value in payload.get("merged_grid", frame_shapes[0])),
    }
    return selected_positions, metadata


def slice_projected_source_cache(payload: Dict, sampled_frame_indices: Sequence[int]) -> Dict:
    selected_positions, metadata = _select_source_cache_frames(payload, sampled_frame_indices)
    selected = torch.tensor(selected_positions, dtype=torch.long)
    output = {
        "feature_space": "projected_quantized",
        **metadata,
    }

    for layer_name in output["layer_names"]:
        quantized = payload[f"{layer_name}_q"].index_select(0, selected)
        scales = payload[f"{layer_name}_scale"].index_select(0, selected)
        output[f"{layer_name}_q"] = quantized
        output[f"{layer_name}_scale"] = scales

    return output


def materialize_projected_source_cache(
    payload: Dict,
    sampled_frame_indices: Sequence[int],
    *,
    output_dtype: torch.dtype = torch.float16,
) -> Dict:
    selected_positions, metadata = _select_source_cache_frames(payload, sampled_frame_indices)
    selected = torch.tensor(selected_positions, dtype=torch.long)
    output = {
        "feature_space": "projected",
        **metadata,
    }

    for layer_name in output["layer_names"]:
        quantized = payload[f"{layer_name}_q"].index_select(0, selected)
        scales = payload[f"{layer_name}_scale"].index_select(0, selected)
        output[layer_name] = dequantize_projected_tokenwise(quantized, scales).to(dtype=output_dtype)

    return output


def extract_window_from_source_joint_pack(payload: Dict, window_key: str) -> Dict:
    if not is_window_ready_source_joint_pack(payload):
        raise ValueError("Payload is not a window-ready Stage 1 source joint pack")
    windows = payload.get("windows", {})
    if window_key not in windows:
        raise KeyError(
            f"window key {window_key} missing from source joint pack "
            f"for source_sample_id={payload.get('source_sample_id', 'unknown')}"
        )
    return windows[window_key]


__all__ = [
    "PROJECTED_SOURCE_CACHE_FORMAT",
    "WINDOW_READY_SOURCE_JOINT_PACK_FORMAT",
    "build_source_frame_index_map",
    "dequantize_projected_tokenwise",
    "extract_window_from_source_joint_pack",
    "is_window_ready_source_joint_pack",
    "is_projected_source_cache",
    "materialize_projected_source_cache",
    "quantize_projected_tokenwise",
    "slice_projected_source_cache",
]

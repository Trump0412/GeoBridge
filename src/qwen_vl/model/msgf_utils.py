"""Utility helpers for DA3 MSGF interaction variants."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class StageRanges:
    warmup_start: int
    warmup_end: int
    write_start: int
    write_end: int
    read_start: int
    read_end: int
    init_start: int
    init_end: int
    refine_start: int
    refine_end: int
    active_end: int


@dataclass(frozen=True)
class FrameLayout:
    token_counts: List[int]
    frame_shapes: List[Tuple[int, int]]


def _clamp_layer_idx(value: int, num_hidden_layers: int) -> int:
    return max(0, min(int(value), max(0, num_hidden_layers - 1)))


def _ratio_to_layer(num_hidden_layers: int, ratio: float) -> int:
    idx = int(math.floor(num_hidden_layers * ratio)) - 1
    return _clamp_layer_idx(idx, num_hidden_layers)


def _resolve_int_attr(config, name: str, default: int) -> int:
    value = getattr(config, name, default)
    if value is None:
        return default
    value = int(value)
    if value < 0:
        return default
    return value


def compute_stage_ranges(num_hidden_layers: int, variant: str, config=None) -> StageRanges:
    """Compute config-driven stage ranges for 7B/3B backbones.

    Defaults follow the taskbook ratio mapping and can be overridden through
    explicit config attributes such as `msgf_warmup_end` or `rmsgf_refine_end`.
    """

    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")

    cfg = config
    if variant in {"msgf", "hmsgf", "sgf"}:
        warmup_start = _resolve_int_attr(cfg, f"{variant}_warmup_start", 0) if cfg else 0
        warmup_end = _resolve_int_attr(cfg, f"{variant}_warmup_end", _ratio_to_layer(num_hidden_layers, 0.28)) if cfg else _ratio_to_layer(num_hidden_layers, 0.28)
        write_start = _resolve_int_attr(cfg, f"{variant}_write_start", warmup_end + 1) if cfg else warmup_end + 1
        write_end = _resolve_int_attr(cfg, f"{variant}_write_end", _ratio_to_layer(num_hidden_layers, 0.57)) if cfg else _ratio_to_layer(num_hidden_layers, 0.57)
        read_start = _resolve_int_attr(cfg, f"{variant}_read_start", write_end + 1) if cfg else write_end + 1
        read_end = _resolve_int_attr(cfg, f"{variant}_read_end", _ratio_to_layer(num_hidden_layers, 0.75)) if cfg else _ratio_to_layer(num_hidden_layers, 0.75)
        active_end = read_end
        return StageRanges(
            warmup_start=_clamp_layer_idx(warmup_start, num_hidden_layers),
            warmup_end=_clamp_layer_idx(warmup_end, num_hidden_layers),
            write_start=_clamp_layer_idx(write_start, num_hidden_layers),
            write_end=_clamp_layer_idx(write_end, num_hidden_layers),
            read_start=_clamp_layer_idx(read_start, num_hidden_layers),
            read_end=_clamp_layer_idx(read_end, num_hidden_layers),
            init_start=-1,
            init_end=-1,
            refine_start=-1,
            refine_end=-1,
            active_end=_clamp_layer_idx(active_end, num_hidden_layers),
        )

    if variant == "rmsgf":
        warmup_start = 0
        warmup_end = _ratio_to_layer(num_hidden_layers, 0.28)
        init_start = _resolve_int_attr(cfg, "rmsgf_init_start", warmup_end + 1) if cfg else warmup_end + 1
        init_end = _resolve_int_attr(cfg, "rmsgf_init_end", _ratio_to_layer(num_hidden_layers, 0.46)) if cfg else _ratio_to_layer(num_hidden_layers, 0.46)
        refine_start = _resolve_int_attr(cfg, "rmsgf_refine_start", init_end + 1) if cfg else init_end + 1
        refine_end = _resolve_int_attr(cfg, "rmsgf_refine_end", _ratio_to_layer(num_hidden_layers, 0.75)) if cfg else _ratio_to_layer(num_hidden_layers, 0.75)
        return StageRanges(
            warmup_start=_clamp_layer_idx(warmup_start, num_hidden_layers),
            warmup_end=_clamp_layer_idx(warmup_end, num_hidden_layers),
            write_start=-1,
            write_end=-1,
            read_start=-1,
            read_end=-1,
            init_start=_clamp_layer_idx(init_start, num_hidden_layers),
            init_end=_clamp_layer_idx(init_end, num_hidden_layers),
            refine_start=_clamp_layer_idx(refine_start, num_hidden_layers),
            refine_end=_clamp_layer_idx(refine_end, num_hidden_layers),
            active_end=_clamp_layer_idx(refine_end, num_hidden_layers),
        )

    raise ValueError(f"Unknown stage variant: {variant}")


def infer_frame_layout(total_tokens: int, grid_thw: torch.Tensor | None, pooling_stride: int) -> FrameLayout:
    """Infer per-frame token layout from Qwen image/video grid metadata."""

    if total_tokens <= 0:
        return FrameLayout(token_counts=[], frame_shapes=[])
    if grid_thw is None or grid_thw.numel() == 0:
        return FrameLayout(token_counts=[total_tokens], frame_shapes=[(total_tokens, 1)])

    token_counts: List[int] = []
    frame_shapes: List[Tuple[int, int]] = []
    for row in grid_thw.detach().cpu().tolist():
        t = max(int(row[0]), 1)
        h = max(int(row[1]) // pooling_stride, 1)
        w = max(int(row[2]) // pooling_stride, 1)
        per_frame = max(h * w, 1)
        for _ in range(t):
            token_counts.append(per_frame)
            frame_shapes.append((h, w))

    if not token_counts:
        return FrameLayout(token_counts=[total_tokens], frame_shapes=[(total_tokens, 1)])

    count_sum = sum(token_counts)
    if count_sum == total_tokens:
        return FrameLayout(token_counts=token_counts, frame_shapes=frame_shapes)

    # Fallback for processor-specific packing differences: evenly distribute tokens.
    num_frames = len(token_counts)
    base = total_tokens // num_frames
    remainder = total_tokens % num_frames
    token_counts = [base + (1 if idx < remainder else 0) for idx in range(num_frames)]
    fallback_shapes = []
    for token_count in token_counts:
        h = int(math.sqrt(max(token_count, 1)))
        while h > 1 and token_count % h != 0:
            h -= 1
        w = max(token_count // max(h, 1), 1)
        fallback_shapes.append((max(h, 1), w))
    return FrameLayout(token_counts=token_counts, frame_shapes=fallback_shapes)


def split_by_layout(hidden_states: torch.Tensor, layout: FrameLayout) -> List[torch.Tensor]:
    """Split [S, H] or [1, S, H] hidden states into per-frame chunks."""

    if hidden_states.dim() == 3:
        hidden_states = hidden_states.squeeze(0)
    if hidden_states.numel() == 0 or not layout.token_counts:
        return []

    outputs: List[torch.Tensor] = []
    start = 0
    for token_count in layout.token_counts:
        end = min(start + token_count, hidden_states.shape[0])
        outputs.append(hidden_states[start:end])
        start = end
    if start < hidden_states.shape[0]:
        outputs[-1] = torch.cat([outputs[-1], hidden_states[start:]], dim=0)
    return outputs


def safe_topk(scores: torch.Tensor, topk: int):
    if scores.numel() == 0 or topk <= 0:
        empty = scores.new_empty((0,), dtype=torch.long)
        return scores.new_empty((0,)), empty
    actual_topk = min(int(topk), int(scores.numel()))
    values, indices = torch.topk(scores, k=actual_topk, dim=0)
    return values, indices


def mean_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        raise ValueError("Cannot mean-pool empty token tensor.")
    return tokens.mean(dim=0, keepdim=True)


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, dim=-1, p=2)

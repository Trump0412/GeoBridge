"""Utilities for DA3-MMR memory routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class MMRStageRanges:
    warmup_start: int
    warmup_end: int
    write_start: int
    write_end: int
    read_start: int
    read_end: int
    active_end: int


def _clamp(value: int, upper: int) -> int:
    return max(0, min(int(value), max(0, upper - 1)))


def _scaled_end(num_hidden_layers: int, numerator: int, denominator: int) -> int:
    return _clamp(round(num_hidden_layers * numerator / denominator) - 1, num_hidden_layers)


def compute_mmr_stage_ranges(num_hidden_layers: int, config=None) -> MMRStageRanges:
    """Default to 7B's 0-7/8-15/16-20 split, scaled to other depths."""

    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")

    warmup_start = getattr(config, "mmr_warmup_start", 0) if config is not None else 0
    warmup_end = getattr(config, "mmr_warmup_end", -1) if config is not None else -1
    write_start = getattr(config, "mmr_write_start", -1) if config is not None else -1
    write_end = getattr(config, "mmr_write_end", -1) if config is not None else -1
    read_start = getattr(config, "mmr_read_start", -1) if config is not None else -1
    read_end = getattr(config, "mmr_read_end", -1) if config is not None else -1

    if warmup_end is None or int(warmup_end) < 0:
        warmup_end = _scaled_end(num_hidden_layers, 8, 28)
    if write_start is None or int(write_start) < 0:
        write_start = _clamp(int(warmup_end) + 1, num_hidden_layers)
    if write_end is None or int(write_end) < 0:
        write_end = _scaled_end(num_hidden_layers, 16, 28)
    if read_start is None or int(read_start) < 0:
        read_start = _clamp(int(write_end) + 1, num_hidden_layers)
    if read_end is None or int(read_end) < 0:
        read_end = _scaled_end(num_hidden_layers, 21, 28)

    warmup_start = _clamp(warmup_start, num_hidden_layers)
    warmup_end = _clamp(warmup_end, num_hidden_layers)
    write_start = _clamp(write_start, num_hidden_layers)
    write_end = _clamp(write_end, num_hidden_layers)
    read_start = _clamp(read_start, num_hidden_layers)
    read_end = _clamp(read_end, num_hidden_layers)

    return MMRStageRanges(
        warmup_start=warmup_start,
        warmup_end=warmup_end,
        write_start=write_start,
        write_end=write_end,
        read_start=read_start,
        read_end=read_end,
        active_end=read_end,
    )


def continuity_bonus(
    current_frame_id: int,
    candidate_frame_id: int,
    use_temporal_continuity: bool,
) -> float:
    if not use_temporal_continuity:
        return 0.0
    return 1.0 / (1.0 + abs(int(current_frame_id) - int(candidate_frame_id)))


def view_bonus(
    current_view_id: Optional[int],
    candidate_view_id: Optional[int],
    use_view_continuity: bool,
) -> float:
    if not use_view_continuity:
        return 0.0
    if current_view_id is None or candidate_view_id is None:
        return 0.0
    return 1.0 if int(current_view_id) == int(candidate_view_id) else 0.0

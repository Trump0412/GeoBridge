"""Memory bank primitives for DA3-MMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class FrameMemory:
    feat: torch.Tensor
    frame_id: int
    view_id: Optional[int] = None


@dataclass
class RegionMemory:
    feat: torch.Tensor
    frame_id: int
    region_id: int
    view_id: Optional[int] = None


class FrameMemoryBank:
    def __init__(self, entries: Sequence[FrameMemory]):
        self.entries: List[FrameMemory] = list(entries)

    @classmethod
    def from_summaries(
        cls,
        frame_summaries: Sequence[torch.Tensor],
        frame_ids: Sequence[int],
        view_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> "FrameMemoryBank":
        entries: List[FrameMemory] = []
        for idx, feat in enumerate(frame_summaries):
            if feat.numel() == 0:
                continue
            entries.append(
                FrameMemory(
                    feat=feat,
                    frame_id=int(frame_ids[idx]),
                    view_id=None if view_ids is None else view_ids[idx],
                )
            )
        return cls(entries)

    @property
    def size(self) -> int:
        return len(self.entries)


class RegionMemoryBank:
    def __init__(self, entries: Sequence[RegionMemory]):
        self.entries: List[RegionMemory] = list(entries)

    @classmethod
    def from_atoms(
        cls,
        region_atoms: Sequence[torch.Tensor],
        frame_ids: Sequence[int],
        view_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> "RegionMemoryBank":
        entries: List[RegionMemory] = []
        for frame_idx, atoms in enumerate(region_atoms):
            if atoms.numel() == 0:
                continue
            for region_idx in range(atoms.shape[0]):
                entries.append(
                    RegionMemory(
                        feat=atoms[region_idx : region_idx + 1],
                        frame_id=int(frame_ids[frame_idx]),
                        region_id=int(region_idx),
                        view_id=None if view_ids is None else view_ids[frame_idx],
                    )
                )
        return cls(entries)

    @property
    def size(self) -> int:
        return len(self.entries)

"""Query-driven retrieval for DA3-MMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .mmr_memory import FrameMemoryBank, RegionMemoryBank
from .mmr_utils import continuity_bonus, view_bonus


@dataclass
class RetrievalResult:
    context: torch.Tensor
    available_frames: int
    available_regions: int
    frame_topk: int
    region_topk: int


class QueryDrivenMMRRetriever:
    def __init__(
        self,
        frame_topk_max: int,
        region_topk_max: int,
        use_view_continuity: bool = True,
        use_temporal_continuity: bool = True,
    ):
        self.frame_topk_max = int(frame_topk_max)
        self.region_topk_max = int(region_topk_max)
        self.use_view_continuity = bool(use_view_continuity)
        self.use_temporal_continuity = bool(use_temporal_continuity)

    def _score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        return torch.matmul(key, query.squeeze(0))

    def retrieve(
        self,
        query: torch.Tensor,
        frame_bank: FrameMemoryBank,
        region_bank: Optional[RegionMemoryBank] = None,
        current_frame_id: Optional[int] = None,
        current_view_id: Optional[int] = None,
        use_region_memory: bool = False,
    ) -> RetrievalResult:
        if frame_bank.size == 0:
            empty = query.new_zeros((0, query.shape[-1]))
            return RetrievalResult(empty, 0, 0, 0, 0)

        frame_feats = torch.cat([entry.feat for entry in frame_bank.entries], dim=0)
        frame_scores = self._score(query, frame_feats)
        if current_frame_id is not None:
            for idx, entry in enumerate(frame_bank.entries):
                frame_scores[idx] += continuity_bonus(
                    current_frame_id=current_frame_id,
                    candidate_frame_id=entry.frame_id,
                    use_temporal_continuity=self.use_temporal_continuity,
                )
                frame_scores[idx] += view_bonus(
                    current_view_id=current_view_id,
                    candidate_view_id=entry.view_id,
                    use_view_continuity=self.use_view_continuity,
                )

        frame_topk = min(self.frame_topk_max, int(frame_scores.numel()))
        _, frame_indices = torch.topk(frame_scores, k=frame_topk, dim=0)
        frame_context = frame_feats[frame_indices] if frame_indices.numel() > 0 else frame_feats[:0]

        if not use_region_memory or region_bank is None or region_bank.size == 0:
            return RetrievalResult(
                context=frame_context,
                available_frames=frame_bank.size,
                available_regions=0,
                frame_topk=int(frame_indices.numel()),
                region_topk=0,
            )

        candidate_frame_ids = {frame_bank.entries[idx].frame_id for idx in frame_indices.tolist()}
        region_entries = [entry for entry in region_bank.entries if entry.frame_id in candidate_frame_ids]
        if not region_entries:
            return RetrievalResult(
                context=frame_context,
                available_frames=frame_bank.size,
                available_regions=0,
                frame_topk=int(frame_indices.numel()),
                region_topk=0,
            )

        region_feats = torch.cat([entry.feat for entry in region_entries], dim=0)
        region_scores = self._score(query, region_feats)
        region_topk = min(self.region_topk_max, int(region_scores.numel()))
        _, region_indices = torch.topk(region_scores, k=region_topk, dim=0)
        region_context = region_feats[region_indices] if region_indices.numel() > 0 else region_feats[:0]

        return RetrievalResult(
            context=torch.cat([frame_context, region_context], dim=0) if frame_context.numel() > 0 else region_context,
            available_frames=frame_bank.size,
            available_regions=region_bank.size,
            frame_topk=int(frame_indices.numel()),
            region_topk=int(region_indices.numel()),
        )

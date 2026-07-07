"""Memory utilities for DA3 MSGF interaction variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn

from .msgf_utils import l2_normalize, mean_pool_tokens, safe_topk


@dataclass
class MemoryAtom:
    feat: torch.Tensor
    frame_id: int
    region_id: int
    view_id: int | None
    importance: torch.Tensor


@dataclass
class FrameMemory:
    feat: torch.Tensor
    frame_id: int
    view_id: int | None = None


@dataclass
class RegionMemory:
    feat: torch.Tensor
    frame_id: int
    region_id: int
    view_id: int | None = None


@dataclass
class RetrievedMemory:
    context: torch.Tensor
    available_frames: int
    available_atoms: int
    frame_topk: int
    atom_topk: int


def _similarity(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    if keys.numel() == 0:
        return keys.new_empty((0,))
    query = l2_normalize(query)
    keys = l2_normalize(keys)
    return torch.matmul(keys, query.squeeze(0))


class BiDirectionalMemoryBank:
    """Stores salient fused atoms and retrieves them with dynamic top-k."""

    def __init__(self, frame_atoms: Sequence[torch.Tensor]):
        self.frame_atoms = [atoms for atoms in frame_atoms if atoms.numel() > 0]
        if self.frame_atoms:
            self.frame_summaries = [mean_pool_tokens(atoms) for atoms in self.frame_atoms]
        else:
            self.frame_summaries = []

    @classmethod
    def from_frame_atoms(cls, frame_atoms: Sequence[torch.Tensor]) -> "BiDirectionalMemoryBank":
        return cls(frame_atoms)

    def retrieve(self, query: torch.Tensor, frame_topk_max: int, atom_topk_max: int) -> RetrievedMemory:
        if not self.frame_atoms:
            empty = query.new_zeros((0, query.shape[-1]))
            return RetrievedMemory(empty, 0, 0, 0, 0)

        frame_bank = torch.cat(self.frame_summaries, dim=0)
        frame_scores = _similarity(query, frame_bank)
        _, frame_indices = safe_topk(frame_scores, frame_topk_max)
        candidate_frames = [self.frame_atoms[idx] for idx in frame_indices.tolist()]

        if not candidate_frames:
            empty = query.new_zeros((0, query.shape[-1]))
            return RetrievedMemory(empty, len(self.frame_atoms), 0, 0, 0)

        candidate_atoms = torch.cat(candidate_frames, dim=0)
        atom_scores = _similarity(query, candidate_atoms)
        _, atom_indices = safe_topk(atom_scores, atom_topk_max)
        selected_atoms = candidate_atoms[atom_indices] if atom_indices.numel() > 0 else candidate_atoms[:0]
        return RetrievedMemory(
            context=selected_atoms,
            available_frames=len(self.frame_atoms),
            available_atoms=int(candidate_atoms.shape[0]),
            frame_topk=int(frame_indices.numel()),
            atom_topk=int(atom_indices.numel()),
        )

    @property
    def total_atoms(self) -> int:
        return sum(int(atoms.shape[0]) for atoms in self.frame_atoms)


class HierarchicalMemoryBank:
    """Two-level memory with frame summaries and region atoms."""

    def __init__(self, frame_summaries: Sequence[torch.Tensor], region_atoms: Sequence[torch.Tensor]):
        self.frame_summaries = [summary for summary in frame_summaries if summary.numel() > 0]
        self.region_atoms = [atoms for atoms in region_atoms if atoms.numel() > 0]

    @classmethod
    def from_frame_atoms(cls, frame_atoms: Sequence[torch.Tensor]) -> "HierarchicalMemoryBank":
        frame_summaries = [mean_pool_tokens(atoms) for atoms in frame_atoms if atoms.numel() > 0]
        return cls(frame_summaries=frame_summaries, region_atoms=frame_atoms)

    def retrieve(self, query: torch.Tensor, frame_topk_max: int, region_topk_max: int) -> RetrievedMemory:
        if not self.frame_summaries:
            empty = query.new_zeros((0, query.shape[-1]))
            return RetrievedMemory(empty, 0, 0, 0, 0)

        frame_bank = torch.cat(self.frame_summaries, dim=0)
        frame_scores = _similarity(query, frame_bank)
        _, frame_indices = safe_topk(frame_scores, frame_topk_max)
        candidate_regions = [self.region_atoms[idx] for idx in frame_indices.tolist()]
        if not candidate_regions:
            empty = query.new_zeros((0, query.shape[-1]))
            return RetrievedMemory(empty, len(self.frame_summaries), 0, 0, 0)

        regions = torch.cat(candidate_regions, dim=0)
        region_scores = _similarity(query, regions)
        _, region_indices = safe_topk(region_scores, region_topk_max)
        selected_regions = regions[region_indices] if region_indices.numel() > 0 else regions[:0]
        return RetrievedMemory(
            context=selected_regions,
            available_frames=len(self.frame_summaries),
            available_atoms=int(regions.shape[0]),
            frame_topk=int(frame_indices.numel()),
            atom_topk=int(region_indices.numel()),
        )


class MemoryRefiner(nn.Module):
    """Residual refiner for recurrent MSGF variants."""

    def __init__(self, hidden_size: int, use_gate: bool = True, residual: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.residual = residual
        self.atom_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.update = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        ) if use_gate else None

    def forward(self, atoms: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if atoms.numel() == 0:
            return atoms
        if context.dim() == 2 and context.shape[0] == 1:
            context = context.expand_as(atoms)
        elif context.dim() == 1:
            context = context.unsqueeze(0).expand_as(atoms)
        elif context.shape[0] != atoms.shape[0]:
            context = context[:1].expand_as(atoms)

        fused = torch.cat([self.atom_norm(atoms), self.context_norm(context)], dim=-1)
        update = self.update(fused)
        if self.gate is not None:
            update = update * self.gate(fused)
        if self.residual:
            return atoms + update
        return update

"""Activated correspondence graph for GeoBridge Stage 1."""

from __future__ import annotations

import torch
import torch.nn as nn


def _edge_reciprocity(neighbor_indices: torch.Tensor, valid_edges: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames, num_patches, topk, _ = neighbor_indices.shape
    if topk == 0:
        return torch.zeros((batch_size, num_frames, num_patches, topk), device=neighbor_indices.device, dtype=torch.float32)

    neighbor_frames = neighbor_indices[..., 0].clamp(0, max(num_frames - 1, 0))
    neighbor_patches = neighbor_indices[..., 1].clamp(0, max(num_patches - 1, 0))
    batch_index = torch.arange(batch_size, device=neighbor_indices.device).view(batch_size, 1, 1, 1).expand_as(neighbor_frames)

    reverse_neighbors = neighbor_indices[batch_index, neighbor_frames, neighbor_patches]
    reverse_valid = valid_edges[batch_index, neighbor_frames, neighbor_patches]
    source_frames = torch.arange(num_frames, device=neighbor_indices.device).view(1, num_frames, 1, 1, 1)
    source_patches = torch.arange(num_patches, device=neighbor_indices.device).view(1, 1, num_patches, 1, 1)
    reciprocal = (
        (reverse_neighbors[..., 0] == source_frames)
        & (reverse_neighbors[..., 1] == source_patches)
        & reverse_valid
    ).any(dim=-1)
    return (reciprocal & valid_edges).to(dtype=torch.float32)


class ActivatedCorrespondenceGraph(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        neighbor_indices: torch.Tensor,
        neighbor_scores: torch.Tensor,
        saliency_probs: torch.Tensor,
    ) -> dict:
        valid_edges = (neighbor_indices[..., 0] >= 0) & (neighbor_indices[..., 1] >= 0)
        reciprocity = _edge_reciprocity(neighbor_indices, valid_edges)

        num_frames = saliency_probs.shape[1]
        denom = float(max(num_frames - 1, 1))
        src_saliency = saliency_probs.unsqueeze(-1).expand_as(neighbor_scores)

        neighbor_frames = neighbor_indices[..., 0].clamp_min(0)
        neighbor_patches = neighbor_indices[..., 1].clamp_min(0)
        batch_index = torch.arange(saliency_probs.shape[0], device=saliency_probs.device).view(-1, 1, 1, 1).expand_as(neighbor_frames)
        dst_saliency = saliency_probs[batch_index, neighbor_frames, neighbor_patches]
        time_offset = (torch.arange(num_frames, device=saliency_probs.device).view(1, num_frames, 1, 1) - neighbor_frames).abs().float() / denom

        features = torch.stack(
            [
                neighbor_scores.float(),
                reciprocity,
                time_offset,
                src_saliency.float(),
                dst_saliency.float(),
            ],
            dim=-1,
        )
        compute_dtype = self.mlp[0].weight.dtype
        logits = self.mlp(features.to(dtype=compute_dtype)).squeeze(-1)
        activation = torch.sigmoid(logits) * valid_edges.float()
        return {
            "activation": activation,
            "logits": logits,
            "features": features,
            "reciprocity": reciprocity,
            "valid_edges": valid_edges,
        }

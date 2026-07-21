"""Continuity utility selector for SpatialFit Stage 1."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_selector_shapes(
    z: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """Strict shape invariant check. Raises RuntimeError with full debug info on mismatch."""
    if z.dim() != 4:
        raise RuntimeError(
            f"[ContinuitySelector] z must be 4D (B, T, P, D), got shape {tuple(z.shape)}. "
            f"Caller must unsqueeze(0) if z is (T, P, D)."
        )
    B_z, T_z, P_z, D_z = z.shape
    if neighbor_indices.dim() != 5:
        raise RuntimeError(
            f"[ContinuitySelector] neighbor_indices must be 5D (B, T, P, K, 2), "
            f"got shape {tuple(neighbor_indices.shape)}"
        )
    B_ni, T_ni, P_ni, K_ni, two = neighbor_indices.shape
    B_ns, T_ns, P_ns, K_ns = neighbor_scores.shape
    B_vm, T_vm, P_vm = valid_mask.shape

    mismatches = []
    if P_z != P_ni:
        mismatches.append(f"z.P={P_z} != neighbor_indices.P={P_ni}")
    if P_z != P_ns:
        mismatches.append(f"z.P={P_z} != neighbor_scores.P={P_ns}")
    if P_z != P_vm:
        mismatches.append(f"z.P={P_z} != valid_mask.P={P_vm}")
    if T_z != T_ni:
        mismatches.append(f"z.T={T_z} != neighbor_indices.T={T_ni}")
    if B_z != B_vm:
        mismatches.append(f"z.B={B_z} != valid_mask.B={B_vm}")

    valid_edges = (neighbor_indices[..., 0] >= 0) & (neighbor_indices[..., 1] >= 0)
    if valid_edges.any():
        max_frame_idx = neighbor_indices[..., 0][valid_edges].max().item()
        max_patch_idx = neighbor_indices[..., 1][valid_edges].max().item()
        if max_frame_idx >= T_z:
            mismatches.append(f"neighbor_indices max frame_idx={max_frame_idx} >= T={T_z}")
        if max_patch_idx >= P_vm:
            mismatches.append(
                f"neighbor_indices max patch_idx={max_patch_idx} >= valid_mask.P={P_vm} "
                f"(z.P={P_z}, neighbor_indices.P={P_ni})"
            )

    if mismatches:
        raise RuntimeError(
            f"[ContinuitySelector] Shape invariant violation in build_selector_stats:\n"
            f"  z.shape={tuple(z.shape)}\n"
            f"  neighbor_indices.shape={tuple(neighbor_indices.shape)}\n"
            f"  neighbor_scores.shape={tuple(neighbor_scores.shape)}\n"
            f"  valid_mask.shape={tuple(valid_mask.shape)}\n"
            f"  Mismatches: {'; '.join(mismatches)}"
        )


def _compute_reciprocity_ratio(neighbor_indices: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames, num_patches, topk, _ = neighbor_indices.shape
    if topk == 0:
        return torch.zeros((batch_size, num_frames, num_patches), device=neighbor_indices.device, dtype=torch.float32)

    valid_edges = (neighbor_indices[..., 0] >= 0) & (neighbor_indices[..., 1] >= 0)
    neighbor_frames = neighbor_indices[..., 0].clamp(0, max(num_frames - 1, 0))
    neighbor_patches = neighbor_indices[..., 1].clamp(0, max(num_patches - 1, 0))
    batch_index = torch.arange(batch_size, device=neighbor_indices.device).view(batch_size, 1, 1, 1).expand_as(neighbor_frames)

    reverse_neighbors = neighbor_indices[batch_index, neighbor_frames, neighbor_patches]
    reverse_valid = valid_edges[batch_index, neighbor_frames, neighbor_patches]
    source_frames = torch.arange(num_frames, device=neighbor_indices.device).view(1, num_frames, 1, 1, 1)
    source_patches = torch.arange(num_patches, device=neighbor_indices.device).view(1, 1, num_patches, 1, 1)
    reciprocal_edges = (
        (reverse_neighbors[..., 0] == source_frames)
        & (reverse_neighbors[..., 1] == source_patches)
        & reverse_valid
    ).any(dim=-1)
    reciprocal_edges = reciprocal_edges & valid_edges
    edge_count = valid_edges.sum(dim=-1).clamp_min(1)
    reciprocity = reciprocal_edges.float().sum(dim=-1) / edge_count
    return reciprocity * valid_mask.float()


def build_selector_stats(
    z: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return selector stats and per-node reciprocity ratio.

    Stats order:
    1. local geometry contrast
    2. mean valid KNN score
    3. reciprocity ratio
    4. temporal coverage ratio
    """
    _validate_selector_shapes(z, neighbor_indices, neighbor_scores, valid_mask)

    z_float = z.float()
    frame_mean = (
        z_float * valid_mask.unsqueeze(-1).to(dtype=z_float.dtype)
    ).sum(dim=2, keepdim=True) / valid_mask.sum(dim=2, keepdim=True).clamp_min(1).unsqueeze(-1)
    local_contrast = (z_float - frame_mean).pow(2).mean(dim=-1).sqrt()

    edge_valid = (neighbor_indices[..., 0] >= 0) & (neighbor_indices[..., 1] >= 0)
    mean_knn = (neighbor_scores.float() * edge_valid.float()).sum(dim=-1) / edge_valid.sum(dim=-1).clamp_min(1.0)

    reciprocity_ratio = _compute_reciprocity_ratio(neighbor_indices, valid_mask)

    num_frames = z.shape[1]
    denom = float(max(num_frames - 1, 1))
    neighbor_frames = neighbor_indices[..., 0].clamp(0, max(num_frames - 1, 0))
    frame_seen = F.one_hot(neighbor_frames, num_classes=num_frames).bool() & edge_valid.unsqueeze(-1)
    time_coverage = frame_seen.any(dim=-2).sum(dim=-1).to(dtype=local_contrast.dtype) / denom
    time_coverage = time_coverage * valid_mask.to(dtype=time_coverage.dtype)

    stats = torch.stack([local_contrast, mean_knn, reciprocity_ratio, time_coverage], dim=-1)
    stats = stats * valid_mask.unsqueeze(-1).to(dtype=stats.dtype)
    return stats, reciprocity_ratio


class ContinuityUtilitySelector(nn.Module):
    def __init__(self, d_geom: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_dim = max(int(d_geom * hidden_ratio), 64)
        self.token_norm = nn.LayerNorm(d_geom)
        self.token_proj = nn.Linear(d_geom, hidden_dim)
        self.stat_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden_dim // 2),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_scores: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict:
        stats, reciprocity_ratio = build_selector_stats(z, neighbor_indices, neighbor_scores, valid_mask)
        compute_dtype = self.token_norm.weight.dtype
        z_cast = z.to(dtype=compute_dtype)
        token_feat = self.token_proj(self.token_norm(z_cast))
        stat_feat = self.stat_proj(stats.to(dtype=compute_dtype))
        fused = torch.cat([token_feat, stat_feat], dim=-1)
        logits = self.head(fused).squeeze(-1)
        probs = torch.sigmoid(logits) * valid_mask.to(dtype=logits.dtype)
        return {
            "logits": logits,
            "probs": probs,
            "stats": stats,
            "reciprocity_ratio": reciprocity_ratio,
        }


def continuity_utility_loss(
    logits: torch.Tensor,
    utility_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    budget_ratio: float,
    budget_weight: float,
) -> Tuple[torch.Tensor, dict]:
    target = utility_target.detach().clamp(0.0, 1.0)
    valid = valid_mask.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)

    probs = torch.sigmoid(logits) * valid
    mean_budget = probs.sum() / valid.sum().clamp_min(1.0)
    budget = (mean_budget - float(budget_ratio)) ** 2
    total = bce + float(budget_weight) * budget
    return total, {
        "bce": bce.detach(),
        "budget": budget.detach(),
        "mean_budget": mean_budget.detach(),
    }

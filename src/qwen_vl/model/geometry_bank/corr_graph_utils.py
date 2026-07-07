"""Utility helpers for lightweight correspondence graph construction."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def build_feature_knn_corr_graph_batch(
    z: torch.Tensor,
    token_counts: Sequence[int],
    *,
    temporal_radius: int = 2,
    topk_neighbors: int = 8,
    feature_norm: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build a lightweight feature-KNN correspondence graph from projected geometry.

    Args:
        z: Tensor of shape ``[T, P, D]``.
        token_counts: Valid token count for each frame. Must satisfy
            ``token_counts[i] <= P`` for all i.
    """

    if z.dim() != 3:
        raise ValueError(f"Expected z with shape [T, P, D], got {tuple(z.shape)}")

    num_frames, max_patches, _ = z.shape

    for i, tc in enumerate(token_counts):
        if int(tc) > max_patches:
            raise ValueError(
                f"[build_feature_knn_corr_graph_batch] token_counts[{i}]={tc} > "
                f"z.shape[1]={max_patches}. This indicates a shape mismatch between "
                f"the feature tensor and frame layout metadata."
            )

    topk_neighbors = int(topk_neighbors)
    if topk_neighbors <= 0:
        return {
            "neighbor_indices": torch.empty((num_frames, max_patches, 0, 2), dtype=torch.long, device=z.device),
            "neighbor_scores": torch.empty((num_frames, max_patches, 0), dtype=torch.float32, device=z.device),
            "method": "feature_knn",
            "temporal_radius": int(temporal_radius),
            "topk_neighbors": topk_neighbors,
        }

    z_score = F.normalize(z.float(), dim=-1) if feature_norm else z.float()
    counts = torch.as_tensor(
        [min(int(token_counts[i]) if i < len(token_counts) else max_patches, max_patches) for i in range(num_frames)],
        device=z.device,
        dtype=torch.long,
    )
    patch_ids = torch.arange(max_patches, device=z.device)
    valid_mask = patch_ids.unsqueeze(0) < counts.unsqueeze(1)

    frame_ids = torch.arange(num_frames, device=z.device)
    temporal_mask = (frame_ids[:, None] - frame_ids[None, :]).abs() <= int(temporal_radius)
    temporal_mask = temporal_mask & (frame_ids[:, None] != frame_ids[None, :])

    scores = torch.einsum("tpd,ukd->tpuk", z_score, z_score)
    candidate_mask = (
        valid_mask[:, :, None, None]
        & valid_mask[None, None, :, :]
        & temporal_mask[:, None, :, None]
    )
    scores = scores.masked_fill(~candidate_mask, float("-inf")).reshape(num_frames, max_patches, -1)

    total_candidates = num_frames * max_patches
    k_eff = min(topk_neighbors, total_candidates)
    topk_score, topk_flat_idx = torch.topk(scores, k=k_eff, dim=-1)
    valid_topk = torch.isfinite(topk_score)

    flat_frame_ids = frame_ids[:, None].expand(num_frames, max_patches).reshape(-1)
    flat_patch_ids = patch_ids[None, :].expand(num_frames, max_patches).reshape(-1)
    topk_frame = flat_frame_ids[topk_flat_idx]
    topk_patch = flat_patch_ids[topk_flat_idx]

    neighbor_indices = torch.full((num_frames, max_patches, topk_neighbors, 2), -1, dtype=torch.long, device=z.device)
    neighbor_scores = torch.zeros((num_frames, max_patches, topk_neighbors), dtype=torch.float32, device=z.device)
    neighbor_indices[..., :k_eff, 0] = topk_frame.masked_fill(~valid_topk, -1)
    neighbor_indices[..., :k_eff, 1] = topk_patch.masked_fill(~valid_topk, -1)
    neighbor_scores[..., :k_eff] = topk_score.masked_fill(~valid_topk, 0.0).to(dtype=torch.float32)

    return {
        "neighbor_indices": neighbor_indices,
        "neighbor_scores": neighbor_scores,
        "method": "feature_knn",
        "temporal_radius": int(temporal_radius),
        "topk_neighbors": int(topk_neighbors),
    }

"""Deterministic masks and frame permutations for Stage1 CBEval."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, Iterable, List, Sequence

import torch


def seed_from_parts(base_seed: int, *parts: Iterable[str]) -> int:
    digest = hashlib.sha1(str(base_seed).encode("utf-8"))
    for part in parts:
        digest.update(b"|")
        digest.update(str(part).encode("utf-8"))
    return int(digest.hexdigest()[:8], 16)


def apply_frame_permutations(tensor: torch.Tensor, permutations: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [tensor[index].index_select(0, permutations[index]) for index in range(tensor.shape[0])],
        dim=0,
    )


def restore_frame_permutations(tensor: torch.Tensor, permutations: Sequence[torch.Tensor]) -> torch.Tensor:
    inverse = [torch.argsort(permutation) for permutation in permutations]
    return torch.stack(
        [tensor[index].index_select(0, inverse[index]) for index in range(tensor.shape[0])],
        dim=0,
    )


def build_frame_permutations(
    sample_keys: Sequence[str],
    valid_frame_mask: torch.Tensor,
    *,
    shuffle_seed: int,
) -> List[torch.Tensor]:
    permutations: List[torch.Tensor] = []
    total_frames = valid_frame_mask.shape[1]
    device = valid_frame_mask.device
    for batch_index, sample_key in enumerate(sample_keys):
        valid_frames = [index for index, flag in enumerate(valid_frame_mask[batch_index].tolist()) if flag]
        perm = list(range(total_frames))
        if len(valid_frames) > 1:
            rng = random.Random(seed_from_parts(shuffle_seed, sample_key, "frame_permutation"))
            shuffled = list(valid_frames)
            rng.shuffle(shuffled)
            for target_index, source_index in zip(valid_frames, shuffled):
                perm[target_index] = source_index
        permutations.append(torch.tensor(perm, dtype=torch.long, device=device))
    return permutations


def _ensure_mask(mask: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return mask
    valid_coords = valid_token_mask.nonzero(as_tuple=False)
    if valid_coords.numel() == 0:
        return mask
    first_frame, first_patch = valid_coords[0].tolist()
    mask[first_frame, first_patch] = True
    return mask


def _build_random_patch_mask_single(
    valid_token_mask: torch.Tensor,
    *,
    masked_ratio: float,
    rng: random.Random,
) -> torch.Tensor:
    mask = torch.zeros_like(valid_token_mask)
    coords = [tuple(coord.tolist()) for coord in valid_token_mask.nonzero(as_tuple=False)]
    if not coords:
        return mask
    target_count = max(1, int(math.ceil(len(coords) * masked_ratio)))
    chosen = coords if target_count >= len(coords) else rng.sample(coords, k=target_count)
    for frame_index, patch_index in chosen:
        mask[frame_index, patch_index] = True
    return _ensure_mask(mask, valid_token_mask)


def _build_frame_block_mask_single(
    valid_token_mask: torch.Tensor,
    *,
    masked_ratio: float,
    rng: random.Random,
) -> torch.Tensor:
    mask = torch.zeros_like(valid_token_mask)
    valid_frames = [index for index, flag in enumerate(valid_token_mask.any(dim=-1).tolist()) if flag]
    if not valid_frames:
        return mask
    frame_index = rng.choice(valid_frames)
    valid_patches = [index for index, flag in enumerate(valid_token_mask[frame_index].tolist()) if flag]
    if not valid_patches:
        return _ensure_mask(mask, valid_token_mask)
    block_size = max(1, int(math.ceil(len(valid_patches) * masked_ratio)))
    if block_size >= len(valid_patches):
        chosen = valid_patches
    else:
        start = rng.randrange(0, len(valid_patches) - block_size + 1)
        chosen = valid_patches[start : start + block_size]
    for patch_index in chosen:
        mask[frame_index, patch_index] = True
    return _ensure_mask(mask, valid_token_mask)


def _build_corr_tube_mask_single(
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    masked_ratio: float,
    positive_topk: int,
    rng: random.Random,
) -> torch.Tensor:
    mask = torch.zeros_like(valid_token_mask)
    coords = [tuple(coord.tolist()) for coord in valid_token_mask.nonzero(as_tuple=False)]
    if not coords:
        return mask
    anchor_count = max(1, int(math.ceil(len(coords) * masked_ratio * 0.35)))
    anchors = coords if anchor_count >= len(coords) else rng.sample(coords, k=anchor_count)
    for frame_index, patch_index in anchors:
        mask[frame_index, patch_index] = True
        current_neighbors = neighbor_indices[frame_index, patch_index, :positive_topk]
        for other_frame, other_patch in current_neighbors.tolist():
            if other_frame < 0 or other_patch < 0:
                continue
            if other_frame >= valid_token_mask.shape[0] or other_patch >= valid_token_mask.shape[1]:
                continue
            if valid_token_mask[other_frame, other_patch]:
                mask[other_frame, other_patch] = True
    return _ensure_mask(mask, valid_token_mask)


def build_eval_masks(
    sample_keys: Sequence[str],
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    mask_seed: int,
    masked_ratio: float,
    positive_topk: int,
) -> Dict[str, torch.Tensor]:
    batch_size = valid_token_mask.shape[0]
    masks = {
        "random_patch": torch.zeros_like(valid_token_mask),
        "corr_tube": torch.zeros_like(valid_token_mask),
        "frame_block": torch.zeros_like(valid_token_mask),
    }
    for batch_index in range(batch_size):
        sample_valid = valid_token_mask[batch_index]
        sample_neighbors = neighbor_indices[batch_index]
        masks["random_patch"][batch_index] = _build_random_patch_mask_single(
            sample_valid,
            masked_ratio=masked_ratio,
            rng=random.Random(seed_from_parts(mask_seed, sample_keys[batch_index], "random_patch")),
        )
        masks["corr_tube"][batch_index] = _build_corr_tube_mask_single(
            sample_valid,
            sample_neighbors,
            masked_ratio=masked_ratio,
            positive_topk=positive_topk,
            rng=random.Random(seed_from_parts(mask_seed, sample_keys[batch_index], "corr_tube")),
        )
        masks["frame_block"][batch_index] = _build_frame_block_mask_single(
            sample_valid,
            masked_ratio=masked_ratio,
            rng=random.Random(seed_from_parts(mask_seed, sample_keys[batch_index], "frame_block")),
        )
    return masks

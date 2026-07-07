"""Metrics for Stage1 CBEval."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from qwen_vl.eval.stage1_cbeval.masks import seed_from_parts
from qwen_vl.model.geometry_bank.correspondence_losses import masked_cosine_similarity_metric, masked_l1_metric


def compute_mccr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    return {
        "cos": float(masked_cosine_similarity_metric(prediction, target, mask).item()),
        "l1": float(masked_l1_metric(prediction, target, mask).item()),
    }


def compute_norm_metric(representation: torch.Tensor, valid_token_mask: torch.Tensor) -> float:
    selected = representation[valid_token_mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.norm(dim=-1).mean().item())


def compute_cosine_alignment(representation: torch.Tensor, target: torch.Tensor, valid_token_mask: torch.Tensor) -> float:
    return float(masked_cosine_similarity_metric(representation, target, valid_token_mask).item())


def compute_tcs(
    representation: torch.Tensor,
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    sample_keys: Sequence[str],
    seed: int,
    positive_topk: int,
    max_anchors: int,
    num_negatives: int,
) -> Dict[str, float]:
    normalized = F.normalize(representation.float(), dim=-1)
    recall1_values: List[float] = []
    recall5_values: List[float] = []
    margin_values: List[float] = []

    for batch_index, sample_key in enumerate(sample_keys):
        rng = random.Random(seed_from_parts(seed, sample_key, "tcs"))
        valid_coords = [tuple(coord.tolist()) for coord in valid_token_mask[batch_index].nonzero(as_tuple=False)]
        anchor_coords = []
        for frame_index, patch_index in valid_coords:
            if (neighbor_indices[batch_index, frame_index, patch_index, :, 0] >= 0).any():
                anchor_coords.append((frame_index, patch_index))
        if not anchor_coords:
            continue
        rng.shuffle(anchor_coords)
        anchor_coords = anchor_coords[:max_anchors]
        valid_coord_set = set(valid_coords)

        for frame_index, patch_index in anchor_coords:
            positives: List[tuple[int, int]] = []
            for other_frame, other_patch in neighbor_indices[batch_index, frame_index, patch_index, :positive_topk].tolist():
                if other_frame < 0 or other_patch < 0:
                    continue
                if (other_frame, other_patch) not in valid_coord_set:
                    continue
                positives.append((other_frame, other_patch))
            if not positives:
                continue

            negative_pool = [
                coord
                for coord in valid_coords
                if coord != (frame_index, patch_index) and coord not in set(positives)
            ]
            if not negative_pool:
                continue
            rng.shuffle(negative_pool)
            negatives = negative_pool[:num_negatives]
            candidates = positives + negatives
            candidate_tensor = torch.stack(
                [normalized[batch_index, other_frame, other_patch] for other_frame, other_patch in candidates],
                dim=0,
            )
            anchor = normalized[batch_index, frame_index, patch_index]
            similarities = torch.einsum("d,nd->n", anchor, candidate_tensor)
            positive_count = len(positives)

            top1_index = int(similarities.argmax().item())
            recall1_values.append(1.0 if top1_index < positive_count else 0.0)

            topk = similarities.topk(k=min(5, similarities.shape[0]), dim=0).indices.tolist()
            recall5_values.append(1.0 if any(index < positive_count for index in topk) else 0.0)

            positive_similarity = float(similarities[:positive_count].mean().item())
            negative_similarity = float(similarities[positive_count:].max().item())
            margin_values.append(positive_similarity - negative_similarity)

    if not recall1_values:
        return {
            "corr_recall@1": 0.0,
            "corr_recall@5": 0.0,
            "tcs_margin": 0.0,
            "tcs_margin_norm": 0.0,
            "tcs": 0.0,
            "tcs_anchor_count": 0.0,
        }

    recall1 = sum(recall1_values) / len(recall1_values)
    recall5 = sum(recall5_values) / len(recall5_values)
    margin = sum(margin_values) / max(len(margin_values), 1)
    margin_norm = max(0.0, min(1.0, (margin + 1.0) / 2.0))
    return {
        "corr_recall@1": recall1,
        "corr_recall@5": recall5,
        "tcs_margin": margin,
        "tcs_margin_norm": margin_norm,
        "tcs": 0.5 * recall5 + 0.5 * margin_norm,
        "tcs_anchor_count": float(len(recall1_values)),
    }


def enrich_method_metrics(
    metrics: Dict[str, float],
    *,
    shuffled_metrics: Dict[str, float],
    current_frame_only_score: float | None = None,
) -> Dict[str, float]:
    enriched = dict(metrics)
    primary_mccr_cos = 0.5 * (
        enriched["mccr_cos_corr_tube"] + enriched["mccr_cos_frame_block"]
    )
    primary_mccr_l1 = 0.5 * (
        enriched["mccr_l1_corr_tube"] + enriched["mccr_l1_frame_block"]
    )
    shuffled_primary_mccr_cos = 0.5 * (
        shuffled_metrics["mccr_cos_corr_tube"] + shuffled_metrics["mccr_cos_frame_block"]
    )
    normal_score = 0.5 * primary_mccr_cos + 0.5 * enriched["tcs"]
    shuffled_score = 0.5 * shuffled_primary_mccr_cos + 0.5 * shuffled_metrics["tcs"]
    enriched["mccr_cos_primary"] = primary_mccr_cos
    enriched["mccr_l1_primary"] = primary_mccr_l1
    enriched["shuffled_mccr_cos_primary"] = shuffled_primary_mccr_cos
    enriched["shuffled_tcs"] = shuffled_metrics["tcs"]
    enriched["shuffled_corr_recall@5"] = shuffled_metrics["corr_recall@5"]
    enriched["shuffled_tcs_margin"] = shuffled_metrics["tcs_margin"]
    enriched["normal_score"] = normal_score
    enriched["shuffled_score"] = shuffled_score
    enriched["shuffle_gap"] = normal_score - shuffled_score
    enriched["cfo_gap"] = 0.0 if current_frame_only_score is None else normal_score - current_frame_only_score
    return enriched


METRIC_KEYS = [
    "mccr_cos_random_patch",
    "mccr_l1_random_patch",
    "mccr_cos_corr_tube",
    "mccr_l1_corr_tube",
    "mccr_cos_frame_block",
    "mccr_l1_frame_block",
    "corr_recall@1",
    "corr_recall@5",
    "tcs_margin",
    "tcs_margin_norm",
    "tcs",
    "tcs_anchor_count",
    "norm_rep",
    "cos_rep_g11",
    "mccr_cos_primary",
    "mccr_l1_primary",
    "shuffled_mccr_cos_primary",
    "shuffled_tcs",
    "shuffled_corr_recall@5",
    "shuffled_tcs_margin",
    "normal_score",
    "shuffled_score",
    "shuffle_gap",
    "cfo_gap",
]


class MetricAccumulator:
    def __init__(self, *, kind: str, checkpoint_path: str = "") -> None:
        self.kind = kind
        self.checkpoint_path = checkpoint_path
        self.weight = 0.0
        self.totals = {key: 0.0 for key in METRIC_KEYS}

    def update(self, metrics: Dict[str, float], *, batch_weight: float) -> None:
        self.weight += batch_weight
        for key in METRIC_KEYS:
            self.totals[key] += float(metrics.get(key, 0.0)) * batch_weight

    def finalize(self) -> Dict:
        denom = max(self.weight, 1.0)
        return {
            "kind": self.kind,
            "checkpoint_path": self.checkpoint_path,
            "num_windows": int(self.weight),
            "metrics": {key: value / denom for key, value in self.totals.items()},
        }

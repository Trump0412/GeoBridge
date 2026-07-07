"""Loss helpers for Stage 1 continuity pretraining."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = valid_mask.unsqueeze(-1).to(dtype=prediction.dtype)
    expanded_mask = expanded_mask.expand_as(prediction)
    denom = expanded_mask.sum().clamp_min(1.0)
    return (prediction - target).abs().mul(expanded_mask).sum() / denom


def masked_cosine_loss(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    masked = cosine * valid_mask.to(dtype=cosine.dtype)
    return masked.sum() / valid_mask.to(dtype=cosine.dtype).sum().clamp_min(1.0)


def geometry_reconstruction_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    total_l1 = predictions["g11"].new_tensor(0.0)
    total_cos = predictions["g11"].new_tensor(0.0)
    for name in ("g11", "g17", "g23"):
        l1 = masked_l1_loss(predictions[name], targets[name].detach(), valid_mask)
        cos = masked_cosine_loss(predictions[name], targets[name].detach(), valid_mask)
        losses[f"{name}_l1"] = l1
        losses[f"{name}_cos"] = cos
        total_l1 = total_l1 + l1
        total_cos = total_cos + cos
    losses["l1"] = total_l1 / 3.0
    losses["cos"] = total_cos / 3.0
    losses["total"] = losses["l1"] + losses["cos"]
    return losses

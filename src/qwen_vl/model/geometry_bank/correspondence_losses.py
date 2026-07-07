"""Losses and metrics for Stage 1 v2 correspondence-aware continuity pretraining."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def _safe_mask_sum(mask: torch.Tensor) -> torch.Tensor:
    return mask.to(dtype=torch.float32).sum().clamp_min(1.0)


def masked_l1_metric(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = valid_mask.unsqueeze(-1).to(dtype=prediction.dtype).expand_as(prediction)
    denom = expanded_mask.sum().clamp_min(1.0)
    return (prediction - target).abs().mul(expanded_mask).sum() / denom


def masked_cosine_similarity_metric(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    masked = cosine * valid_mask.to(dtype=cosine.dtype)
    return masked.sum() / _safe_mask_sum(valid_mask)


def reconstruction_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    *,
    layer_names: Sequence[str] | None = None,
    cosine_weight: float = 1.0,
    l1_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    active_layers = tuple(layer_names or predictions.keys())
    first_layer = active_layers[0]
    total_l1 = predictions[first_layer].new_tensor(0.0)
    total_cos = predictions[first_layer].new_tensor(0.0)
    metrics: Dict[str, torch.Tensor] = {}
    for name in active_layers:
        target = targets[name].detach()
        l1 = masked_l1_metric(predictions[name], target, valid_mask)
        cos_sim = masked_cosine_similarity_metric(predictions[name], target, valid_mask)
        metrics[f"{name}_l1"] = l1
        metrics[f"{name}_cos"] = cos_sim
        total_l1 = total_l1 + l1
        total_cos = total_cos + cos_sim
    metrics["l1"] = total_l1 / float(len(active_layers))
    metrics["cos"] = total_cos / float(len(active_layers))
    metrics["cos_loss"] = 1.0 - metrics["cos"]
    metrics["total"] = cosine_weight * metrics["cos_loss"] + l1_weight * metrics["l1"]
    return metrics


def pool_frame_tokens(hidden_states: torch.Tensor, valid_patch_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_patch_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denom


def lov_global_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float = 1.0,
    l1_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    l1 = (prediction - target).abs().mean()
    cos = F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean()
    cos_loss = 1.0 - cos
    return {
        "l1": l1,
        "cos": cos,
        "cos_loss": cos_loss,
        "total": cosine_weight * cos_loss + l1_weight * l1,
    }


def variance_loss(hidden_states: torch.Tensor, valid_mask: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    selected = hidden_states[valid_mask]
    if selected.shape[0] <= 1:
        return hidden_states.new_tensor(0.0)
    std = torch.sqrt(selected.float().var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(torch.tensor(gamma, device=hidden_states.device, dtype=std.dtype) - std).mean()


def corr_recall_from_attention(
    attention_weights: torch.Tensor,
    neighbor_valid_mask: torch.Tensor,
    *,
    positive_topk: int = 3,
) -> Dict[str, torch.Tensor]:
    if attention_weights is None or neighbor_valid_mask is None:
        zero = torch.tensor(0.0)
        return {"recall@1": zero, "recall@5": zero}
    valid_anchor = neighbor_valid_mask.any(dim=-1)
    if not valid_anchor.any():
        zero = attention_weights.new_tensor(0.0)
        return {"recall@1": zero, "recall@5": zero}

    attn = attention_weights[valid_anchor]
    valid = neighbor_valid_mask[valid_anchor]
    top1 = attn.argmax(dim=-1)
    recall1 = (top1 == 0).to(dtype=attn.dtype).mean()

    k = min(5, attn.shape[-1])
    topk = attn.topk(k=k, dim=-1).indices
    positive_limit = min(max(int(positive_topk), 1), attn.shape[-1])
    positive_mask = torch.zeros_like(attn, dtype=torch.bool)
    positive_mask[:, :positive_limit] = valid[:, :positive_limit]
    hits = positive_mask.gather(dim=-1, index=topk)
    recall5 = hits.any(dim=-1).to(dtype=attn.dtype).mean()
    return {"recall@1": recall1, "recall@5": recall5}


def attention_alignment_kl(
    attention_weights: torch.Tensor,
    neighbor_scores: torch.Tensor,
    neighbor_valid_mask: torch.Tensor,
    *,
    positive_topk: int = 3,
) -> torch.Tensor:
    if attention_weights is None:
        return neighbor_scores.new_tensor(0.0)
    valid_anchor = neighbor_valid_mask.any(dim=-1)
    if not valid_anchor.any():
        return neighbor_scores.new_tensor(0.0)

    attn = attention_weights[valid_anchor]
    scores = neighbor_scores[valid_anchor]
    valid = neighbor_valid_mask[valid_anchor]
    limit = min(max(int(positive_topk), 1), attn.shape[-1])
    attn_slice = attn[:, :limit]
    score_slice = scores[:, :limit]
    valid_slice = valid[:, :limit]
    score_slice = score_slice.masked_fill(~valid_slice, -1e4)
    pseudo = F.softmax(score_slice, dim=-1)
    pseudo = pseudo * valid_slice.to(dtype=pseudo.dtype)
    pseudo = pseudo / pseudo.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    attn_slice = attn_slice * valid_slice.to(dtype=attn_slice.dtype)
    attn_slice = attn_slice / attn_slice.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(attn_slice.clamp_min(1e-8).log(), pseudo, reduction="batchmean")


def multi_positive_infonce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    positive_weights: torch.Tensor,
    negative_pool: torch.Tensor,
    *,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if anchors.numel() == 0 or positives.numel() == 0 or negative_pool.numel() == 0:
        zero = anchors.new_tensor(0.0) if anchors.numel() else negative_pool.new_tensor(0.0)
        return zero, zero

    anchors = F.normalize(anchors.float(), dim=-1)
    positives = F.normalize(positives.float(), dim=-1)
    negative_pool = F.normalize(negative_pool.float(), dim=-1)

    pos_logits = torch.einsum("ad,akd->ak", anchors, positives) / temperature
    neg_logits = torch.einsum("ad,nd->an", anchors, negative_pool) / temperature

    positive_weights = positive_weights.float()
    positive_weights = positive_weights / positive_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    pos_logsumexp = torch.logsumexp(pos_logits + positive_weights.clamp_min(1e-6).log(), dim=-1)
    neg_logsumexp = torch.logsumexp(neg_logits, dim=-1)
    loss = -(pos_logsumexp - torch.logaddexp(pos_logsumexp, neg_logsumexp)).mean()

    all_logits = torch.cat([pos_logits, neg_logits], dim=-1)
    accuracy = (all_logits.argmax(dim=-1) < pos_logits.shape[-1]).to(dtype=anchors.dtype).mean()
    return loss, accuracy

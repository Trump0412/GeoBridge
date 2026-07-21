"""Same-position local geometry router for SpatialFit Stage 2."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalLevelRouter(nn.Module):
    """Select among same-position local geometry levels g11/g17/g23."""

    def __init__(
        self,
        hidden_size: int,
        d_geom: int,
        *,
        topk: int = 2,
        normalize_query: bool = True,
        normalize_bank: bool = True,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.topk = int(topk)
        self.temperature = max(float(temperature), 1e-6)
        self.normalize_query = bool(normalize_query)
        self.normalize_bank = bool(normalize_bank)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, d_geom, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_bank: torch.Tensor,
        candidate_logit_bias: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hidden_states.numel() == 0:
            empty_hist = local_bank.new_zeros((local_bank.shape[1],))
            return hidden_states.new_zeros((0, local_bank.shape[-1])), {
                "selection_histogram": empty_hist,
                "topk_idx": local_bank.new_empty((0, 0), dtype=torch.long),
                "topk_weight": local_bank.new_empty((0, 0)),
                "raw_candidate_norm": empty_hist.clone(),
                "router_probs": local_bank.new_empty((0, local_bank.shape[1])),
                "entropy": local_bank.new_empty((0,)),
                "max_prob": local_bank.new_empty((0,)),
            }

        router_input = self.input_norm(hidden_states.to(dtype=self.query_proj.weight.dtype))
        raw_bank = local_bank.to(dtype=self.query_proj.weight.dtype)
        query = self.query_proj(router_input)
        score_query = F.normalize(query, dim=-1) if self.normalize_query else query
        score_bank = F.normalize(raw_bank, dim=-1) if self.normalize_bank else raw_bank
        score = torch.einsum("pd,pcd->pc", score_query, score_bank) / self.temperature
        if candidate_logit_bias is not None:
            bias = candidate_logit_bias.to(device=score.device, dtype=score.dtype)
            if bias.dim() == 1:
                bias = bias.unsqueeze(0)
            score = score + bias

        router_probs = F.softmax(score, dim=-1)
        entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1)
        if raw_bank.shape[1] > 1:
            entropy = entropy / math.log(raw_bank.shape[1])
        max_prob = router_probs.max(dim=-1).values

        topk = min(self.topk, raw_bank.shape[1])
        topk_score, topk_idx = torch.topk(score, k=topk, dim=-1)
        topk_weight = F.softmax(topk_score, dim=-1)
        gathered_bank = torch.gather(
            raw_bank,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, raw_bank.shape[-1]),
        )
        selected_memory = (gathered_bank * topk_weight.unsqueeze(-1)).sum(dim=1)
        selection_histogram = torch.bincount(topk_idx.reshape(-1), minlength=raw_bank.shape[1]).to(raw_bank.dtype)
        raw_candidate_norm = raw_bank.norm(dim=-1).mean(dim=0)
        return selected_memory, {
            "selection_histogram": selection_histogram,
            "topk_idx": topk_idx,
            "topk_weight": topk_weight,
            "raw_candidate_norm": raw_candidate_norm,
            "router_probs": router_probs,
            "entropy": entropy,
            "max_prob": max_prob,
        }

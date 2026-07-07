"""Top-k routing over geometry-bank candidates."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BankRouter(nn.Module):
    """Route each visual token to Top-k geometry candidates."""

    def __init__(
        self,
        hidden_size: int,
        d_geom: int,
        num_layers: int,
        topk: int = 2,
        use_layer_embedding: bool = True,
        normalize_query: bool = False,
        normalize_bank: bool = False,
        temperature: float = 1.0,
        candidate_dropout_enabled: bool = False,
        g11_drop_prob: float = 0.0,
        g17_drop_prob: float = 0.0,
        g23_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.d_geom = int(d_geom)
        self.topk = int(topk)
        self.use_layer_embedding = bool(use_layer_embedding)
        self.normalize_query = bool(normalize_query)
        self.normalize_bank = bool(normalize_bank)
        self.temperature = max(float(temperature), 1e-6)
        self.candidate_dropout_enabled = bool(candidate_dropout_enabled)
        self.dropout_probs = (float(g11_drop_prob), float(g17_drop_prob), float(g23_drop_prob), 0.0)

        self.input_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, d_geom, bias=False)
        self.layer_embeddings = nn.Embedding(num_layers, hidden_size) if use_layer_embedding else None

    def _candidate_keep_mask(self, frame_bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keep_mask = torch.ones(frame_bank.shape[:2], device=frame_bank.device, dtype=torch.bool)
        if (not self.training) or (not self.candidate_dropout_enabled) or frame_bank.shape[1] < 4:
            return keep_mask, keep_mask.new_zeros(keep_mask.shape, dtype=torch.bool)

        dropped_mask = keep_mask.new_zeros(keep_mask.shape, dtype=torch.bool)
        for candidate_idx, drop_prob in enumerate(self.dropout_probs[: min(3, frame_bank.shape[1])]):
            if drop_prob <= 0.0:
                continue
            candidate_drop = torch.rand(frame_bank.shape[0], device=frame_bank.device) < drop_prob
            keep_mask[:, candidate_idx] = ~candidate_drop
            dropped_mask[:, candidate_idx] = candidate_drop

        valid_after_dropout = keep_mask.sum(dim=-1)
        fallback_mask = valid_after_dropout < self.topk
        if fallback_mask.any():
            keep_mask[fallback_mask] = True
            dropped_mask[fallback_mask] = False

        return keep_mask, dropped_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_bank: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hidden_states.numel() == 0:
            empty_hist = frame_bank.new_zeros((frame_bank.shape[1],))
            return hidden_states.new_zeros((0, self.d_geom)), {
                "topk_idx": frame_bank.new_empty((0, 0), dtype=torch.long),
                "topk_weight": frame_bank.new_empty((0, 0)),
                "selection_histogram": empty_hist,
                "drop_histogram": empty_hist.clone(),
                "drop_ratio": empty_hist.clone(),
                "raw_candidate_norm": empty_hist.clone(),
            }

        compute_dtype = self.input_norm.weight.dtype
        router_input = self.input_norm(hidden_states.to(dtype=compute_dtype))
        if self.layer_embeddings is not None:
            layer_index = min(int(layer_id), self.layer_embeddings.num_embeddings - 1)
            layer_embed = self.layer_embeddings(
                torch.full((1,), layer_index, dtype=torch.long, device=hidden_states.device)
            )
            router_input = router_input + layer_embed.expand_as(router_input)

        raw_bank = frame_bank.to(dtype=self.query_proj.weight.dtype)
        query = self.query_proj(router_input)
        query_for_score = F.normalize(query, dim=-1) if self.normalize_query else query
        bank_for_score = F.normalize(raw_bank, dim=-1) if self.normalize_bank else raw_bank
        score = torch.einsum("pd,pcd->pc", query_for_score, bank_for_score) / self.temperature

        keep_mask, dropped_mask = self._candidate_keep_mask(raw_bank)
        if dropped_mask.any():
            score = score.masked_fill(~keep_mask, torch.finfo(score.dtype).min)

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
        drop_histogram = dropped_mask.to(raw_bank.dtype).sum(dim=0)
        drop_ratio = drop_histogram / max(raw_bank.shape[0], 1)
        raw_candidate_norm = raw_bank.norm(dim=-1).mean(dim=0)

        return selected_memory, {
            "topk_idx": topk_idx,
            "topk_weight": topk_weight,
            "selection_histogram": selection_histogram,
            "drop_histogram": drop_histogram,
            "drop_ratio": drop_ratio,
            "raw_candidate_norm": raw_candidate_norm,
        }

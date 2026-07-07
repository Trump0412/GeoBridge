"""Geometry decoder heads for Stage 1 continuity pretraining."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn


class _DecoderHead(nn.Module):
    def __init__(self, d_geom: int, hidden_ratio: float = 2.0):
        super().__init__()
        hidden_dim = max(int(d_geom * hidden_ratio), d_geom)
        self.net = nn.Sequential(
            nn.LayerNorm(d_geom),
            nn.Linear(d_geom, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_geom),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        compute_dtype = next(self.parameters()).dtype
        return self.net(hidden_states.to(dtype=compute_dtype))


class GeometryDecoder(nn.Module):
    """Decode continuity features back into three local geometry branches."""

    def __init__(self, d_geom: int, hidden_ratio: float = 2.0, layer_names: Sequence[str] = ("g11", "g17", "g23")):
        super().__init__()
        self.layer_names = tuple(layer_names)
        self.heads = nn.ModuleDict(
            {
                layer_name: _DecoderHead(d_geom=d_geom, hidden_ratio=hidden_ratio)
                for layer_name in self.layer_names
            }
        )

    def forward(self, continuity: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(continuity) for name, head in self.heads.items()}

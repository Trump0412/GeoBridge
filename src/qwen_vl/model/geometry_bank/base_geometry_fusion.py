"""Base fusion for local multi-layer geometry tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class BaseGeometryFusion(nn.Module):
    """Fuse g11/g17/g23 into a shared geometry context token z."""

    def __init__(self, d_geom: int, hidden_ratio: float = 2.0, num_layers: int = 3):
        super().__init__()
        self.num_layers = int(num_layers)
        hidden_dim = max(int(d_geom * hidden_ratio), d_geom)
        self.fuser = nn.Sequential(
            nn.LayerNorm(d_geom * self.num_layers),
            nn.Linear(d_geom * self.num_layers, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_geom),
        )

    def forward(self, *layers: torch.Tensor) -> torch.Tensor:
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = tuple(layers[0])
        if len(layers) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} geometry layers, got {len(layers)}")
        stacked = torch.cat(list(layers), dim=-1)
        compute_dtype = next(self.parameters()).dtype
        return self.fuser(stacked.to(dtype=compute_dtype))

"""Projection heads for geometry-bank layers."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class _ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        compute_dtype = self.norm.weight.dtype
        normalized = self.norm(features.to(dtype=compute_dtype))
        return self.proj(normalized)


class GeoProjector(nn.Module):
    """Independent per-layer projection heads into the bank hidden space."""

    def __init__(self, input_dims: Dict[str, int], d_geom: int):
        super().__init__()
        self.projectors = nn.ModuleDict(
            {name: _ProjectionHead(input_dim, d_geom) for name, input_dim in input_dims.items()}
        )

    def forward(self, layer_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for name, features in layer_tokens.items():
            outputs[name.replace("_raw", "")] = self.projectors[name](features)
        return outputs

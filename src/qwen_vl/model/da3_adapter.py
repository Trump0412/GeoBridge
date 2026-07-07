"""DA3 adapter helpers shared by SGF and MSGF variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DA3AdapterOutput:
    geo_embeds: torch.Tensor
    image_grid_thw: torch.Tensor
    geo_frame_splits: Optional[list[int]] = None
    frame_ids: Optional[torch.Tensor] = None
    pose_embeds: Optional[torch.Tensor] = None
    conf_embeds: Optional[torch.Tensor] = None


class DA3Projector(nn.Module):
    """Project DA3 features into the decoder hidden space."""

    def __init__(self, da3_dim: int, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(da3_dim)
        self.proj = nn.Sequential(
            nn.Linear(da3_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, da3_feats: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(da3_feats))

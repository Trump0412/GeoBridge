"""Unified geometry bank containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..msgf_utils import FrameLayout


BANK_G11 = 0
BANK_G17 = 1
BANK_G23 = 2
BANK_CONT = 3


@dataclass
class GeometryBankOutput:
    """Unified geometry bank with aligned frame slices."""

    bank: torch.Tensor
    frame_layout: FrameLayout
    valid_mask: torch.Tensor
    stats: Dict[str, float]
    use_continuity: bool
    saliency: Optional[torch.Tensor] = None

    def get_frame_bank(self, frame_idx: int) -> torch.Tensor:
        token_count = self.frame_layout.token_counts[frame_idx]
        return self.bank[frame_idx, :token_count]

    def get_frame_saliency(self, frame_idx: int) -> Optional[torch.Tensor]:
        if self.saliency is None:
            return None
        token_count = self.frame_layout.token_counts[frame_idx]
        return self.saliency[frame_idx, :token_count]


class GeometryBank(nn.Module):
    """Assemble the unified geometry bank tensor."""

    def __init__(self, use_continuity: bool = True):
        super().__init__()
        self.use_continuity = bool(use_continuity)

    def _compute_stats(self, g11: torch.Tensor, g17: torch.Tensor, g23: torch.Tensor, continuity: torch.Tensor) -> Dict[str, float]:
        return {
            "g11_norm": float(g11.norm(dim=-1).mean().detach().item()),
            "g17_norm": float(g17.norm(dim=-1).mean().detach().item()),
            "g23_norm": float(g23.norm(dim=-1).mean().detach().item()),
            "cont_norm": float(continuity.norm(dim=-1).mean().detach().item()),
        }

    def forward(
        self,
        g11: torch.Tensor,
        g17: torch.Tensor,
        g23: torch.Tensor,
        continuity: torch.Tensor,
        frame_layout: FrameLayout,
        saliency: Optional[torch.Tensor] = None,
    ) -> GeometryBankOutput:
        if not self.use_continuity:
            continuity = torch.zeros_like(g11)
        bank = torch.stack([g11, g17, g23, continuity], dim=2)
        valid_mask = torch.ones(bank.shape[0], bank.shape[1], dtype=torch.bool, device=bank.device)
        return GeometryBankOutput(
            bank=bank,
            frame_layout=frame_layout,
            valid_mask=valid_mask,
            stats=self._compute_stats(g11, g17, g23, continuity),
            use_continuity=self.use_continuity,
            saliency=saliency,
        )

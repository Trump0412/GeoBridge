"""Residual write-back for geometry-bank memory."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BankFusionBlock(nn.Module):
    """Gated residual fusion from selected geometry memory."""

    def __init__(self, hidden_size: int, d_geom: int, gate_mode: str = "scalar"):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.gate_mode = gate_mode

        self.output_proj = nn.Linear(d_geom, hidden_size, bias=False)
        self.gate_norm = nn.LayerNorm(hidden_size)
        if gate_mode == "scalar":
            self.gate_proj = nn.Linear(hidden_size, 1)
        elif gate_mode == "channel":
            self.gate_proj = nn.Linear(hidden_size, hidden_size)
        elif gate_mode == "none":
            self.gate_proj = None
        else:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

    def forward(self, hidden_states: torch.Tensor, selected_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        compute_dtype = self.gate_norm.weight.dtype
        hidden_states_for_gate = hidden_states.to(dtype=compute_dtype)
        selected_memory = selected_memory.to(dtype=self.output_proj.weight.dtype)
        delta = self.output_proj(selected_memory)
        if self.gate_proj is None:
            gate = torch.ones_like(delta)
            return delta.to(dtype=hidden_states.dtype), gate.to(dtype=hidden_states.dtype)

        gate = torch.sigmoid(self.gate_proj(self.gate_norm(hidden_states_for_gate)))
        fused = gate * delta
        return fused.to(dtype=hidden_states.dtype), gate.to(dtype=hidden_states.dtype)

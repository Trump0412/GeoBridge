"""Competitive none/local/continuity gate for GeoBridge Stage 2."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompetitiveBridgeGate(nn.Module):
    """Three-way softmax gate over none/local/continuity residuals."""

    def __init__(
        self,
        hidden_size: int,
        d_geom: int,
        *,
        use_saliency_prior: bool = True,
        layer_scale_init: float = 0.05,
        gate_none_bias: float = 0.0,
        gate_local_bias: float = 0.4,
        gate_cont_bias: float = 0.6,
        use_gate_bias_init: bool = True,
    ):
        super().__init__()
        self.use_saliency_prior = bool(use_saliency_prior)
        fused_dim = hidden_size + 2 * d_geom
        self.input_norm = nn.LayerNorm(hidden_size)
        self.local_norm = nn.LayerNorm(d_geom)
        self.cont_norm = nn.LayerNorm(d_geom)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )
        self.saliency_proj = nn.Linear(1, 3, bias=False) if self.use_saliency_prior else None
        # FSDP does not support scalar parameters; keep this as a one-element
        # vector while preserving scalar-style broadcasting in forward.
        self.layer_scale = nn.Parameter(torch.full((1,), float(layer_scale_init)))
        if use_gate_bias_init:
            final_linear = self.mlp[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError("CompetitiveBridgeGate expects the final MLP layer to be nn.Linear.")
            if final_linear.bias is None:
                raise ValueError("CompetitiveBridgeGate final MLP layer must have bias for gate bias init.")
            with torch.no_grad():
                final_linear.bias.copy_(
                    torch.tensor(
                        [float(gate_none_bias), float(gate_local_bias), float(gate_cont_bias)],
                        dtype=final_linear.bias.dtype,
                        device=final_linear.bias.device,
                    )
                )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        layer_scale_key = prefix + "layer_scale"
        layer_scale = state_dict.get(layer_scale_key)
        if isinstance(layer_scale, torch.Tensor) and layer_scale.ndim == 0:
            state_dict[layer_scale_key] = layer_scale.reshape(1)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_memory: torch.Tensor,
        cont_memory: torch.Tensor,
        local_delta: torch.Tensor,
        cont_delta: torch.Tensor,
        saliency_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden_feat = F.layer_norm(
            hidden_states.float(),
            self.input_norm.normalized_shape,
            self.input_norm.weight.float() if self.input_norm.weight is not None else None,
            self.input_norm.bias.float() if self.input_norm.bias is not None else None,
            self.input_norm.eps,
        )
        local_feat = F.layer_norm(
            local_memory.float(),
            self.local_norm.normalized_shape,
            self.local_norm.weight.float() if self.local_norm.weight is not None else None,
            self.local_norm.bias.float() if self.local_norm.bias is not None else None,
            self.local_norm.eps,
        )
        cont_feat = F.layer_norm(
            cont_memory.float(),
            self.cont_norm.normalized_shape,
            self.cont_norm.weight.float() if self.cont_norm.weight is not None else None,
            self.cont_norm.bias.float() if self.cont_norm.bias is not None else None,
            self.cont_norm.eps,
        )
        mlp_dtype = self.mlp[0].weight.dtype
        logits = self.mlp(torch.cat([hidden_feat, local_feat, cont_feat], dim=-1).to(dtype=mlp_dtype)).float()
        if self.saliency_proj is not None and saliency_prior is not None:
            saliency_input = saliency_prior.float().unsqueeze(-1).to(dtype=self.saliency_proj.weight.dtype)
            logits = logits + self.saliency_proj(saliency_input).float()
        probs = F.softmax(logits, dim=-1)
        mixed = probs[:, 1:2] * local_delta.float() + probs[:, 2:3] * cont_delta.float()
        layer_scale_tanh = torch.tanh(self.layer_scale)
        mixed = layer_scale_tanh * mixed
        decision = torch.argmax(probs, dim=-1)
        gate_histogram = torch.bincount(decision, minlength=3).to(dtype=probs.dtype, device=probs.device)
        local_delta_norm_mean = local_delta.float().norm(dim=-1).mean() if local_delta.numel() > 0 else probs.new_tensor(0.0)
        cont_delta_norm_mean = cont_delta.float().norm(dim=-1).mean() if cont_delta.numel() > 0 else probs.new_tensor(0.0)
        mixed_norm_mean = mixed.norm(dim=-1).mean() if mixed.numel() > 0 else probs.new_tensor(0.0)
        return mixed.to(dtype=hidden_states.dtype), {
            "gate_probs": probs,
            "gate_histogram": gate_histogram,
            "decision": decision,
            "layer_scale_tanh": layer_scale_tanh.detach(),
            "mixed_norm_mean": mixed_norm_mean.detach(),
            "local_delta_norm_mean": local_delta_norm_mean.detach(),
            "cont_delta_norm_mean": cont_delta_norm_mean.detach(),
        }

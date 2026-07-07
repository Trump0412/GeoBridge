"""Continuity token builder for the ZenView geometry bank."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuityBuilder(nn.Module):
    """Build a continuity token from multi-level local geometry."""

    def __init__(
        self,
        d_geom: int,
        radius_t: int = 1,
        use_spatial_neighbors: bool = False,
        mlp_hidden_ratio: float = 2.0,
        attention_heads: int = 4,
        corr_score_beta: float = 1.0,
        time_bias_init: float = -0.10,
    ):
        super().__init__()
        hidden_dim = int(d_geom * mlp_hidden_ratio)
        self.radius_t = int(radius_t)
        self.use_spatial_neighbors = bool(use_spatial_neighbors)
        self.attention_heads = int(attention_heads)
        self.d_geom = int(d_geom)

        self.layer_fuser = nn.Sequential(
            nn.LayerNorm(d_geom * 3),
            nn.Linear(d_geom * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_geom),
        )
        self.q_proj = nn.Linear(d_geom, d_geom, bias=False)
        self.k_proj = nn.Linear(d_geom, d_geom, bias=False)
        self.v_proj = nn.Linear(d_geom, d_geom, bias=False)
        self.output_norm = nn.LayerNorm(d_geom)
        self.corr_score_beta = nn.Parameter(torch.full((1,), float(corr_score_beta)))
        self.time_bias_scale = nn.Parameter(torch.full((1,), float(time_bias_init)))
        self.edge_activation_scale = nn.Parameter(torch.ones(1))

    def _project_qkv(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_float = z.float()
        q = F.linear(z_float, self.q_proj.weight.float(), None)
        k = F.linear(z_float, self.k_proj.weight.float(), None)
        v = F.linear(z_float, self.v_proj.weight.float(), None)
        return q, k, v

    def _same_position_attention(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeeze_batch = True
        batch_size, num_frames, _, hidden_dim = z.shape
        q, k, v = self._project_qkv(z)
        outputs = []
        for frame_idx in range(num_frames):
            start = max(0, frame_idx - self.radius_t)
            end = min(num_frames, frame_idx + self.radius_t + 1)
            scores = torch.einsum("bpd,bspd->bsp", q[:, frame_idx], k[:, start:end]) / math.sqrt(hidden_dim)
            scores = scores.clamp(min=-60.0, max=60.0)
            attn = F.softmax(scores, dim=1)
            outputs.append(torch.einsum("bsp,bspd->bpd", attn, v[:, start:end]))
        output = torch.stack(outputs, dim=1)
        if squeeze_batch:
            return output.squeeze(0)
        return output

    def _current_frame_only(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        residual_dtype = self.output_norm.weight.dtype
        output = self.output_norm(z.to(dtype=residual_dtype))
        return output, {
            "attention_weights": None,
            "neighbor_valid_mask": None,
        }

    def _spatial_temporal_attention(self, z: torch.Tensor, frame_shape: Tuple[int, int]) -> torch.Tensor:
        squeeze_batch = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeeze_batch = True

        if z.shape[0] > 1:
            outputs = []
            for batch_idx in range(z.shape[0]):
                sample_shape = frame_shape
                if isinstance(frame_shape, Sequence) and frame_shape and isinstance(frame_shape[0], tuple):
                    sample_shape = frame_shape[batch_idx][0]
                outputs.append(self._spatial_temporal_attention(z[batch_idx], sample_shape))
            output = torch.stack(outputs, dim=0)
            if squeeze_batch:
                return output.squeeze(0)
            return output

        height, width = frame_shape
        if height * width != z.shape[2]:
            return self._same_position_attention(z)

        sample_z = z[0]
        q, k_proj, v_proj = self._project_qkv(sample_z)
        q = q
        k = k_proj.reshape(sample_z.shape[0], height, width, sample_z.shape[-1]).permute(0, 3, 1, 2)
        v = v_proj.reshape(sample_z.shape[0], height, width, sample_z.shape[-1]).permute(0, 3, 1, 2)

        k_neighbors = F.unfold(k, kernel_size=3, padding=1).transpose(1, 2).reshape(sample_z.shape[0], sample_z.shape[1], 9, sample_z.shape[-1])
        v_neighbors = F.unfold(v, kernel_size=3, padding=1).transpose(1, 2).reshape(sample_z.shape[0], sample_z.shape[1], 9, sample_z.shape[-1])

        outputs = []
        scale = math.sqrt(sample_z.shape[-1])
        for frame_idx in range(sample_z.shape[0]):
            start = max(0, frame_idx - self.radius_t)
            end = min(sample_z.shape[0], frame_idx + self.radius_t + 1)
            scores = torch.einsum("pd,spnd->spn", q[frame_idx], k_neighbors[start:end]) / scale
            scores = scores.clamp(min=-60.0, max=60.0)
            scores = scores.permute(1, 0, 2).reshape(sample_z.shape[1], -1)
            attn = F.softmax(scores, dim=-1).reshape(sample_z.shape[1], end - start, 9)
            values = v_neighbors[start:end].permute(1, 0, 2, 3)
            outputs.append((attn.unsqueeze(-1) * values).sum(dim=(1, 2)))
        output = torch.stack(outputs, dim=0)
        if squeeze_batch:
            return output
        return output.unsqueeze(0)

    def _corr_graph_attention(
        self,
        z: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_scores: torch.Tensor,
        edge_activation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        squeeze_batch = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            neighbor_indices = neighbor_indices.unsqueeze(0)
            neighbor_scores = neighbor_scores.unsqueeze(0)
            squeeze_batch = True

        batch_size, num_frames, num_patches, hidden_dim = z.shape
        q, k, v = self._project_qkv(z)
        flat_k = k.reshape(batch_size, num_frames * num_patches, hidden_dim)
        flat_v = v.reshape(batch_size, num_frames * num_patches, hidden_dim)

        neighbor_indices = neighbor_indices.long()
        neighbor_scores = neighbor_scores.float()
        valid_mask = (neighbor_indices[..., 0] >= 0) & (neighbor_indices[..., 1] >= 0)
        neighbor_frames = neighbor_indices[..., 0].clamp(min=0)
        neighbor_patches = neighbor_indices[..., 1].clamp(min=0)
        flat_indices = neighbor_frames * num_patches + neighbor_patches
        batch_index = torch.arange(batch_size, device=z.device).view(batch_size, 1, 1, 1).expand_as(flat_indices)
        gathered_k = flat_k[batch_index, flat_indices]
        gathered_v = flat_v[batch_index, flat_indices]

        scores = (q.unsqueeze(-2) * gathered_k).sum(dim=-1) / math.sqrt(hidden_dim)
        scores = scores + self.corr_score_beta.to(dtype=scores.dtype) * neighbor_scores.to(dtype=scores.dtype)
        if edge_activation is not None:
            scores = scores + self.edge_activation_scale.to(dtype=scores.dtype) * edge_activation.to(dtype=scores.dtype)
        time_positions = torch.arange(num_frames, device=z.device).view(1, num_frames, 1, 1)
        time_bias = (time_positions - neighbor_frames).abs().to(dtype=scores.dtype)
        scores = scores + self.time_bias_scale.to(dtype=scores.dtype) * time_bias
        scores = scores.masked_fill(~valid_mask, -1e4)
        no_neighbor = ~valid_mask.any(dim=-1)
        scores = scores.masked_fill(no_neighbor.unsqueeze(-1), 0.0)
        attn = F.softmax(scores, dim=-1)
        attn = attn * valid_mask.to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = (attn.unsqueeze(-1) * gathered_v).sum(dim=-2)
        delta = delta.masked_fill(no_neighbor.unsqueeze(-1), 0.0)

        residual_dtype = self.output_norm.weight.dtype
        output = self.output_norm(z.to(dtype=residual_dtype) + delta.to(dtype=residual_dtype))
        aux = {
            "attention_weights": attn.squeeze(0) if squeeze_batch else attn,
            "neighbor_valid_mask": valid_mask.squeeze(0) if squeeze_batch else valid_mask,
            "edge_activation": edge_activation.squeeze(0) if (edge_activation is not None and squeeze_batch) else edge_activation,
        }
        if squeeze_batch:
            return output.squeeze(0), aux
        return output, aux

    def forward_from_fused(
        self,
        z: torch.Tensor,
        frame_shapes: Sequence[Tuple[int, int]] | None = None,
        *,
        mode: str | None = None,
        corr_neighbor_indices: torch.Tensor | None = None,
        corr_neighbor_scores: torch.Tensor | None = None,
        corr_edge_activation: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if mode == "current_frame_only":
            output, aux = self._current_frame_only(z)
        elif mode == "corr_graph":
            if corr_neighbor_indices is None or corr_neighbor_scores is None:
                raise ValueError("corr_graph mode requires corr_neighbor_indices and corr_neighbor_scores")
            output, aux = self._corr_graph_attention(z, corr_neighbor_indices, corr_neighbor_scores, corr_edge_activation)
        elif self.use_spatial_neighbors and frame_shapes and mode != "same_position":
            continuity_delta = self._spatial_temporal_attention(z, frame_shapes[0])
            residual_dtype = self.output_norm.weight.dtype
            continuity_input = z.to(dtype=residual_dtype) + continuity_delta.to(dtype=residual_dtype)
            output = self.output_norm(continuity_input)
            aux = {
                "attention_weights": None,
                "neighbor_valid_mask": None,
            }
        else:
            continuity_delta = self._same_position_attention(z)
            residual_dtype = self.output_norm.weight.dtype
            continuity_input = z.to(dtype=residual_dtype) + continuity_delta.to(dtype=residual_dtype)
            output = self.output_norm(continuity_input)
            aux = {
                "attention_weights": None,
                "neighbor_valid_mask": None,
            }
        if return_aux:
            return output, aux
        return output

    def forward(
        self,
        g11: torch.Tensor,
        g17: torch.Tensor | None = None,
        g23: torch.Tensor | None = None,
        frame_shapes: Sequence[Tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        if g17 is None or g23 is None:
            return self.forward_from_fused(g11, frame_shapes=frame_shapes)

        compute_dtype = self.q_proj.weight.dtype
        stacked = torch.cat([g11, g17, g23], dim=-1).to(dtype=compute_dtype)
        z = self.layer_fuser(stacked)
        return self.forward_from_fused(z, frame_shapes=frame_shapes)

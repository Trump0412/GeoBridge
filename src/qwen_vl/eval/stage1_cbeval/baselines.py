"""Baseline and FCP utilities for Stage1 CBEval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from qwen_vl.eval.stage1_cbeval.masks import apply_frame_permutations, restore_frame_permutations
from qwen_vl.model.geometry_bank import (
    ActivatedCorrespondenceGraph,
    ContinuityBuilder,
    ContinuityUtilitySelector,
    GeometryDecoder,
)
from qwen_vl.train.stage1_compact_cache import dequantize_projected_tokenwise


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = dict(batch)
    moved["valid_frame_mask"] = batch["valid_frame_mask"].to(device, non_blocking=device.type == "cuda")
    moved["valid_patch_mask"] = batch["valid_patch_mask"].to(device, non_blocking=device.type == "cuda")
    moved["cached_features"] = {
        key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
        for key, value in batch["cached_features"].items()
    }
    moved["corr_graph"] = {
        key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
        for key, value in batch["corr_graph"].items()
    }
    return moved


def load_projected_g11(batch: dict) -> torch.Tensor:
    cached = batch["cached_features"]
    feature_space = cached.get("feature_space")
    if feature_space == "projected":
        return cached["g11"].to(dtype=torch.float32)
    if feature_space == "projected_quantized":
        return dequantize_projected_tokenwise(cached["g11_q"], cached["g11_scale"]).to(dtype=torch.float32)
    raise ValueError(f"Stage1 CBEval currently expects projected g11 cache, got feature_space={feature_space}")


def build_same_position_neighbors(valid_token_mask: torch.Tensor, *, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_frames, num_patches = valid_token_mask.shape
    max_neighbors = max(int(radius) * 2, 1)
    neighbor_indices = torch.full(
        (batch_size, num_frames, num_patches, max_neighbors, 2),
        -1,
        dtype=torch.long,
        device=valid_token_mask.device,
    )
    neighbor_scores = torch.zeros(
        (batch_size, num_frames, num_patches, max_neighbors),
        dtype=torch.float32,
        device=valid_token_mask.device,
    )
    for batch_index in range(batch_size):
        for frame_index in range(num_frames):
            for patch_index in range(num_patches):
                if not bool(valid_token_mask[batch_index, frame_index, patch_index]):
                    continue
                write_index = 0
                for other_frame in range(max(0, frame_index - radius), min(num_frames, frame_index + radius + 1)):
                    if other_frame == frame_index:
                        continue
                    if write_index >= max_neighbors:
                        break
                    if bool(valid_token_mask[batch_index, other_frame, patch_index]):
                        neighbor_indices[batch_index, frame_index, patch_index, write_index] = torch.tensor(
                            [other_frame, patch_index],
                            dtype=torch.long,
                            device=valid_token_mask.device,
                        )
                        write_index += 1
    return neighbor_indices, neighbor_scores


def aggregate_neighbors(
    source: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    *,
    support_mask: torch.Tensor,
    topk: int,
    fill_source_if_missing: bool,
) -> torch.Tensor:
    batch_size, num_frames, num_patches, hidden_dim = source.shape
    take_k = min(topk, neighbor_indices.shape[-2])
    current_neighbors = neighbor_indices[..., :take_k, :].long()
    current_scores = neighbor_scores[..., :take_k].float()
    valid_neighbor = (current_neighbors[..., 0] >= 0) & (current_neighbors[..., 1] >= 0)
    frames = current_neighbors[..., 0].clamp(min=0)
    patches = current_neighbors[..., 1].clamp(min=0)
    flat_source = source.reshape(batch_size, num_frames * num_patches, hidden_dim)
    flat_support = support_mask.reshape(batch_size, num_frames * num_patches)
    flat_indices = frames * num_patches + patches
    batch_index = torch.arange(batch_size, device=source.device).view(batch_size, 1, 1, 1).expand_as(flat_indices)
    gathered = flat_source[batch_index, flat_indices]
    neighbor_supported = flat_support[batch_index, flat_indices]
    valid_neighbor = valid_neighbor & neighbor_supported

    weights = current_scores.masked_fill(~valid_neighbor, -1e4)
    no_neighbor = ~valid_neighbor.any(dim=-1)
    weights = weights.masked_fill(no_neighbor.unsqueeze(-1), 0.0)
    attention = torch.softmax(weights, dim=-1)
    attention = attention * valid_neighbor.to(dtype=attention.dtype)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    aggregated = (attention.unsqueeze(-1) * gathered).sum(dim=-2)
    if fill_source_if_missing:
        aggregated = torch.where(no_neighbor.unsqueeze(-1), source, aggregated)
    else:
        aggregated = aggregated.masked_fill(no_neighbor.unsqueeze(-1), 0.0)
    return aggregated


def current_frame_only_context(source: torch.Tensor) -> torch.Tensor:
    return source


def current_frame_only_recovery(
    source: torch.Tensor,
    valid_token_mask: torch.Tensor,
    recovery_mask: torch.Tensor,
) -> torch.Tensor:
    visible_mask = valid_token_mask & ~recovery_mask
    weights = visible_mask.to(dtype=source.dtype).unsqueeze(-1)
    frame_mean = (source * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    prediction = source.clone()
    expanded = frame_mean.unsqueeze(2).expand_as(source)
    prediction[recovery_mask] = expanded[recovery_mask]
    return prediction


def same_position_context(
    source: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    radius: int,
    permutations: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    working_source = apply_frame_permutations(source, permutations) if permutations is not None else source
    working_valid = (
        apply_frame_permutations(valid_token_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else valid_token_mask
    )
    neighbors, scores = build_same_position_neighbors(working_valid, radius=radius)
    output = aggregate_neighbors(
        working_source,
        neighbors,
        scores,
        support_mask=working_valid,
        topk=max(radius * 2, 1),
        fill_source_if_missing=True,
    )
    return restore_frame_permutations(output, permutations) if permutations is not None else output


def same_position_recovery(
    source: torch.Tensor,
    valid_token_mask: torch.Tensor,
    recovery_mask: torch.Tensor,
    *,
    radius: int,
    permutations: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    working_source = apply_frame_permutations(source, permutations) if permutations is not None else source
    working_valid = (
        apply_frame_permutations(valid_token_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else valid_token_mask
    )
    working_mask = (
        apply_frame_permutations(recovery_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else recovery_mask
    )
    neighbors, scores = build_same_position_neighbors(working_valid, radius=radius)
    visible_mask = working_valid & ~working_mask
    recovered = aggregate_neighbors(
        working_source,
        neighbors,
        scores,
        support_mask=visible_mask,
        topk=max(radius * 2, 1),
        fill_source_if_missing=False,
    )
    prediction = working_source.clone()
    prediction[working_mask] = recovered[working_mask]
    return restore_frame_permutations(prediction, permutations) if permutations is not None else prediction


def g11_knn_context(
    source: torch.Tensor,
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    *,
    topk: int,
    permutations: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    working_source = apply_frame_permutations(source, permutations) if permutations is not None else source
    working_valid = (
        apply_frame_permutations(valid_token_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else valid_token_mask
    )
    output = aggregate_neighbors(
        working_source,
        neighbor_indices,
        neighbor_scores,
        support_mask=working_valid,
        topk=topk,
        fill_source_if_missing=True,
    )
    return restore_frame_permutations(output, permutations) if permutations is not None else output


def g11_knn_recovery(
    source: torch.Tensor,
    valid_token_mask: torch.Tensor,
    recovery_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    *,
    topk: int,
    permutations: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    working_source = apply_frame_permutations(source, permutations) if permutations is not None else source
    working_valid = (
        apply_frame_permutations(valid_token_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else valid_token_mask
    )
    working_mask = (
        apply_frame_permutations(recovery_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else recovery_mask
    )
    visible_mask = working_valid & ~working_mask
    recovered = aggregate_neighbors(
        working_source,
        neighbor_indices,
        neighbor_scores,
        support_mask=visible_mask,
        topk=topk,
        fill_source_if_missing=False,
    )
    prediction = working_source.clone()
    prediction[working_mask] = recovered[working_mask]
    return restore_frame_permutations(prediction, permutations) if permutations is not None else prediction


@dataclass
class CheckpointMethod:
    name: str
    checkpoint_path: str
    model: "Stage1CBEvalModel"


class Stage1CBEvalModel(nn.Module):
    def __init__(
        self,
        *,
        d_geom: int = 1024,
        continuity_radius: int = 2,
        continuity_mlp_hidden_ratio: float = 2.0,
        continuity_attention_heads: int = 4,
        corr_score_beta: float = 1.0,
        time_bias_init: float = -0.10,
    ) -> None:
        super().__init__()
        self.continuity_selector = ContinuityUtilitySelector(d_geom=d_geom)
        self.activated_corr_graph = ActivatedCorrespondenceGraph()
        self.continuity_builder = ContinuityBuilder(
            d_geom=d_geom,
            radius_t=continuity_radius,
            use_spatial_neighbors=False,
            mlp_hidden_ratio=continuity_mlp_hidden_ratio,
            attention_heads=continuity_attention_heads,
            corr_score_beta=corr_score_beta,
            time_bias_init=time_bias_init,
        )
        self.geometry_decoder = GeometryDecoder(
            d_geom=d_geom,
            hidden_ratio=continuity_mlp_hidden_ratio,
            layer_names=("g11",),
        )
        self.mask_token = nn.Parameter(torch.zeros(d_geom))
        nn.init.normal_(self.mask_token, std=0.02)

    def load_checkpoint(self, checkpoint_path: str, device: torch.device) -> dict:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = payload.get("model", payload)
        selector_state = {
            key.removeprefix("continuity_selector."): value
            for key, value in state_dict.items()
            if key.startswith("continuity_selector.")
        }
        if selector_state:
            self.continuity_selector.load_state_dict(selector_state, strict=False)
        activated_state = {
            key.removeprefix("activated_corr_graph."): value
            for key, value in state_dict.items()
            if key.startswith("activated_corr_graph.")
        }
        if activated_state:
            self.activated_corr_graph.load_state_dict(activated_state, strict=False)
        self.continuity_builder.load_state_dict(
            {
                key.removeprefix("continuity_builder."): value
                for key, value in state_dict.items()
                if key.startswith("continuity_builder.")
            },
            strict=False,
        )
        self.geometry_decoder.load_state_dict(
            {
                key.removeprefix("geometry_decoder."): value
                for key, value in state_dict.items()
                if key.startswith("geometry_decoder.")
            },
            strict=False,
        )
        if "mask_token" in state_dict:
            self.mask_token.data.copy_(state_dict["mask_token"].to(device=device, dtype=self.mask_token.dtype))
        self.to(device)
        self.eval()
        return payload


def _fcp_edge_activation(
    model: Stage1CBEvalModel,
    source: torch.Tensor,
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    *,
    use_continuity_selector: bool,
    use_activated_corr_graph: bool,
) -> torch.Tensor | None:
    if not use_continuity_selector:
        return None
    selector_output = model.continuity_selector(source.detach(), neighbor_indices, neighbor_scores, valid_token_mask)
    if not use_activated_corr_graph:
        return None
    activated = model.activated_corr_graph(
        neighbor_indices,
        neighbor_scores,
        selector_output["probs"].detach(),
    )
    return activated["activation"]


def fcp_context(
    model: Stage1CBEvalModel,
    source: torch.Tensor,
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    *,
    use_continuity_selector: bool,
    use_activated_corr_graph: bool,
    permutations: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    working_source = apply_frame_permutations(source, permutations) if permutations is not None else source
    working_valid = (
        apply_frame_permutations(valid_token_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else valid_token_mask
    )
    edge_activation = _fcp_edge_activation(
        model,
        working_source,
        working_valid,
        neighbor_indices,
        neighbor_scores,
        use_continuity_selector=use_continuity_selector,
        use_activated_corr_graph=use_activated_corr_graph,
    )
    continuity, _ = model.continuity_builder.forward_from_fused(
        working_source,
        mode="corr_graph",
        corr_neighbor_indices=neighbor_indices,
        corr_neighbor_scores=neighbor_scores,
        corr_edge_activation=edge_activation,
        return_aux=True,
    )
    return restore_frame_permutations(continuity, permutations) if permutations is not None else continuity


def fcp_recovery(
    model: Stage1CBEvalModel,
    source: torch.Tensor,
    valid_token_mask: torch.Tensor,
    recovery_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    *,
    use_continuity_selector: bool,
    use_activated_corr_graph: bool,
    permutations: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    working_source = apply_frame_permutations(source, permutations) if permutations is not None else source
    working_valid = (
        apply_frame_permutations(valid_token_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else valid_token_mask
    )
    working_mask = (
        apply_frame_permutations(recovery_mask.unsqueeze(-1).to(dtype=source.dtype), permutations).squeeze(-1).bool()
        if permutations is not None
        else recovery_mask
    )
    edge_activation = _fcp_edge_activation(
        model,
        working_source,
        working_valid,
        neighbor_indices,
        neighbor_scores,
        use_continuity_selector=use_continuity_selector,
        use_activated_corr_graph=use_activated_corr_graph,
    )
    masked_source = working_source.clone()
    masked_source[working_mask] = model.mask_token.to(dtype=masked_source.dtype)
    continuity, _ = model.continuity_builder.forward_from_fused(
        masked_source,
        mode="corr_graph",
        corr_neighbor_indices=neighbor_indices,
        corr_neighbor_scores=neighbor_scores,
        corr_edge_activation=edge_activation,
        return_aux=True,
    )
    prediction = model.geometry_decoder(continuity)["g11"]
    return restore_frame_permutations(prediction, permutations) if permutations is not None else prediction

"""Stage 1 v2 correspondence-aware continuity pretraining."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qwen_vl.model.geometry_bank import (
    ActivatedCorrespondenceGraph,
    BaseGeometryFusion,
    ContinuityBuilder,
    ContinuityUtilitySelector,
    GeoProjector,
    GeometryDecoder,
    VGGTBankExtractor,
    VGGTBankFeatureOutput,
    continuity_utility_loss,
)
from qwen_vl.model.geometry_bank.correspondence_losses import (
    attention_alignment_kl,
    corr_recall_from_attention,
    lov_global_metrics,
    masked_cosine_similarity_metric,
    multi_positive_infonce,
    pool_frame_tokens,
    reconstruction_metrics,
    variance_loss,
)
from qwen_vl.train.stage1_geometry_v2 import (
    Stage1GeometryDatasetV2,
    Stage1SourceGroupedBatchSampler,
    stage1_v2_collate_fn,
    visualize_lov,
    visualize_mgc,
    visualize_shuffle_ablation,
)
from qwen_vl.train.stage1_compact_cache import dequantize_projected_tokenwise

DEFAULT_FEATURE_LAYERS = ("g11", "g17", "g23")
FEATURE_LAYER_TO_VGGT_ID = {"g11": 11, "g17": 17, "g23": 23}


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_feature_layers(value: str) -> List[str]:
    layers = [item.strip() for item in str(value).split(",") if item.strip()]
    if not layers:
        raise ValueError("feature_layers must not be empty")
    invalid = [layer for layer in layers if layer not in FEATURE_LAYER_TO_VGGT_ID]
    if invalid:
        raise ValueError(f"Unsupported feature_layers: {invalid}")
    return layers


def raw_feature_name(layer_name: str) -> str:
    return f"{layer_name}_raw"


def metric_value_or_zero(metric_dict: Dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    return metric_dict.get(key, reference.new_tensor(0.0))


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0_print(*args):
    if not is_dist() or dist.get_rank() == 0:
        print(*args, flush=True)


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")
    return rank, world_size, device


class GlobalGeometryDecoder(nn.Module):
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


class Stage1ContinuityModelV2(nn.Module):
    def __init__(
        self,
        geometry_encoder_path: str,
        *,
        feature_layers: Sequence[str] = DEFAULT_FEATURE_LAYERS,
        d_geom: int = 1024,
        continuity_radius: int = 1,
        continuity_use_spatial_neighbors: bool = False,
        continuity_mlp_hidden_ratio: float = 2.0,
        continuity_attention_heads: int = 4,
        corr_score_beta: float = 1.0,
        time_bias_init: float = -0.10,
    ):
        super().__init__()
        self.feature_layers = tuple(feature_layers)
        self.raw_feature_layers = tuple(raw_feature_name(name) for name in self.feature_layers)
        self.geometry_encoder = VGGTBankExtractor(
            model_path=geometry_encoder_path,
            layer_ids=tuple(FEATURE_LAYER_TO_VGGT_ID[name] for name in self.feature_layers),
            freeze_encoder=True,
        )
        self.geo_projector = GeoProjector(
            input_dims={name: 8192 for name in self.raw_feature_layers},
            d_geom=d_geom,
        )
        self.base_geometry_fusion = (
            BaseGeometryFusion(
                d_geom=d_geom,
                hidden_ratio=continuity_mlp_hidden_ratio,
                num_layers=len(self.feature_layers),
            )
            if len(self.feature_layers) > 1
            else None
        )
        self.continuity_selector = ContinuityUtilitySelector(d_geom=d_geom)
        self.activated_corr_graph = ActivatedCorrespondenceGraph()
        self.continuity_builder = ContinuityBuilder(
            d_geom=d_geom,
            radius_t=continuity_radius,
            use_spatial_neighbors=continuity_use_spatial_neighbors,
            mlp_hidden_ratio=continuity_mlp_hidden_ratio,
            attention_heads=continuity_attention_heads,
            corr_score_beta=corr_score_beta,
            time_bias_init=time_bias_init,
        )
        self.geometry_decoder = GeometryDecoder(
            d_geom=d_geom,
            hidden_ratio=continuity_mlp_hidden_ratio,
            layer_names=self.feature_layers,
        )
        self.lov_global_head = GlobalGeometryDecoder(d_geom=d_geom, hidden_ratio=continuity_mlp_hidden_ratio)
        self.mask_token = nn.Parameter(torch.zeros(d_geom))
        nn.init.normal_(self.mask_token, std=0.02)

        for parameter in self.geometry_encoder.parameters():
            parameter.requires_grad = False

    def initialize_from_checkpoint(self, checkpoint_path: str, device: torch.device) -> None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = payload.get("model", payload)
        own_state = self.state_dict()
        loaded = {}
        for key, value in state_dict.items():
            if key in own_state and own_state[key].shape == value.shape:
                loaded[key] = value
        self.load_state_dict(loaded, strict=False)

    def _feature_output_from_cache(self, cached_features: Dict) -> VGGTBankFeatureOutput:
        return VGGTBankFeatureOutput(
            layer_tokens={
                raw_name: cached_features[raw_name]
                for raw_name in self.raw_feature_layers
            },
            frame_layout=type("FrameLayoutLike", (), {
                "token_counts": [int(v) for v in cached_features["token_counts"]],
                "frame_shapes": [tuple(int(x) for x in shape) for shape in cached_features["frame_shapes"]],
            })(),
            patch_grid=tuple(int(v) for v in cached_features.get("patch_grid", cached_features["frame_shapes"][0])),
            merged_grid=tuple(int(v) for v in cached_features.get("merged_grid", cached_features["frame_shapes"][0])),
        )

    def fuse_projected(self, projected: Dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self.feature_layers) == 1:
            return projected[self.feature_layers[0]]
        if self.base_geometry_fusion is None:
            raise RuntimeError("base_geometry_fusion is missing for multi-layer Stage1 configuration")
        return self.base_geometry_fusion([projected[name] for name in self.feature_layers])

    def encode_batch(self, batch: Dict) -> tuple[Dict[str, torch.Tensor], Sequence, torch.Tensor]:
        if batch["cached_features"] is not None:
            if batch["cached_features"].get("feature_space") == "projected":
                projected = {name: batch["cached_features"][name] for name in self.feature_layers}
            elif batch["cached_features"].get("feature_space") == "projected_quantized":
                projected = {
                    name: dequantize_projected_tokenwise(
                        batch["cached_features"][f"{name}_q"],
                        batch["cached_features"][f"{name}_scale"],
                    )
                    for name in self.feature_layers
                }
            else:
                projected = self.geo_projector(
                    {raw_name: batch["cached_features"][raw_name] for raw_name in self.raw_feature_layers}
                )
            return projected, batch["frame_shapes"], batch["valid_patch_mask"]

        if batch["geometry_inputs"] is None:
            raise ValueError("Stage 1 v2 batch must provide cached_features or geometry_inputs")

        geometry_inputs = batch["geometry_inputs"]
        sample_projected: List[Dict[str, torch.Tensor]] = []
        frame_shapes: List[List[tuple[int, int]]] = []
        token_count_rows: List[List[int]] = []
        max_frames = 0
        max_patches = 0
        for sample_inputs_all in geometry_inputs:
            extracted = self.geometry_encoder.extract(sample_inputs_all)
            sample_projected.append(
                self.geo_projector(
                    {raw_name: extracted.layer_tokens[raw_name] for raw_name in self.raw_feature_layers}
                )
            )
            sample_frame_shapes = [tuple(int(x) for x in shape) for shape in extracted.frame_layout.frame_shapes]
            sample_token_counts = [int(v) for v in extracted.frame_layout.token_counts]
            frame_shapes.append(sample_frame_shapes)
            token_count_rows.append(sample_token_counts)
            max_frames = max(max_frames, len(sample_token_counts))
            if sample_token_counts:
                max_patches = max(max_patches, max(sample_token_counts))

        batch_size = len(geometry_inputs)
        d_geom = sample_projected[0][self.feature_layers[0]].shape[-1]
        padded = {
            key: sample_projected[0][key].new_zeros((batch_size, max_frames, max_patches, d_geom))
            for key in self.feature_layers
        }
        valid_patch_mask = torch.zeros(
            (batch_size, max_frames, max_patches),
            dtype=torch.bool,
            device=padded[self.feature_layers[0]].device,
        )
        for batch_idx, projected_sample in enumerate(sample_projected):
            num_frames, num_patches = projected_sample[self.feature_layers[0]].shape[:2]
            for key in self.feature_layers:
                padded[key][batch_idx, :num_frames, :num_patches] = projected_sample[key]
            sample_valid_frames = batch["valid_frame_mask"][batch_idx]
            for frame_idx, token_count in enumerate(token_count_rows[batch_idx]):
                if token_count <= 0:
                    continue
                valid_patch_mask[batch_idx, frame_idx, :token_count] = sample_valid_frames[frame_idx]
        return padded, frame_shapes, valid_patch_mask


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(model, optimizer, scheduler, step: int, output_dir: str, extra: Dict):
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, f"checkpoint-{step}.pt")
    module = model.module if isinstance(model, DDP) else model
    model_state = {}
    for prefix, submodule in {
        "geo_projector": module.geo_projector,
        "base_geometry_fusion": module.base_geometry_fusion,
        "continuity_selector": module.continuity_selector,
        "activated_corr_graph": module.activated_corr_graph,
        "continuity_builder": module.continuity_builder,
        "geometry_decoder": module.geometry_decoder,
        "lov_global_head": module.lov_global_head,
    }.items():
        if submodule is None:
            continue
        for key, value in submodule.state_dict().items():
            model_state[f"{prefix}.{key}"] = value.cpu()
    model_state["mask_token"] = module.mask_token.detach().cpu()
    payload = {
        "step": step,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "extra": extra,
    }
    torch.save(payload, target)
    torch.save(payload, os.path.join(output_dir, "latest.pt"))
    return target


def load_stage1_v2_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = payload.get("model", payload)
    module = model.module if isinstance(model, DDP) else model
    projector_state = module.geo_projector.state_dict()
    module.geo_projector.load_state_dict(
        {
            key.removeprefix("geo_projector."): value
            for key, value in state_dict.items()
            if key.startswith("geo_projector.")
            and key.removeprefix("geo_projector.") in projector_state
            and projector_state[key.removeprefix("geo_projector.")].shape == value.shape
        },
        strict=False,
    )
    if module.base_geometry_fusion is not None:
        base_state = {
            key.removeprefix("base_geometry_fusion."): value
            for key, value in state_dict.items()
            if key.startswith("base_geometry_fusion.")
        }
        if base_state:
            module.base_geometry_fusion.load_state_dict(base_state, strict=False)
    selector_state = {
        key.removeprefix("continuity_selector."): value
        for key, value in state_dict.items()
        if key.startswith("continuity_selector.")
    }
    if selector_state:
        module.continuity_selector.load_state_dict(selector_state, strict=False)
    activated_state = {
        key.removeprefix("activated_corr_graph."): value
        for key, value in state_dict.items()
        if key.startswith("activated_corr_graph.")
    }
    if activated_state:
        module.activated_corr_graph.load_state_dict(activated_state, strict=False)
    module.continuity_builder.load_state_dict(
        {
            key.removeprefix("continuity_builder."): value
            for key, value in state_dict.items()
            if key.startswith("continuity_builder.")
        },
        strict=False,
    )
    module.geometry_decoder.load_state_dict(
        {
            key.removeprefix("geometry_decoder."): value
            for key, value in state_dict.items()
            if key.startswith("geometry_decoder.")
            and key.removeprefix("geometry_decoder.") in module.geometry_decoder.state_dict()
            and module.geometry_decoder.state_dict()[key.removeprefix("geometry_decoder.")].shape == value.shape
        },
        strict=False,
    )
    lov_state = {
        key.removeprefix("lov_global_head."): value
        for key, value in state_dict.items()
        if key.startswith("lov_global_head.")
    }
    if lov_state:
        module.lov_global_head.load_state_dict(lov_state, strict=False)
    if "mask_token" in state_dict:
        module.mask_token.data.copy_(state_dict["mask_token"].to(device=device, dtype=module.mask_token.dtype))
    return payload


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    non_blocking = device.type == "cuda"
    moved = dict(batch)
    moved["valid_frame_mask"] = batch["valid_frame_mask"].to(device, non_blocking=non_blocking)
    moved["valid_patch_mask"] = (
        batch["valid_patch_mask"].to(device, non_blocking=non_blocking)
        if batch["valid_patch_mask"] is not None
        else None
    )
    moved["cached_features"] = None
    if batch["cached_features"] is not None:
        moved["cached_features"] = {
            key: value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
            for key, value in batch["cached_features"].items()
        }
    moved["corr_graph"] = {
        key: value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
        for key, value in batch["corr_graph"].items()
    }
    moved["geometry_inputs"] = None
    if batch["geometry_inputs"] is not None:
        if isinstance(batch["geometry_inputs"], torch.Tensor):
            moved["geometry_inputs"] = batch["geometry_inputs"].to(device, non_blocking=non_blocking)
        else:
            moved["geometry_inputs"] = [inputs.to(device, non_blocking=non_blocking) for inputs in batch["geometry_inputs"]]
    return moved


def prepare_targets(projected: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in projected.items()}


def batch_question_type(question_types: Sequence[str]) -> str:
    filtered = [question_type for question_type in question_types if isinstance(question_type, str) and question_type]
    if not filtered:
        return "unknown"
    values = set(filtered)
    if len(values) == 1:
        return filtered[0]
    return "mixed"


def curriculum_config(step: int, args) -> Dict:
    if step < args.phase0_steps:
        return {
            "phase": "phase0",
            "corr_nce_weight": 0.0,
            "attn_weight": 0.0,
            "lov_global_weight": 0.0,
            "var_weight": 0.0,
            "mask_probs": {"random_patch": 1.0, "corr_tube": 0.0, "frame_block": 0.0},
        }
    if step < args.phase0_steps + args.phase1_steps:
        return {
            "phase": "phase1",
            "corr_nce_weight": args.corr_nce_weight,
            "attn_weight": 0.0,
            "lov_global_weight": 0.0,
            "var_weight": args.var_weight,
            "mask_probs": {
                "random_patch": args.random_patch_prob,
                "corr_tube": args.corr_tube_prob,
                "frame_block": args.frame_block_prob,
            },
        }
    if step < args.phase3_start_step:
        return {
            "phase": "phase2",
            "corr_nce_weight": args.corr_nce_weight,
            "attn_weight": args.attn_weight,
            "lov_global_weight": args.lov_global_weight,
            "var_weight": args.var_weight,
            "mask_probs": {
                "random_patch": args.random_patch_prob,
                "corr_tube": args.corr_tube_prob,
                "frame_block": args.frame_block_prob,
            },
        }
    return {
        "phase": "phase3",
        "corr_nce_weight": args.corr_nce_weight,
        "attn_weight": args.attn_weight,
        "lov_global_weight": args.lov_global_weight,
        "var_weight": args.var_weight,
        "mask_probs": {
            "random_patch": args.random_patch_prob_phase3,
            "corr_tube": args.corr_tube_prob_phase3,
            "frame_block": args.frame_block_prob_phase3,
        },
    }


def ensure_mask(mask: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        if not mask.any():
            first_valid = valid_token_mask.nonzero(as_tuple=False)[0]
            mask[first_valid[0], first_valid[1]] = True
        return mask
    missing = ~mask.view(mask.shape[0], -1).any(dim=1)
    for batch_idx in missing.nonzero(as_tuple=False).view(-1):
        first_valid = valid_token_mask[batch_idx].nonzero(as_tuple=False)[0]
        mask[batch_idx, first_valid[0], first_valid[1]] = True
    return mask


def build_random_patch_mask(valid_token_mask: torch.Tensor, masked_ratio: float) -> torch.Tensor:
    sampled = torch.rand_like(valid_token_mask.float()) < masked_ratio
    mask = valid_token_mask & sampled
    return ensure_mask(mask, valid_token_mask)


def build_frame_block_mask(valid_token_mask: torch.Tensor, masked_ratio: float) -> torch.Tensor:
    mask = torch.zeros_like(valid_token_mask)
    for batch_idx in range(valid_token_mask.shape[0]):
        valid_frames = valid_token_mask[batch_idx].any(dim=-1).nonzero(as_tuple=False).view(-1)
        if valid_frames.numel() == 0:
            continue
        frame_idx = int(valid_frames[torch.randint(valid_frames.numel(), (1,), device=valid_token_mask.device)].item())
        valid_patches = valid_token_mask[batch_idx, frame_idx].nonzero(as_tuple=False).view(-1)
        if valid_patches.numel() == 0:
            continue
        block_size = max(1, int(math.ceil(valid_patches.numel() * masked_ratio)))
        start_max = max(valid_patches.numel() - block_size, 0)
        start = int(torch.randint(start_max + 1, (1,), device=valid_token_mask.device).item()) if start_max > 0 else 0
        chosen = valid_patches[start : start + block_size]
        mask[batch_idx, frame_idx, chosen] = True
    return ensure_mask(mask, valid_token_mask)


def build_corr_tube_mask(
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    masked_ratio: float,
    positive_topk: int,
    saliency_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = torch.zeros_like(valid_token_mask)
    for batch_idx in range(valid_token_mask.shape[0]):
        valid_coords = valid_token_mask[batch_idx].nonzero(as_tuple=False)
        if valid_coords.numel() == 0:
            continue
        anchor_count = max(1, int(math.ceil(valid_coords.shape[0] * masked_ratio * 0.35)))
        if saliency_probs is not None:
            saliency_values = saliency_probs[batch_idx, valid_coords[:, 0], valid_coords[:, 1]]
            if anchor_count >= valid_coords.shape[0]:
                top_idx = torch.argsort(saliency_values, descending=True)
            else:
                top_idx = torch.topk(saliency_values, k=anchor_count, dim=0).indices
            chosen_coords = valid_coords[top_idx]
        elif anchor_count >= valid_coords.shape[0]:
            chosen_coords = valid_coords
        else:
            perm = torch.randperm(valid_coords.shape[0], device=valid_token_mask.device)[:anchor_count]
            chosen_coords = valid_coords[perm]
        for frame_idx, patch_idx in chosen_coords.tolist():
            mask[batch_idx, frame_idx, patch_idx] = True
            neighbors = neighbor_indices[batch_idx, frame_idx, patch_idx, :positive_topk]
            for neighbor_frame, neighbor_patch in neighbors.tolist():
                if neighbor_frame < 0 or neighbor_patch < 0:
                    continue
                if neighbor_frame >= valid_token_mask.shape[1] or neighbor_patch >= valid_token_mask.shape[2]:
                    continue
                if valid_token_mask[batch_idx, neighbor_frame, neighbor_patch]:
                    mask[batch_idx, neighbor_frame, neighbor_patch] = True
    return ensure_mask(mask, valid_token_mask)


def build_mgc_mask_v2(
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    masked_ratio: float,
    mask_probs: Dict[str, float],
    positive_topk: int,
    saliency_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, List[str]]:
    mask = torch.zeros_like(valid_token_mask)
    chosen_modes: List[str] = []
    names = list(mask_probs.keys())
    probs = torch.tensor([mask_probs[name] for name in names], dtype=torch.float32)
    probs = probs / probs.sum().clamp_min(1e-6)
    sampled_mode_indices = torch.multinomial(probs, num_samples=valid_token_mask.shape[0], replacement=True)
    for batch_idx, mode_idx in enumerate(sampled_mode_indices.tolist()):
        mode = names[mode_idx]
        chosen_modes.append(mode)
        sample_valid = valid_token_mask[batch_idx : batch_idx + 1]
        sample_neighbors = neighbor_indices[batch_idx : batch_idx + 1]
        if mode == "corr_tube":
            sample_mask = build_corr_tube_mask(
                sample_valid,
                sample_neighbors,
                masked_ratio=masked_ratio,
                positive_topk=positive_topk,
                saliency_probs=None if saliency_probs is None else saliency_probs[batch_idx : batch_idx + 1],
            )
        elif mode == "frame_block":
            sample_mask = build_frame_block_mask(sample_valid, masked_ratio=masked_ratio)
        else:
            sample_mask = build_random_patch_mask(sample_valid, masked_ratio=masked_ratio)
        mask[batch_idx] = sample_mask[0]
    return ensure_mask(mask, valid_token_mask), chosen_modes


def gather_neighbor_targets(
    tensor: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_frames, num_patches, hidden_dim = tensor.shape
    take_k = min(topk, neighbor_indices.shape[-2])
    neighbors = neighbor_indices[..., :take_k, :].long()
    valid = (neighbors[..., 0] >= 0) & (neighbors[..., 1] >= 0)
    frames = neighbors[..., 0].clamp(min=0)
    patches = neighbors[..., 1].clamp(min=0)
    flat = tensor.reshape(batch_size, num_frames * num_patches, hidden_dim)
    flat_indices = frames * num_patches + patches
    batch_index = torch.arange(batch_size, device=tensor.device).view(batch_size, 1, 1, 1).expand_as(flat_indices)
    gathered = flat[batch_index, flat_indices]
    return gathered, valid


def compute_corr_contrastive(
    continuity: torch.Tensor,
    z_target: torch.Tensor,
    anchor_mask: torch.Tensor,
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    *,
    positive_topk: int,
    num_negatives: int,
    temperature: float,
    max_anchors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_mask = anchor_mask & (neighbor_indices[..., 0, 0] >= 0)
    candidate_coords = candidate_mask.nonzero(as_tuple=False)
    if candidate_coords.shape[0] == 0:
        zero = continuity.new_tensor(0.0)
        return zero, zero
    if candidate_coords.shape[0] > max_anchors:
        perm = torch.randperm(candidate_coords.shape[0], device=continuity.device)[:max_anchors]
        candidate_coords = candidate_coords[perm]

    negative_pool = z_target[valid_token_mask]
    if negative_pool.shape[0] == 0:
        zero = continuity.new_tensor(0.0)
        return zero, zero
    if negative_pool.shape[0] > num_negatives:
        perm = torch.randperm(negative_pool.shape[0], device=continuity.device)[:num_negatives]
        negative_pool = negative_pool[perm]

    anchors = []
    positives = []
    positive_weights = []
    for batch_idx, frame_idx, patch_idx in candidate_coords.tolist():
        anchor = continuity[batch_idx, frame_idx, patch_idx]
        current_neighbors = neighbor_indices[batch_idx, frame_idx, patch_idx, :positive_topk]
        current_scores = neighbor_scores[batch_idx, frame_idx, patch_idx, :positive_topk]
        valid = (current_neighbors[:, 0] >= 0) & (current_neighbors[:, 1] >= 0)
        if not valid.any():
            continue
        gathered = []
        scores = []
        for (other_frame, other_patch), score, is_valid in zip(current_neighbors.tolist(), current_scores.tolist(), valid.tolist()):
            if not is_valid:
                continue
            gathered.append(z_target[batch_idx, other_frame, other_patch])
            scores.append(score)
        if not gathered:
            continue
        while len(gathered) < positive_topk:
            gathered.append(gathered[-1])
            scores.append(scores[-1])
        anchors.append(anchor)
        positives.append(torch.stack(gathered[:positive_topk], dim=0))
        positive_weights.append(torch.tensor(scores[:positive_topk], device=continuity.device, dtype=torch.float32))
    if not anchors:
        zero = continuity.new_tensor(0.0)
        return zero, zero

    anchors_tensor = torch.stack(anchors, dim=0)
    positives_tensor = torch.stack(positives, dim=0)
    positive_weights_tensor = torch.stack(positive_weights, dim=0)
    positive_weights_tensor = F.softmax(positive_weights_tensor, dim=-1)
    return multi_positive_infonce(
        anchors_tensor,
        positives_tensor.detach(),
        positive_weights_tensor.detach(),
        negative_pool.detach(),
        temperature=temperature,
    )


def build_context_hidden(
    module: Stage1ContinuityModelV2,
    z: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    mode: str,
    corr_neighbor_indices: torch.Tensor,
    corr_neighbor_scores: torch.Tensor,
    corr_edge_activation: torch.Tensor | None = None,
    shuffle_frames: bool = False,
) -> tuple[torch.Tensor, Dict]:
    working_z = z
    inverse_perm = None
    if shuffle_frames:
        perms = []
        shuffled = []
        for sample in working_z:
            perm = torch.randperm(sample.shape[0], device=sample.device)
            perms.append(perm)
            shuffled.append(sample[perm])
        working_z = torch.stack(shuffled, dim=0)
        inverse_perm = [torch.argsort(perm) for perm in perms]

    continuity, aux = module.continuity_builder.forward_from_fused(
        working_z,
        mode=mode,
        corr_neighbor_indices=corr_neighbor_indices,
        corr_neighbor_scores=corr_neighbor_scores,
        corr_edge_activation=corr_edge_activation,
        return_aux=True,
    )
    if inverse_perm is not None:
        continuity = torch.stack([continuity[idx, inv] for idx, inv in enumerate(inverse_perm)], dim=0)
        if aux.get("attention_weights") is not None:
            aux["attention_weights"] = torch.stack([aux["attention_weights"][idx, inv] for idx, inv in enumerate(inverse_perm)], dim=0)
        if aux.get("neighbor_valid_mask") is not None:
            aux["neighbor_valid_mask"] = torch.stack([aux["neighbor_valid_mask"][idx, inv] for idx, inv in enumerate(inverse_perm)], dim=0)
    return continuity, aux


def shuffle_frames_per_sample(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([sample[torch.randperm(sample.shape[0], device=sample.device)] for sample in z], dim=0)


def build_mean_predictor_continuity(module: Stage1ContinuityModelV2, masked_z: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
    weights = visible_mask.to(dtype=masked_z.dtype).unsqueeze(-1)
    pooled = (masked_z * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)
    expanded = pooled[:, None, None, :].expand_as(masked_z)
    return module.continuity_builder.forward_from_fused(expanded, mode="current_frame_only")


def score_proxy(metrics: Dict[str, float]) -> float:
    return float(metrics.get("mgc_cos", 0.0) + metrics.get("lov_global_cos", 0.0) + max(metrics.get("shuffle_gap", 0.0), 0.0))


def norm_or_zero(projected: Dict[str, torch.Tensor], layer_name: str, reference: torch.Tensor) -> float:
    if layer_name not in projected:
        return 0.0
    return float(projected[layer_name].norm(dim=-1).mean().detach().item())


def cos_or_zero(
    continuity: torch.Tensor,
    projected: Dict[str, torch.Tensor],
    layer_name: str,
    valid_token_mask: torch.Tensor,
) -> float:
    if layer_name not in projected:
        return 0.0
    return float(masked_cosine_similarity_metric(continuity, projected[layer_name].detach(), valid_token_mask).item())


def per_token_reconstruction_gain(
    corr_predictions: Dict[str, torch.Tensor],
    current_predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    valid_token_mask: torch.Tensor,
    *,
    layer_names: Sequence[str],
    temperature: float = 0.10,
) -> torch.Tensor:
    gains = []
    for name in layer_names:
        corr_cos = F.cosine_similarity(corr_predictions[name].float(), targets[name].float(), dim=-1)
        curr_cos = F.cosine_similarity(current_predictions[name].float(), targets[name].float(), dim=-1)
        gains.append(corr_cos - curr_cos)
    gain = torch.stack(gains, dim=0).mean(dim=0)
    gain = gain * valid_token_mask.float()
    return torch.sigmoid(gain / max(temperature, 1e-6)).detach()


def run_baseline_eval(
    module: Stage1ContinuityModelV2,
    z: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    valid_token_mask: torch.Tensor,
    corr_graph: Dict[str, torch.Tensor],
    holdout_frame_indices: List[int],
    *,
    mgc_mask: torch.Tensor,
    positive_topk: int,
    baseline_name: str,
    use_continuity_selector: bool = False,
    use_activated_corr_graph: bool = False,
) -> Dict[str, float]:
    neighbor_indices = corr_graph["neighbor_indices"]
    neighbor_scores = corr_graph["neighbor_scores"]
    selector_output = None
    edge_activation = None
    if use_continuity_selector:
        selector_output = module.continuity_selector(z, neighbor_indices, neighbor_scores, valid_token_mask)
    if use_activated_corr_graph and selector_output is not None:
        edge_activation = module.activated_corr_graph(
            neighbor_indices,
            neighbor_scores,
            selector_output["probs"],
        )["activation"]
    visible_mask = valid_token_mask & ~mgc_mask
    masked_z = z.clone()
    masked_z[mgc_mask] = module.mask_token.to(dtype=masked_z.dtype)
    holdout_z = z.clone()
    for batch_idx, frame_idx in enumerate(holdout_frame_indices):
        holdout_z[batch_idx, frame_idx] = module.mask_token.to(dtype=holdout_z.dtype)

    if baseline_name == "mean_predictor":
        mgc_cont = build_mean_predictor_continuity(module, masked_z, visible_mask)
        lov_cont = build_mean_predictor_continuity(module, holdout_z, valid_token_mask & ~torch.zeros_like(valid_token_mask))
    elif baseline_name == "current_frame_only":
        mgc_cont, _ = module.continuity_builder.forward_from_fused(masked_z, mode="current_frame_only", return_aux=True)
        lov_cont, _ = module.continuity_builder.forward_from_fused(holdout_z, mode="current_frame_only", return_aux=True)
    elif baseline_name == "same_position":
        mgc_cont, _ = module.continuity_builder.forward_from_fused(masked_z, mode="same_position", return_aux=True)
        lov_cont, _ = module.continuity_builder.forward_from_fused(holdout_z, mode="same_position", return_aux=True)
    elif baseline_name == "shuffled_feature_knn":
        mgc_cont, _ = build_context_hidden(
            module, masked_z, valid_token_mask, mode="corr_graph", corr_neighbor_indices=neighbor_indices, corr_neighbor_scores=neighbor_scores, corr_edge_activation=edge_activation, shuffle_frames=True
        )
        lov_cont, _ = build_context_hidden(
            module, holdout_z, valid_token_mask, mode="corr_graph", corr_neighbor_indices=neighbor_indices, corr_neighbor_scores=neighbor_scores, corr_edge_activation=edge_activation, shuffle_frames=True
        )
    else:
        mgc_cont, _ = build_context_hidden(
            module, masked_z, valid_token_mask, mode="corr_graph", corr_neighbor_indices=neighbor_indices, corr_neighbor_scores=neighbor_scores, corr_edge_activation=edge_activation
        )
        lov_cont, _ = build_context_hidden(
            module, holdout_z, valid_token_mask, mode="corr_graph", corr_neighbor_indices=neighbor_indices, corr_neighbor_scores=neighbor_scores, corr_edge_activation=edge_activation
        )

    mgc_predictions = module.geometry_decoder(mgc_cont)
    mgc_metrics = reconstruction_metrics(mgc_predictions, targets, mgc_mask, layer_names=module.feature_layers)

    lov_preds = []
    lov_targets = []
    for batch_idx, frame_idx in enumerate(holdout_frame_indices):
        pooled_pred = pool_frame_tokens(lov_cont[batch_idx, frame_idx : frame_idx + 1], valid_token_mask[batch_idx, frame_idx : frame_idx + 1])[0]
        pooled_target = pool_frame_tokens(z[batch_idx, frame_idx : frame_idx + 1].detach(), valid_token_mask[batch_idx, frame_idx : frame_idx + 1])[0]
        lov_preds.append(module.lov_global_head(pooled_pred))
        lov_targets.append(pooled_target)
    lov_metrics = lov_global_metrics(torch.stack(lov_preds, dim=0), torch.stack(lov_targets, dim=0))

    shuffled_cont, _ = build_context_hidden(
        module, masked_z, valid_token_mask, mode="corr_graph" if baseline_name in {"feature_knn", "shuffled_feature_knn"} else ("same_position" if baseline_name == "same_position" else "current_frame_only"), corr_neighbor_indices=neighbor_indices, corr_neighbor_scores=neighbor_scores, corr_edge_activation=edge_activation, shuffle_frames=True
    ) if baseline_name != "mean_predictor" else (build_mean_predictor_continuity(module, shuffle_frames_per_sample(masked_z), visible_mask), {})
    shuffled_predictions = module.geometry_decoder(shuffled_cont)
    shuffled_metrics = reconstruction_metrics(shuffled_predictions, targets, mgc_mask, layer_names=module.feature_layers)
    shuffle_gap = float(mgc_metrics["cos"].item() - shuffled_metrics["cos"].item())

    return {
        "mgc_l1": float(mgc_metrics["l1"].item()),
        "mgc_cos": float(mgc_metrics["cos"].item()),
        "lov_global_l1": float(lov_metrics["l1"].item()),
        "lov_global_cos": float(lov_metrics["cos"].item()),
        "shuffle_gap": shuffle_gap,
    }


def compute_stage1_v2_batch(model: nn.Module, batch: Dict, args, step: int) -> Dict:
    module = model.module if isinstance(model, DDP) else model
    batch = move_batch_to_device(batch, next(module.parameters()).device)
    projected, _, valid_patch_mask = module.encode_batch(batch)
    targets = prepare_targets(projected)
    z = module.fuse_projected(projected)
    valid_frame_mask = batch["valid_frame_mask"]
    valid_token_mask = batch["valid_patch_mask"] if batch["valid_patch_mask"] is not None else valid_patch_mask
    corr_graph = batch["corr_graph"]
    neighbor_indices = corr_graph["neighbor_indices"]
    neighbor_scores = corr_graph["neighbor_scores"]
    neighbor_valid_mask = corr_graph["neighbor_valid_mask"]

    selector_output = None
    utility_target = None
    selector_loss = z.new_tensor(0.0)
    selector_loss_dict = {"bce": z.new_tensor(0.0), "budget": z.new_tensor(0.0), "mean_budget": z.new_tensor(0.0)}
    if parse_bool(getattr(args, "use_continuity_selector", "False")):
        selector_output = module.continuity_selector(z.detach(), neighbor_indices, neighbor_scores, valid_token_mask)
        with torch.no_grad():
            current_frame_cont, _ = module.continuity_builder.forward_from_fused(
                z,
                mode="current_frame_only",
                return_aux=True,
            )
            corr_seed_cont, _ = module.continuity_builder.forward_from_fused(
                z,
                mode="corr_graph",
                corr_neighbor_indices=neighbor_indices,
                corr_neighbor_scores=neighbor_scores,
                return_aux=True,
            )
            utility_target = per_token_reconstruction_gain(
                module.geometry_decoder(corr_seed_cont),
                module.geometry_decoder(current_frame_cont),
                targets,
                valid_token_mask,
                layer_names=module.feature_layers,
                temperature=args.cus_target_temperature,
            )
        selector_loss, selector_loss_dict = continuity_utility_loss(
            selector_output["logits"],
            utility_target,
            valid_token_mask,
            budget_ratio=args.cus_budget_ratio,
            budget_weight=args.cus_budget_weight,
        )

    edge_activation = None
    edge_activation_logits = None
    if parse_bool(getattr(args, "use_activated_corr_graph", "False")) and selector_output is not None:
        activated_output = module.activated_corr_graph(
            neighbor_indices,
            neighbor_scores,
            selector_output["probs"].detach(),
        )
        edge_activation = activated_output["activation"]
        edge_activation_logits = activated_output["logits"]

    curriculum = curriculum_config(step, args)
    mgc_mask, mgc_modes = build_mgc_mask_v2(
        valid_token_mask,
        neighbor_indices,
        masked_ratio=args.masked_ratio,
        mask_probs=curriculum["mask_probs"],
        positive_topk=args.positive_topk,
        saliency_probs=None if selector_output is None else selector_output["probs"].detach(),
    )
    masked_z = z.clone()
    masked_z[mgc_mask] = module.mask_token.to(dtype=masked_z.dtype)

    continuity_mgc, continuity_aux = module.continuity_builder.forward_from_fused(
        masked_z,
        mode=args.continuity_mode,
        corr_neighbor_indices=neighbor_indices,
        corr_neighbor_scores=neighbor_scores,
        corr_edge_activation=edge_activation,
        return_aux=True,
    )
    mgc_predictions = module.geometry_decoder(continuity_mgc)
    mgc_metrics = reconstruction_metrics(
        mgc_predictions,
        targets,
        mgc_mask,
        layer_names=module.feature_layers,
        cosine_weight=args.mgc_cosine_weight,
        l1_weight=args.mgc_l1_weight,
    )

    lov_global_metrics_dict = {
        "l1": z.new_tensor(0.0),
        "cos": z.new_tensor(0.0),
        "cos_loss": z.new_tensor(0.0),
        "total": z.new_tensor(0.0),
    }
    holdout_frames: List[int] = []
    if curriculum["lov_global_weight"] > 0:
        holdout_z = z.clone()
        for batch_idx in range(valid_frame_mask.shape[0]):
            valid_frames = valid_frame_mask[batch_idx].nonzero(as_tuple=False).view(-1)
            holdout = int(valid_frames[torch.randint(valid_frames.numel(), (1,), device=valid_frame_mask.device)].item())
            holdout_frames.append(holdout)
            holdout_z[batch_idx, holdout] = module.mask_token.to(dtype=holdout_z.dtype)
        continuity_lov, _ = module.continuity_builder.forward_from_fused(
            holdout_z,
            mode=args.continuity_mode,
            corr_neighbor_indices=neighbor_indices,
            corr_neighbor_scores=neighbor_scores,
            corr_edge_activation=edge_activation,
            return_aux=True,
        )
        lov_preds = []
        lov_targets = []
        for batch_idx, holdout in enumerate(holdout_frames):
            pooled_pred = pool_frame_tokens(
                continuity_lov[batch_idx, holdout : holdout + 1],
                valid_token_mask[batch_idx, holdout : holdout + 1],
            )[0]
            lov_preds.append(module.lov_global_head(pooled_pred))
            lov_targets.append(
                pool_frame_tokens(
                    z[batch_idx, holdout : holdout + 1].detach(),
                    valid_token_mask[batch_idx, holdout : holdout + 1],
                )[0]
            )
        lov_global_metrics_dict = lov_global_metrics(torch.stack(lov_preds, dim=0), torch.stack(lov_targets, dim=0))
    else:
        holdout_frames = [0 for _ in range(valid_frame_mask.shape[0])]

    contrastive_loss, info_nce_acc = compute_corr_contrastive(
        continuity_mgc,
        z.detach(),
        mgc_mask,
        valid_token_mask,
        neighbor_indices,
        neighbor_scores,
        positive_topk=args.positive_topk,
        num_negatives=args.num_negatives,
        temperature=args.temperature,
        max_anchors=args.max_contrastive_anchors,
    )
    attn_loss = attention_alignment_kl(
        continuity_aux["attention_weights"],
        neighbor_scores,
        neighbor_valid_mask,
        positive_topk=args.positive_topk,
    )
    var_loss = variance_loss(continuity_mgc, valid_token_mask, gamma=args.variance_gamma)
    corr_recall = corr_recall_from_attention(
        continuity_aux["attention_weights"],
        continuity_aux["neighbor_valid_mask"] if continuity_aux["neighbor_valid_mask"] is not None else neighbor_valid_mask,
        positive_topk=args.positive_topk,
    )

    total_loss = (
        mgc_metrics["total"]
        + curriculum["corr_nce_weight"] * contrastive_loss
        + curriculum["attn_weight"] * attn_loss
        + curriculum["lov_global_weight"] * lov_global_metrics_dict["total"]
        + curriculum["var_weight"] * var_loss
        + args.cus_loss_weight * selector_loss
    )

    with torch.no_grad():
        shuffled_cont, _ = module.continuity_builder.forward_from_fused(
            masked_z,
            mode=args.continuity_mode,
            corr_neighbor_indices=neighbor_indices,
            corr_neighbor_scores=neighbor_scores,
            corr_edge_activation=edge_activation,
            return_aux=True,
        )
        shuffled_perm = torch.stack([sample[torch.randperm(sample.shape[0])] for sample in masked_z], dim=0)
        shuffled_ctx, _ = module.continuity_builder.forward_from_fused(
            shuffled_perm,
            mode=args.continuity_mode,
            corr_neighbor_indices=neighbor_indices,
            corr_neighbor_scores=neighbor_scores,
            corr_edge_activation=edge_activation,
            return_aux=True,
        )
        shuffled_metrics = reconstruction_metrics(
            module.geometry_decoder(shuffled_ctx),
            targets,
            mgc_mask,
            layer_names=module.feature_layers,
            cosine_weight=args.mgc_cosine_weight,
            l1_weight=args.mgc_l1_weight,
        )
        shuffle_gap = float(mgc_metrics["cos"].item() - shuffled_metrics["cos"].item())

    return {
        "total_loss": total_loss,
        "mgc_metrics": mgc_metrics,
        "lov_global_metrics": lov_global_metrics_dict,
        "contrastive_loss": contrastive_loss,
        "attn_loss": attn_loss,
        "var_loss": var_loss,
        "selector_loss": selector_loss,
        "selector_bce": selector_loss_dict["bce"],
        "selector_budget": selector_loss_dict["budget"],
        "selector_mean_budget": selector_loss_dict["mean_budget"],
        "info_nce_acc": info_nce_acc,
        "corr_recall": corr_recall,
        "shuffle_gap": shuffle_gap,
        "holdout_frames": holdout_frames,
        "question_type": batch_question_type(batch["question_type"]),
        "phase": curriculum["phase"],
        "mask_mode_histogram": {mode: mgc_modes.count(mode) for mode in set(mgc_modes)},
        "norm_g11": norm_or_zero(projected, "g11", z),
        "norm_g17": norm_or_zero(projected, "g17", z),
        "norm_g23": norm_or_zero(projected, "g23", z),
        "norm_c": float(continuity_mgc.norm(dim=-1).mean().detach().item()),
        "cos_c_g11": cos_or_zero(continuity_mgc, projected, "g11", valid_token_mask),
        "cos_c_g17": cos_or_zero(continuity_mgc, projected, "g17", valid_token_mask),
        "cos_c_g23": cos_or_zero(continuity_mgc, projected, "g23", valid_token_mask),
        "saliency_mean": float(selector_output["probs"][valid_token_mask].mean().item()) if selector_output is not None and valid_token_mask.any() else 0.0,
        "utility_target_mean": float(utility_target[valid_token_mask].mean().item()) if utility_target is not None and valid_token_mask.any() else 0.0,
        "edge_activation_mean": float(edge_activation[neighbor_valid_mask].mean().item()) if edge_activation is not None and neighbor_valid_mask.any() else 0.0,
        "current_frame_only_metrics": None,
    }


@torch.no_grad()
def evaluate_stage1_v2(
    model: nn.Module,
    dataset: Stage1GeometryDatasetV2,
    device: torch.device,
    *,
    masked_ratio: float,
    max_groups: int,
    positive_topk: int,
    continuity_mode: str,
    use_continuity_selector: bool = False,
    use_activated_corr_graph: bool = False,
) -> Dict[str, float]:
    module = model.module if isinstance(model, DDP) else model
    was_training = module.training
    module.eval()
    totals = {
        "count": 0.0,
        "mgc_l1": 0.0,
        "mgc_cos": 0.0,
        "lov_global_l1": 0.0,
        "lov_global_cos": 0.0,
        "corr_recall@1": 0.0,
        "corr_recall@5": 0.0,
        "info_nce_acc": 0.0,
        "shuffle_gap": 0.0,
        "current_frame_only_gap": 0.0,
        "norm_g11": 0.0,
        "norm_g17": 0.0,
        "norm_g23": 0.0,
        "norm_c": 0.0,
        "cos(c,g11)": 0.0,
        "cos(c,g17)": 0.0,
        "cos(c,g23)": 0.0,
    }
    baseline_names = [
        "mean_predictor",
        "current_frame_only",
        "same_position",
        "feature_knn",
        "shuffled_feature_knn",
    ]
    baseline_totals = {
        name: {
            "mgc_l1": 0.0,
            "mgc_cos": 0.0,
            "lov_global_l1": 0.0,
            "lov_global_cos": 0.0,
            "shuffle_gap": 0.0,
            "current_frame_only_gap": 0.0,
        }
        for name in baseline_names
    }

    limit = len(dataset) if max_groups <= 0 else min(len(dataset), max_groups)
    for index in range(limit):
        sample = stage1_v2_collate_fn([dataset[index]])
        batch = move_batch_to_device(sample, device)
        projected, _, valid_patch_mask = module.encode_batch(batch)
        z = module.fuse_projected(projected)
        targets = prepare_targets(projected)
        valid_token_mask = batch["valid_patch_mask"] if batch["valid_patch_mask"] is not None else valid_patch_mask
        corr_graph = batch["corr_graph"]
        selector_output = None
        edge_activation = None
        if use_continuity_selector:
            selector_output = module.continuity_selector(z.detach(), corr_graph["neighbor_indices"], corr_graph["neighbor_scores"], valid_token_mask)
        if use_activated_corr_graph and selector_output is not None:
            edge_activation = module.activated_corr_graph(
                corr_graph["neighbor_indices"],
                corr_graph["neighbor_scores"],
                selector_output["probs"].detach(),
            )["activation"]
        mgc_mask = build_random_patch_mask(valid_token_mask, masked_ratio=masked_ratio)
        holdout_frames = [0]

        masked_z = z.clone()
        masked_z[mgc_mask] = module.mask_token.to(dtype=masked_z.dtype)
        continuity, aux = module.continuity_builder.forward_from_fused(
            masked_z,
            mode=continuity_mode,
            corr_neighbor_indices=corr_graph["neighbor_indices"],
            corr_neighbor_scores=corr_graph["neighbor_scores"],
            corr_edge_activation=edge_activation,
            return_aux=True,
        )
        mgc_metrics = reconstruction_metrics(
            module.geometry_decoder(continuity),
            targets,
            mgc_mask,
            layer_names=module.feature_layers,
        )

        lov_z = z.clone()
        lov_z[:, 0] = module.mask_token.to(dtype=lov_z.dtype)
        lov_continuity, _ = module.continuity_builder.forward_from_fused(
            lov_z,
            mode=continuity_mode,
            corr_neighbor_indices=corr_graph["neighbor_indices"],
            corr_neighbor_scores=corr_graph["neighbor_scores"],
            corr_edge_activation=edge_activation,
            return_aux=True,
        )
        lov_pred = module.lov_global_head(
            pool_frame_tokens(lov_continuity[:, 0:1], valid_token_mask[:, 0:1])[:, 0]
        )
        lov_target = pool_frame_tokens(z[:, 0:1].detach(), valid_token_mask[:, 0:1])[:, 0]
        lov_metrics_dict = lov_global_metrics(lov_pred, lov_target)

        shuffled_perm = torch.stack([sample_z[torch.randperm(sample_z.shape[0])] for sample_z in masked_z], dim=0)
        shuffled_continuity, _ = module.continuity_builder.forward_from_fused(
            shuffled_perm,
            mode=continuity_mode,
            corr_neighbor_indices=corr_graph["neighbor_indices"],
            corr_neighbor_scores=corr_graph["neighbor_scores"],
            corr_edge_activation=edge_activation,
            return_aux=True,
        )
        shuffled_metrics = reconstruction_metrics(
            module.geometry_decoder(shuffled_continuity),
            targets,
            mgc_mask,
            layer_names=module.feature_layers,
        )
        shuffle_gap = float(mgc_metrics["cos"].item() - shuffled_metrics["cos"].item())

        current_frame_only = run_baseline_eval(
            module,
            z,
            targets,
            valid_token_mask,
            corr_graph,
            holdout_frames,
            mgc_mask=mgc_mask,
            positive_topk=positive_topk,
            baseline_name="current_frame_only",
            use_continuity_selector=use_continuity_selector,
            use_activated_corr_graph=use_activated_corr_graph,
        )
        full_metrics = {
            "mgc_l1": float(mgc_metrics["l1"].item()),
            "mgc_cos": float(mgc_metrics["cos"].item()),
            "lov_global_l1": float(lov_metrics_dict["l1"].item()),
            "lov_global_cos": float(lov_metrics_dict["cos"].item()),
            "shuffle_gap": shuffle_gap,
        }
        current_gap = score_proxy(full_metrics) - score_proxy(current_frame_only)
        recalls = corr_recall_from_attention(aux["attention_weights"], aux["neighbor_valid_mask"], positive_topk=positive_topk)

        totals["count"] += 1.0
        totals["mgc_l1"] += full_metrics["mgc_l1"]
        totals["mgc_cos"] += full_metrics["mgc_cos"]
        totals["lov_global_l1"] += full_metrics["lov_global_l1"]
        totals["lov_global_cos"] += full_metrics["lov_global_cos"]
        totals["corr_recall@1"] += float(recalls["recall@1"].item())
        totals["corr_recall@5"] += float(recalls["recall@5"].item())
        totals["info_nce_acc"] += 0.0
        totals["shuffle_gap"] += shuffle_gap
        totals["current_frame_only_gap"] += current_gap
        totals["norm_g11"] += norm_or_zero(projected, "g11", z)
        totals["norm_g17"] += norm_or_zero(projected, "g17", z)
        totals["norm_g23"] += norm_or_zero(projected, "g23", z)
        totals["norm_c"] += float(continuity.norm(dim=-1).mean().item())
        totals["cos(c,g11)"] += cos_or_zero(continuity, projected, "g11", valid_token_mask)
        totals["cos(c,g17)"] += cos_or_zero(continuity, projected, "g17", valid_token_mask)
        totals["cos(c,g23)"] += cos_or_zero(continuity, projected, "g23", valid_token_mask)

        for baseline_name in baseline_names:
            baseline = current_frame_only if baseline_name == "current_frame_only" else run_baseline_eval(
                module,
                z,
                targets,
                valid_token_mask,
                corr_graph,
                holdout_frames,
                mgc_mask=mgc_mask,
                positive_topk=positive_topk,
                baseline_name=baseline_name,
                use_continuity_selector=use_continuity_selector,
                use_activated_corr_graph=use_activated_corr_graph,
            )
            baseline["current_frame_only_gap"] = score_proxy(baseline) - score_proxy(current_frame_only)
            for key in baseline_totals[baseline_name]:
                baseline_totals[baseline_name][key] += baseline[key]

    count = max(totals.pop("count"), 1.0)
    metrics = {key: value / count for key, value in totals.items()}
    metrics["baselines"] = {
        name: {key: value / count for key, value in baseline_totals[name].items()}
        for name in baseline_names
    }
    if was_training:
        module.train()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--geometry_encoder_path", type=str, required=True)
    parser.add_argument("--geometry_cache_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--max_groups", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d_geom", type=int, default=1024)
    parser.add_argument("--feature_layers", type=str, default="g11,g17,g23")
    parser.add_argument("--continuity_radius", type=int, default=1)
    parser.add_argument("--continuity_use_spatial_neighbors", type=str, default="False")
    parser.add_argument("--continuity_mlp_hidden_ratio", type=float, default=2.0)
    parser.add_argument("--continuity_attention_heads", type=int, default=4)
    parser.add_argument("--continuity_mode", type=str, default="corr_graph", choices=("same_position", "corr_graph"))
    parser.add_argument("--corr_score_beta", type=float, default=1.0)
    parser.add_argument("--time_bias_init", type=float, default=-0.10)
    parser.add_argument("--use_continuity_selector", type=str, default="False")
    parser.add_argument("--use_activated_corr_graph", type=str, default="False")
    parser.add_argument("--cus_loss_weight", type=float, default=0.0)
    parser.add_argument("--cus_budget_ratio", type=float, default=0.20)
    parser.add_argument("--cus_budget_weight", type=float, default=0.10)
    parser.add_argument("--cus_target_temperature", type=float, default=0.10)
    parser.add_argument("--masked_ratio", type=float, default=0.20)
    parser.add_argument("--mgc_cosine_weight", type=float, default=1.0)
    parser.add_argument("--mgc_l1_weight", type=float, default=0.2)
    parser.add_argument("--corr_nce_weight", type=float, default=0.5)
    parser.add_argument("--attn_weight", type=float, default=0.05)
    parser.add_argument("--lov_global_weight", type=float, default=0.2)
    parser.add_argument("--var_weight", type=float, default=0.01)
    parser.add_argument("--variance_gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_negatives", type=int, default=64)
    parser.add_argument("--positive_topk", type=int, default=3)
    parser.add_argument("--max_contrastive_anchors", type=int, default=256)
    parser.add_argument("--random_patch_prob", type=float, default=0.20)
    parser.add_argument("--corr_tube_prob", type=float, default=0.50)
    parser.add_argument("--frame_block_prob", type=float, default=0.30)
    parser.add_argument("--random_patch_prob_phase3", type=float, default=0.10)
    parser.add_argument("--corr_tube_prob_phase3", type=float, default=0.55)
    parser.add_argument("--frame_block_prob_phase3", type=float, default=0.35)
    parser.add_argument("--phase0_steps", type=int, default=1000)
    parser.add_argument("--phase1_steps", type=int, default=5000)
    parser.add_argument("--phase3_start_step", type=int, default=15000)
    parser.add_argument("--geometry_cache_required", type=str, default="True")
    parser.add_argument("--corr_cache_required", type=str, default="True")
    parser.add_argument("--online_fallback", type=str, default="False")
    parser.add_argument("--freeze_geo_projector", type=str, default="False")
    parser.add_argument("--group_windows_by_source", type=str, default="True")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--memory_cache_size", type=int, default=8)
    parser.add_argument("--persistent_workers", type=str, default="True")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--eval_only", type=str, default="False")
    parser.add_argument("--eval_output_path", type=str, default="")
    parser.add_argument("--eval_checkpoint_path", type=str, default="")
    parser.add_argument("--eval_max_groups", type=int, default=-1)
    parser.add_argument("--resume_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_model_only", type=str, default="False")
    parser.add_argument("--reset_scheduler_on_resume", type=str, default="False")
    parser.add_argument("--init_checkpoint_path", type=str, default="")
    args = parser.parse_args()

    rank, world_size, device = init_distributed()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    image_processor = AutoProcessor.from_pretrained(args.model_name_or_path).image_processor
    feature_layers = parse_feature_layers(args.feature_layers)
    dataset = Stage1GeometryDatasetV2(
        manifest_path=args.geometry_cache_manifest,
        image_processor=image_processor,
        geometry_cache_required=parse_bool(args.geometry_cache_required),
        corr_cache_required=parse_bool(args.corr_cache_required),
        online_fallback=parse_bool(args.online_fallback),
        max_groups=args.max_groups,
        memory_cache_size=args.memory_cache_size,
    )
    use_grouped_batches = parse_bool(args.group_windows_by_source)
    batch_sampler = None
    if use_grouped_batches:
        batch_sampler = Stage1SourceGroupedBatchSampler(
            dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            drop_last=False,
            world_size=world_size,
            rank=rank,
            seed=args.seed,
        )
    dataloader_kwargs = {
        "dataset": dataset,
        "collate_fn": stage1_v2_collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = parse_bool(args.persistent_workers)
        dataloader_kwargs["prefetch_factor"] = max(int(args.prefetch_factor), 2)
    if batch_sampler is not None:
        dataloader_kwargs["batch_sampler"] = batch_sampler
    else:
        dataloader_kwargs["batch_size"] = args.per_device_train_batch_size
        dataloader_kwargs["shuffle"] = True
    dataloader = DataLoader(**dataloader_kwargs)

    model = Stage1ContinuityModelV2(
        geometry_encoder_path=args.geometry_encoder_path,
        feature_layers=feature_layers,
        d_geom=args.d_geom,
        continuity_radius=args.continuity_radius,
        continuity_use_spatial_neighbors=parse_bool(args.continuity_use_spatial_neighbors),
        continuity_mlp_hidden_ratio=args.continuity_mlp_hidden_ratio,
        continuity_attention_heads=args.continuity_attention_heads,
        corr_score_beta=args.corr_score_beta,
        time_bias_init=args.time_bias_init,
    ).to(device)
    if args.init_checkpoint_path and os.path.exists(args.init_checkpoint_path):
        model.initialize_from_checkpoint(args.init_checkpoint_path, device)
        rank0_print(f"Initialized Stage 1 v2 from checkpoint: {args.init_checkpoint_path}")
    if dataset.uses_projected_cache or parse_bool(args.freeze_geo_projector):
        for parameter in model.geo_projector.parameters():
            parameter.requires_grad = False
        rank0_print("Stage 1 v2 geo_projector frozen because projected compact cache is active.")
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)

    if parse_bool(args.eval_only):
        checkpoint_path = args.eval_checkpoint_path or os.path.join(args.output_dir, "latest.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Stage 1 v2 eval checkpoint not found: {checkpoint_path}")
        payload = load_stage1_v2_checkpoint(model, checkpoint_path, device)
        metrics = evaluate_stage1_v2(
            model,
            dataset,
            device,
            masked_ratio=args.masked_ratio,
            max_groups=args.eval_max_groups if args.eval_max_groups > 0 else args.max_groups,
            positive_topk=args.positive_topk,
            continuity_mode=args.continuity_mode,
            use_continuity_selector=parse_bool(args.use_continuity_selector),
            use_activated_corr_graph=parse_bool(args.use_activated_corr_graph),
        )
        metrics["checkpoint_path"] = checkpoint_path
        metrics["step"] = int(payload.get("step", 0))
        metrics["metric_semantics"] = "similarity"
        if not is_dist() or dist.get_rank() == 0:
            output_path = args.eval_output_path or os.path.join(args.log_dir, "stage1_v2_eval.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, ensure_ascii=False, indent=2)
            rank0_print(json.dumps(metrics, ensure_ascii=False))
        if is_dist():
            dist.barrier()
            dist.destroy_process_group()
        return

    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(args.max_steps * max(args.warmup_ratio, 0.0))

    def lr_lambda(current_step: int) -> float:
        if args.max_steps <= 1:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(warmup_steps, 1))
        if args.max_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(args.max_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    step = 0
    epoch = 0
    last_log: Dict = {}
    resume_checkpoint_path = (args.resume_checkpoint_path or "").strip()
    if resume_checkpoint_path:
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(f"Stage 1 v2 resume checkpoint not found: {resume_checkpoint_path}")
        payload = load_stage1_v2_checkpoint(model, resume_checkpoint_path, device)
        if parse_bool(args.resume_model_only):
            step = 0
            last_log = {}
            epoch = 0
            rank0_print(f"Initialized Stage 1 v2 model-only from checkpoint: {resume_checkpoint_path}")
        else:
            optimizer_state = payload.get("optimizer")
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
                move_optimizer_state_to_device(optimizer, device)
            scheduler_state = None if parse_bool(args.reset_scheduler_on_resume) else payload.get("scheduler")
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            else:
                resumed_step = int(payload.get("step", 0))
                for _ in range(resumed_step):
                    scheduler.step()
            step = int(payload.get("step", 0))
            last_log = dict(payload.get("extra", {}) or {})
            if len(dataloader) > 0:
                epoch = step // max(len(dataloader), 1)
            rank0_print(f"Resumed Stage 1 v2 from checkpoint: {resume_checkpoint_path} @ step {step}")
    if step >= args.max_steps:
        rank0_print(f"Stage 1 v2 already reached max_steps={args.max_steps}; nothing to do.")
        if is_dist():
            dist.barrier()
            dist.destroy_process_group()
        return

    if batch_sampler is not None:
        batch_sampler.set_epoch(epoch)
    dataloader_iter = iter(dataloader)
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.tensor(0.0, device=device)
        last_outputs = None
        for _ in range(max(args.gradient_accumulation_steps, 1)):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                epoch += 1
                if batch_sampler is not None:
                    batch_sampler.set_epoch(epoch)
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            outputs = compute_stage1_v2_batch(model, batch, args, step)
            (outputs["total_loss"] / max(args.gradient_accumulation_steps, 1)).backward()
            accum_loss = accum_loss + outputs["total_loss"].detach()
            last_outputs = outputs

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.logging_steps == 0 and last_outputs is not None:
            current_frame_only = evaluate_stage1_v2(
                model,
                dataset,
                device,
                masked_ratio=args.masked_ratio,
                max_groups=1,
                positive_topk=args.positive_topk,
                continuity_mode="current_frame_only",
                use_continuity_selector=parse_bool(args.use_continuity_selector),
                use_activated_corr_graph=parse_bool(args.use_activated_corr_graph),
            )
            log_payload = {
                "step": step,
                "loss": float((accum_loss / max(args.gradient_accumulation_steps, 1)).item()),
                "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "mgc_l1": float(last_outputs["mgc_metrics"]["l1"].item()),
                "mgc_cos": float(last_outputs["mgc_metrics"]["cos"].item()),
                "lov_global_l1": float(last_outputs["lov_global_metrics"]["l1"].item()),
                "lov_global_cos": float(last_outputs["lov_global_metrics"]["cos"].item()),
                "corr_recall@1": float(last_outputs["corr_recall"]["recall@1"].item()),
                "corr_recall@5": float(last_outputs["corr_recall"]["recall@5"].item()),
                "info_nce_acc": float(last_outputs["info_nce_acc"].item()),
                "corr_nce_loss": float(last_outputs["contrastive_loss"].item()),
                "attn_loss": float(last_outputs["attn_loss"].item()),
                "var_loss": float(last_outputs["var_loss"].item()),
                "selector_loss": float(last_outputs["selector_loss"].item()),
                "selector_bce": float(last_outputs["selector_bce"].item()),
                "selector_budget": float(last_outputs["selector_budget"].item()),
                "selector_mean_budget": float(last_outputs["selector_mean_budget"].item()),
                "shuffle_gap": float(last_outputs["shuffle_gap"]),
                "question_type": last_outputs["question_type"],
                "phase": last_outputs["phase"],
                "mask_mode_histogram": last_outputs["mask_mode_histogram"],
                "norm_g11": last_outputs["norm_g11"],
                "norm_g17": last_outputs["norm_g17"],
                "norm_g23": last_outputs["norm_g23"],
                "norm_c": last_outputs["norm_c"],
                "cos(c,g11)": last_outputs["cos_c_g11"],
                "cos(c,g17)": last_outputs["cos_c_g17"],
                "cos(c,g23)": last_outputs["cos_c_g23"],
                "saliency_mean": last_outputs["saliency_mean"],
                "utility_target_mean": last_outputs["utility_target_mean"],
                "edge_activation_mean": last_outputs["edge_activation_mean"],
                "current_frame_only_gap": float(
                    (last_outputs["mgc_metrics"]["cos"].item() + last_outputs["lov_global_metrics"]["cos"].item() + max(last_outputs["shuffle_gap"], 0.0))
                    - (current_frame_only["mgc_cos"] + current_frame_only["lov_global_cos"] + max(current_frame_only["shuffle_gap"], 0.0))
                ),
            }
            last_log = log_payload
            rank0_print(json.dumps(log_payload, ensure_ascii=False))
            if rank == 0:
                mgc_path = os.path.join(args.log_dir, f"mgc_step{step}.json")
                lov_path = os.path.join(args.log_dir, f"lov_step{step}.json")
                shuffle_path = os.path.join(args.log_dir, f"shuffle_step{step}.json")
                visualize_mgc(
                    mgc_path,
                    masked_ratio=args.masked_ratio,
                    question_type=last_outputs["question_type"],
                    loss_dict={
                        "mgc_l1": log_payload["mgc_l1"],
                        "mgc_cos": log_payload["mgc_cos"],
                        "corr_recall@5": log_payload["corr_recall@5"],
                    },
                )
                visualize_lov(
                    lov_path,
                    heldout_frame=last_outputs["holdout_frames"][0] if last_outputs["holdout_frames"] else 0,
                    question_type=last_outputs["question_type"],
                    loss_dict={
                        "lov_global_l1": log_payload["lov_global_l1"],
                        "lov_global_cos": log_payload["lov_global_cos"],
                    },
                )
                visualize_shuffle_ablation(
                    shuffle_path,
                    question_type=last_outputs["question_type"],
                    original_loss=log_payload["mgc_cos"],
                    shuffled_loss=log_payload["mgc_cos"] - log_payload["shuffle_gap"],
                )

        if step % args.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, step, args.output_dir, extra=last_log)

    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, step, args.output_dir, extra=last_log)
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

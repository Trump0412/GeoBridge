"""Stage 1 continuity pretraining for ZenView continuity-bank v2."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qwen_vl.model.geometry_bank import (
    BaseGeometryFusion,
    ContinuityBuilder,
    GeoProjector,
    GeometryDecoder,
    VGGTBankExtractor,
    VGGTBankFeatureOutput,
    geometry_reconstruction_loss,
)
from qwen_vl.train.stage1_geometry import (
    Stage1SourceGroupedBatchSampler,
    Stage1GeometryDataset,
    stage1_collate_fn,
    visualize_lov,
    visualize_mgc,
    visualize_shuffle_ablation,
)


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


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


class Stage1ContinuityModel(nn.Module):
    def __init__(
        self,
        geometry_encoder_path: str,
        d_geom: int = 1024,
        continuity_radius: int = 1,
        continuity_use_spatial_neighbors: bool = False,
        continuity_mlp_hidden_ratio: float = 2.0,
        continuity_attention_heads: int = 4,
    ):
        super().__init__()
        self.geometry_encoder = VGGTBankExtractor(
            model_path=geometry_encoder_path,
            layer_ids=(11, 17, 23),
            freeze_encoder=True,
        )
        self.geo_projector = GeoProjector(
            input_dims={name: 8192 for name in ("g11_raw", "g17_raw", "g23_raw")},
            d_geom=d_geom,
        )
        self.base_geometry_fusion = BaseGeometryFusion(d_geom=d_geom, hidden_ratio=continuity_mlp_hidden_ratio)
        self.continuity_builder = ContinuityBuilder(
            d_geom=d_geom,
            radius_t=continuity_radius,
            use_spatial_neighbors=continuity_use_spatial_neighbors,
            mlp_hidden_ratio=continuity_mlp_hidden_ratio,
            attention_heads=continuity_attention_heads,
        )
        self.geometry_decoder = GeometryDecoder(d_geom=d_geom, hidden_ratio=continuity_mlp_hidden_ratio)
        self.mask_token = nn.Parameter(torch.zeros(d_geom))
        nn.init.normal_(self.mask_token, std=0.02)

        for parameter in self.geometry_encoder.parameters():
            parameter.requires_grad = False

    def _feature_output_from_cache(self, cached_features: Dict) -> VGGTBankFeatureOutput:
        return VGGTBankFeatureOutput(
            layer_tokens={
                "g11_raw": cached_features["g11_raw"],
                "g17_raw": cached_features["g17_raw"],
                "g23_raw": cached_features["g23_raw"],
            },
            frame_layout=type("FrameLayoutLike", (), {
                "token_counts": [int(v) for v in cached_features["token_counts"]],
                "frame_shapes": [tuple(int(x) for x in shape) for shape in cached_features["frame_shapes"]],
            })(),
            patch_grid=tuple(int(v) for v in cached_features.get("patch_grid", cached_features["frame_shapes"][0])),
            merged_grid=tuple(int(v) for v in cached_features.get("merged_grid", cached_features["frame_shapes"][0])),
        )

    def encode(self, sample: Dict) -> tuple[Dict[str, torch.Tensor], List[tuple[int, int]]]:
        if sample["cached_features"] is not None:
            extracted = self._feature_output_from_cache(sample["cached_features"])
        else:
            extracted = self.geometry_encoder.extract(sample["geometry_inputs"])
        projected = self.geo_projector(extracted.layer_tokens)
        return projected, list(extracted.frame_layout.frame_shapes)

    def encode_batch(self, batch: Dict) -> tuple[Dict[str, torch.Tensor], Sequence]:
        if batch["cached_features"] is not None:
            projected = self.geo_projector(
                {
                    "g11_raw": batch["cached_features"]["g11_raw"],
                    "g17_raw": batch["cached_features"]["g17_raw"],
                    "g23_raw": batch["cached_features"]["g23_raw"],
                }
            )
            return projected, batch["frame_shapes"]

        if batch["geometry_inputs"] is None:
            raise ValueError("Stage 1 batch must provide cached_features or geometry_inputs")

        geometry_inputs = batch["geometry_inputs"]
        valid_frame_mask = batch["valid_frame_mask"]
        projected_outputs = []
        frame_shapes = []
        for batch_idx, sample_inputs_all in enumerate(geometry_inputs):
            sample_inputs = sample_inputs_all[valid_frame_mask[batch_idx][: sample_inputs_all.shape[0]]]
            extracted = self.geometry_encoder.extract(sample_inputs)
            projected_outputs.append(self.geo_projector(extracted.layer_tokens))
            frame_shapes.append(list(extracted.frame_layout.frame_shapes))

        max_frames = max(output["g11"].shape[0] for output in projected_outputs)
        max_patches = max(output["g11"].shape[1] for output in projected_outputs)
        batched = {}
        batch_size = len(geometry_inputs) if isinstance(geometry_inputs, list) else geometry_inputs.shape[0]
        for key in ("g11", "g17", "g23"):
            feature_dim = projected_outputs[0][key].shape[-1]
            output_tensor = projected_outputs[0][key].new_zeros(
                batch_size, max_frames, max_patches, feature_dim
            )
            for batch_idx, output in enumerate(projected_outputs):
                num_frames, num_patches = output[key].shape[:2]
                output_tensor[batch_idx, :num_frames, :num_patches] = output[key]
            batched[key] = output_tensor
        return batched, frame_shapes


def curriculum_lov_weight(progress: float) -> float:
    if progress < 0.10:
        return 0.0
    if progress < 0.40:
        return 0.3
    return 1.0


def build_mgc_mask(valid_token_mask: torch.Tensor, masked_ratio: float) -> torch.Tensor:
    sampled = torch.rand_like(valid_token_mask.float()) < masked_ratio
    mask = valid_token_mask & sampled
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


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = dict(batch)
    moved["valid_frame_mask"] = batch["valid_frame_mask"].to(device)
    moved["valid_patch_mask"] = (
        batch["valid_patch_mask"].to(device) if isinstance(batch["valid_patch_mask"], torch.Tensor) else None
    )
    if batch["cached_features"] is not None:
        moved["cached_features"] = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch["cached_features"].items()
        }
    if batch["geometry_inputs"] is not None:
        if isinstance(batch["geometry_inputs"], torch.Tensor):
            moved["geometry_inputs"] = batch["geometry_inputs"].to(device)
        else:
            moved["geometry_inputs"] = [inputs.to(device) for inputs in batch["geometry_inputs"]]
    return moved


def prepare_targets(projected: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in projected.items()}


def batch_question_type(question_types: Sequence[str]) -> str:
    filtered = [question_type for question_type in question_types if isinstance(question_type, str) and question_type]
    if not filtered:
        return "unknown"
    counter = collections.Counter(filtered)
    if len(counter) == 1:
        return filtered[0]
    return "mixed"


def run_shuffle_ablation(model: Stage1ContinuityModel, z: torch.Tensor, frame_shapes, targets, valid_token_mask):
    with torch.no_grad():
        original = model.continuity_builder.forward_from_fused(z, frame_shapes=frame_shapes)
        original_loss = geometry_reconstruction_loss(model.geometry_decoder(original), targets, valid_token_mask)
        if z.dim() == 4:
            shuffled_z = torch.stack([sample[torch.randperm(sample.shape[0])] for sample in z], dim=0)
        else:
            shuffled_z = z[torch.randperm(z.shape[0])]
        shuffled = model.continuity_builder.forward_from_fused(shuffled_z, frame_shapes=frame_shapes)
        shuffled_loss = geometry_reconstruction_loss(model.geometry_decoder(shuffled), targets, valid_token_mask)
        return float(original_loss["total"].item()), float(shuffled_loss["total"].item())


def compute_stage1_batch(
    model: nn.Module,
    batch: Dict,
    *,
    masked_ratio: float,
    lov_weight: float,
) -> Dict:
    module = model.module if isinstance(model, DDP) else model
    batch = move_batch_to_device(batch, next(module.parameters()).device)
    projected, frame_shapes = module.encode_batch(batch)
    z = module.base_geometry_fusion(projected["g11"], projected["g17"], projected["g23"])
    targets = prepare_targets(projected)
    valid_frame_mask = batch["valid_frame_mask"]
    valid_token_mask = batch["valid_patch_mask"]
    if valid_token_mask is None or valid_token_mask.shape[-1] == 1:
        valid_token_mask = valid_frame_mask[:, :, None].expand(-1, -1, z.shape[2])

    mgc_mask = build_mgc_mask(valid_token_mask, masked_ratio)
    masked_z = z.clone()
    masked_z[mgc_mask] = module.mask_token.to(dtype=masked_z.dtype)
    continuity_mgc = module.continuity_builder.forward_from_fused(masked_z, frame_shapes=frame_shapes)
    mgc_predictions = module.geometry_decoder(continuity_mgc)
    mgc_losses = geometry_reconstruction_loss(mgc_predictions, targets, mgc_mask)

    lov_z = z.clone()
    lov_mask = torch.zeros_like(valid_token_mask)
    heldout_frames: List[int] = []
    for batch_idx in range(valid_frame_mask.shape[0]):
        valid_frame_indices = valid_frame_mask[batch_idx].nonzero(as_tuple=False).view(-1)
        heldout_frame = int(valid_frame_indices[torch.randint(len(valid_frame_indices), (1,), device=valid_frame_mask.device)].item())
        heldout_frames.append(heldout_frame)
        lov_z[batch_idx, heldout_frame] = module.mask_token.to(dtype=lov_z.dtype)
        lov_mask[batch_idx, heldout_frame] = valid_token_mask[batch_idx, heldout_frame]

    continuity_lov = module.continuity_builder.forward_from_fused(lov_z, frame_shapes=frame_shapes)
    lov_predictions = module.geometry_decoder(continuity_lov)
    lov_losses = geometry_reconstruction_loss(lov_predictions, targets, lov_mask)
    total_loss = mgc_losses["total"] + lov_weight * lov_losses["total"]

    return {
        "total_loss": total_loss,
        "mgc_losses": mgc_losses,
        "lov_losses": lov_losses,
        "heldout_frames": heldout_frames,
        "question_type": batch_question_type(batch["question_type"]),
        "z": z,
        "frame_shapes": frame_shapes,
        "targets": targets,
        "valid_token_mask": valid_token_mask,
        "mgc_continuity_norm": float(continuity_mgc.norm(dim=-1).mean().detach().item()),
        "lov_continuity_norm": float(continuity_lov.norm(dim=-1).mean().detach().item()),
        "z_norm": float(z.norm(dim=-1).mean().detach().item()),
    }


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
        "continuity_builder": module.continuity_builder,
        "geometry_decoder": module.geometry_decoder,
    }.items():
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


def load_stage1_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model", payload)
    module = model.module if isinstance(model, DDP) else model
    module.geo_projector.load_state_dict(
        {key.removeprefix("geo_projector."): value for key, value in state_dict.items() if key.startswith("geo_projector.")}
    )
    module.base_geometry_fusion.load_state_dict(
        {
            key.removeprefix("base_geometry_fusion."): value
            for key, value in state_dict.items()
            if key.startswith("base_geometry_fusion.")
        }
    )
    module.continuity_builder.load_state_dict(
        {
            key.removeprefix("continuity_builder."): value
            for key, value in state_dict.items()
            if key.startswith("continuity_builder.")
        }
    )
    module.geometry_decoder.load_state_dict(
        {
            key.removeprefix("geometry_decoder."): value
            for key, value in state_dict.items()
            if key.startswith("geometry_decoder.")
        }
    )
    if "mask_token" in state_dict:
        module.mask_token.data.copy_(state_dict["mask_token"].to(device=device, dtype=module.mask_token.dtype))
    return payload


@torch.no_grad()
def evaluate_stage1(
    model: nn.Module,
    dataset: Stage1GeometryDataset,
    device: torch.device,
    *,
    masked_ratio: float,
    max_groups: int = -1,
) -> Dict[str, float]:
    module = model.module if isinstance(model, DDP) else model
    was_training = module.training
    module.eval()
    totals = {
        "count": 0.0,
        "mgc_l1": 0.0,
        "mgc_cos": 0.0,
        "lov_l1": 0.0,
        "lov_cos": 0.0,
        "shuffle_original": 0.0,
        "shuffle_shuffled": 0.0,
    }
    limit = len(dataset) if max_groups <= 0 else min(len(dataset), max_groups)
    for index in range(limit):
        sample = dataset[index]
        valid_frame_mask = sample["valid_frame_mask"].to(device)
        if sample["cached_features"] is not None:
            sample["cached_features"] = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in sample["cached_features"].items()
            }
        if sample["geometry_inputs"] is not None:
            sample["geometry_inputs"] = sample["geometry_inputs"].to(device)

        projected, frame_shapes = module.encode(sample)
        z = module.base_geometry_fusion(projected["g11"], projected["g17"], projected["g23"])
        targets = prepare_targets(projected)
        valid_token_mask = valid_frame_mask[:, None].expand(-1, z.shape[1])

        mgc_mask = build_mgc_mask(valid_token_mask, masked_ratio)
        masked_z = z.clone()
        masked_z[mgc_mask] = module.mask_token.to(dtype=masked_z.dtype)
        continuity_mgc = module.continuity_builder.forward_from_fused(masked_z, frame_shapes=frame_shapes)
        mgc_predictions = module.geometry_decoder(continuity_mgc)
        mgc_losses = geometry_reconstruction_loss(mgc_predictions, targets, mgc_mask)

        valid_frame_indices = valid_frame_mask.nonzero(as_tuple=False).view(-1)
        heldout_frame = int(valid_frame_indices[0].item())
        lov_z = z.clone()
        lov_z[heldout_frame] = module.mask_token.to(dtype=lov_z.dtype)
        continuity_lov = module.continuity_builder.forward_from_fused(lov_z, frame_shapes=frame_shapes)
        lov_predictions = module.geometry_decoder(continuity_lov)
        lov_mask = torch.zeros_like(mgc_mask)
        lov_mask[heldout_frame] = valid_token_mask[heldout_frame]
        lov_losses = geometry_reconstruction_loss(lov_predictions, targets, lov_mask)

        shuffle_original, shuffle_shuffled = run_shuffle_ablation(module, z, frame_shapes, targets, valid_token_mask)
        totals["count"] += 1.0
        totals["mgc_l1"] += float(mgc_losses["l1"].item())
        totals["mgc_cos"] += float(mgc_losses["cos"].item())
        totals["lov_l1"] += float(lov_losses["l1"].item())
        totals["lov_cos"] += float(lov_losses["cos"].item())
        totals["shuffle_original"] += float(shuffle_original)
        totals["shuffle_shuffled"] += float(shuffle_shuffled)

    if was_training:
        module.train()
    count = max(totals.pop("count"), 1.0)
    return {key: value / count for key, value in totals.items()}


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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d_geom", type=int, default=1024)
    parser.add_argument("--continuity_radius", type=int, default=1)
    parser.add_argument("--continuity_use_spatial_neighbors", type=str, default="False")
    parser.add_argument("--continuity_mlp_hidden_ratio", type=float, default=2.0)
    parser.add_argument("--continuity_attention_heads", type=int, default=4)
    parser.add_argument("--masked_ratio", type=float, default=0.20)
    parser.add_argument("--geometry_cache_required", type=str, default="False")
    parser.add_argument("--online_fallback", type=str, default="True")
    parser.add_argument("--group_windows_by_source", type=str, default="True")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_only", type=str, default="False")
    parser.add_argument("--eval_output_path", type=str, default="")
    parser.add_argument("--eval_checkpoint_path", type=str, default="")
    parser.add_argument("--eval_max_groups", type=int, default=-1)
    parser.add_argument("--resume_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_model_only", type=str, default="False")
    args = parser.parse_args()

    rank, world_size, device = init_distributed()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    image_processor = AutoProcessor.from_pretrained(args.model_name_or_path).image_processor
    dataset = Stage1GeometryDataset(
        manifest_path=args.geometry_cache_manifest,
        image_processor=image_processor,
        geometry_cache_required=parse_bool(args.geometry_cache_required),
        online_fallback=parse_bool(args.online_fallback),
        max_groups=args.max_groups,
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
        "collate_fn": stage1_collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if batch_sampler is not None:
        dataloader_kwargs["batch_sampler"] = batch_sampler
    else:
        dataloader_kwargs["batch_size"] = args.per_device_train_batch_size
        dataloader_kwargs["shuffle"] = True
    dataloader = DataLoader(**dataloader_kwargs)

    model = Stage1ContinuityModel(
        geometry_encoder_path=args.geometry_encoder_path,
        d_geom=args.d_geom,
        continuity_radius=args.continuity_radius,
        continuity_use_spatial_neighbors=parse_bool(args.continuity_use_spatial_neighbors),
        continuity_mlp_hidden_ratio=args.continuity_mlp_hidden_ratio,
        continuity_attention_heads=args.continuity_attention_heads,
    ).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)

    if parse_bool(args.eval_only):
        checkpoint_path = args.eval_checkpoint_path or os.path.join(args.output_dir, "latest.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Stage 1 eval checkpoint not found: {checkpoint_path}")
        payload = load_stage1_checkpoint(model, checkpoint_path, device)
        metrics = evaluate_stage1(
            model,
            dataset,
            device,
            masked_ratio=args.masked_ratio,
            max_groups=args.eval_max_groups if args.eval_max_groups > 0 else args.max_groups,
        )
        metrics["checkpoint_path"] = checkpoint_path
        metrics["step"] = int(payload.get("step", 0))
        if not is_dist() or dist.get_rank() == 0:
            output_path = args.eval_output_path or os.path.join(args.log_dir, "stage1_eval.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, ensure_ascii=False, indent=2)
            rank0_print(json.dumps(metrics, ensure_ascii=False))
        if is_dist():
            dist.barrier()
            dist.destroy_process_group()
        return

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=False)

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
            raise FileNotFoundError(f"Stage 1 resume checkpoint not found: {resume_checkpoint_path}")
        payload = load_stage1_checkpoint(model, resume_checkpoint_path, device)
        if parse_bool(args.resume_model_only):
            step = 0
            last_log = {}
            epoch = 0
            rank0_print(f"Initialized Stage 1 model-only from checkpoint: {resume_checkpoint_path}")
        else:
            optimizer_state = payload.get("optimizer")
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
                move_optimizer_state_to_device(optimizer, device)
            scheduler_state = payload.get("scheduler")
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
            rank0_print(f"Resumed Stage 1 from checkpoint: {resume_checkpoint_path} @ step {step}")
    if step >= args.max_steps:
        rank0_print(f"Stage 1 already reached max_steps={args.max_steps}; nothing to do.")
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
        for _ in range(max(args.gradient_accumulation_steps, 1)):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                epoch += 1
                if batch_sampler is not None:
                    batch_sampler.set_epoch(epoch)
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            progress = step / max(args.max_steps, 1)
            lov_weight = curriculum_lov_weight(progress)
            batch_outputs = compute_stage1_batch(
                model,
                batch,
                masked_ratio=args.masked_ratio,
                lov_weight=lov_weight,
            )
            total_loss = batch_outputs["total_loss"]
            (total_loss / max(args.gradient_accumulation_steps, 1)).backward()
            accum_loss = accum_loss + total_loss.detach()
            last_log = {
                "mgc_l1": float(batch_outputs["mgc_losses"]["l1"].detach().item()),
                "mgc_cos": float(batch_outputs["mgc_losses"]["cos"].detach().item()),
                "mgc_total": float(batch_outputs["mgc_losses"]["total"].detach().item()),
                "lov_l1": float(batch_outputs["lov_losses"]["l1"].detach().item()),
                "lov_cos": float(batch_outputs["lov_losses"]["cos"].detach().item()),
                "lov_total": float(batch_outputs["lov_losses"]["total"].detach().item()),
                "heldout_frame": int(batch_outputs["heldout_frames"][0]),
                "question_type": batch_outputs["question_type"],
                "lov_weight": lov_weight,
                "batch_size": int(batch["valid_frame_mask"].shape[0]),
                "num_unique_sources": int(len(set(batch["source_sample_id"]))),
                "z_norm": float(batch_outputs["z_norm"]),
                "mgc_cont_norm": float(batch_outputs["mgc_continuity_norm"]),
                "lov_cont_norm": float(batch_outputs["lov_continuity_norm"]),
            }

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.logging_steps == 0:
            original_shuffle_loss, shuffled_shuffle_loss = run_shuffle_ablation(
                model.module if isinstance(model, DDP) else model,
                batch_outputs["z"].detach(),
                batch_outputs["frame_shapes"],
                batch_outputs["targets"],
                batch_outputs["valid_token_mask"],
            )
            if not is_dist() or dist.get_rank() == 0:
                rank0_print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(accum_loss.item() / max(args.gradient_accumulation_steps, 1)),
                            "grad_norm": float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                            "learning_rate": float(scheduler.get_last_lr()[0]),
                            **last_log,
                            "shuffle_original": original_shuffle_loss,
                            "shuffle_shuffled": shuffled_shuffle_loss,
                        },
                        ensure_ascii=False,
                    )
                )
                visualize_mgc(
                    os.path.join(args.log_dir, f"mgc_step{step}.json"),
                    masked_ratio=args.masked_ratio,
                    loss_dict={"l1": last_log["mgc_l1"], "cos": last_log["mgc_cos"]},
                    question_type=last_log["question_type"],
                )
                visualize_lov(
                    os.path.join(args.log_dir, f"lov_step{step}.json"),
                    heldout_frame=last_log["heldout_frame"],
                    loss_dict={"l1": last_log["lov_l1"], "cos": last_log["lov_cos"]},
                    question_type=last_log["question_type"],
                )
                visualize_shuffle_ablation(
                    os.path.join(args.log_dir, f"shuffle_step{step}.json"),
                    original_loss=original_shuffle_loss,
                    shuffled_loss=shuffled_shuffle_loss,
                    question_type=last_log["question_type"],
                )

        if step % args.save_steps == 0 and (not is_dist() or dist.get_rank() == 0):
            save_checkpoint(model, optimizer, scheduler, step, args.output_dir, extra=last_log)

    if not is_dist() or dist.get_rank() == 0:
        save_checkpoint(model, optimizer, scheduler, step, args.output_dir, extra=last_log)

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

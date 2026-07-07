"""Stage 1 continuity-pretraining dataset and batching helpers."""

from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from qwen_vl.data.geometry_cache import GeometryCacheIndex
from qwen_vl.data.utils import prepare_image_inputs


@dataclass
class Stage1Sample:
    group_id: str
    source_dataset: str
    source_sample_id: str
    frame_paths: List[str]
    valid_frame_mask: List[bool]
    question_type: str
    cache_path: str = ""
    window_id: str = "window_0"
    cache_window_mode: str = "fixed8"


class Stage1GeometryDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_processor,
        geometry_cache_required: bool = False,
        online_fallback: bool = True,
        max_groups: int = -1,
    ):
        super().__init__()
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Stage 1 manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        self.feature_cache_dir = os.path.join(os.path.dirname(manifest_path), "features")
        self.image_processor = image_processor
        self.geometry_cache_required = bool(geometry_cache_required)
        self.online_fallback = bool(online_fallback)
        self.cache_index = GeometryCacheIndex(manifest_path)
        entries = list(self.cache_index.items())
        if max_groups > 0:
            entries = entries[:max_groups]
        self.entries = [
            Stage1Sample(
                group_id=entry["group_id"],
                source_dataset=entry["source_dataset"],
                source_sample_id=entry.get("source_sample_id", entry["group_id"]),
                frame_paths=list(entry["frame_paths"]),
                valid_frame_mask=list(entry.get("valid_frame_mask", [True] * len(entry["frame_paths"]))),
                question_type=entry.get("question_type", "unknown"),
                cache_path=entry.get("cache_path", ""),
                window_id=entry.get("window_id", "window_0"),
                cache_window_mode=entry.get("cache_window_mode", "fixed8"),
            )
            for entry in entries
        ]
        self._source_groups: List[List[int]] = []
        source_group_map: Dict[str, List[int]] = {}
        for index, sample in enumerate(self.entries):
            source_group_map.setdefault(sample.source_sample_id, []).append(index)
        for source_sample_id in sorted(source_group_map):
            indices = sorted(source_group_map[source_sample_id], key=lambda item: self.entries[item].window_id)
            self._source_groups.append(indices)

    def __len__(self) -> int:
        return len(self.entries)

    def get_source_groups(self) -> List[List[int]]:
        return [list(group) for group in self._source_groups]

    def _resolve_cache_path(self, sample: Stage1Sample) -> str:
        if sample.cache_path:
            return sample.cache_path
        return os.path.join(self.feature_cache_dir, f"{sample.group_id}.pt")

    def _load_images(self, frame_paths: Sequence[str]) -> List[Image.Image]:
        images = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                images.append(image.convert("RGB").copy())
        if len(images) > 1:
            width, height = images[0].size
            images = [images[0]] + [
                image if image.size == (width, height) else image.resize((width, height), Image.BILINEAR)
                for image in images[1:]
            ]
        return images

    def _load_geometry_inputs(self, frame_paths: Sequence[str]) -> torch.Tensor:
        inputs = []
        for image in self._load_images(frame_paths):
            prepared = prepare_image_inputs(image, self.image_processor)
            inputs.append(prepared["geometry_encoder_inputs"])
        return torch.stack(inputs)

    def __getitem__(self, index: int) -> Dict:
        sample = self.entries[index]
        cache_payload = None
        resolved_cache_path = self._resolve_cache_path(sample)
        if resolved_cache_path and os.path.exists(resolved_cache_path):
            cache_payload = torch.load(resolved_cache_path, map_location="cpu")
        elif self.geometry_cache_required and not self.online_fallback:
            raise FileNotFoundError(f"Cached Stage 1 feature missing: {resolved_cache_path}")

        geometry_inputs = None
        if cache_payload is None and self.online_fallback:
            geometry_inputs = self._load_geometry_inputs(sample.frame_paths)

        return {
            "group_id": sample.group_id,
            "source_dataset": sample.source_dataset,
            "source_sample_id": sample.source_sample_id,
            "frame_paths": sample.frame_paths,
            "valid_frame_mask": torch.tensor(sample.valid_frame_mask, dtype=torch.bool),
            "question_type": sample.question_type,
            "cached_features": cache_payload,
            "geometry_inputs": geometry_inputs,
            "window_id": sample.window_id,
            "cache_window_mode": sample.cache_window_mode,
        }


class Stage1SourceGroupedBatchSampler(Sampler[List[int]]):
    """Pack same-source windows contiguously while still filling each batch."""

    def __init__(
        self,
        dataset: Stage1GeometryDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_global_batches(self) -> List[List[int]]:
        source_groups = self.dataset.get_source_groups()
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            permutation = torch.randperm(len(source_groups), generator=generator).tolist()
            source_groups = [source_groups[index] for index in permutation]

        batches: List[List[int]] = []
        current_batch: List[int] = []
        for group in source_groups:
            cursor = 0
            while cursor < len(group):
                remaining = self.batch_size - len(current_batch)
                take = min(remaining, len(group) - cursor)
                current_batch.extend(group[cursor : cursor + take])
                cursor += take
                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        return batches

    def __iter__(self):
        batches = self._build_global_batches()
        if not batches:
            return iter(())

        if self.world_size > 1:
            if self.drop_last:
                usable = len(batches) - (len(batches) % self.world_size)
                batches = batches[:usable]
            else:
                original_batches = [list(batch) for batch in batches]
                target_size = math.ceil(len(batches) / self.world_size) * self.world_size
                while len(batches) < target_size:
                    batches.append(list(original_batches[(len(batches) - len(original_batches)) % len(original_batches)]))
            batches = batches[self.rank :: self.world_size]
        return iter(batches)

    def __len__(self) -> int:
        global_batch_count = len(self.dataset)
        if self.drop_last:
            global_batch_count = global_batch_count // self.batch_size
        else:
            global_batch_count = math.ceil(global_batch_count / self.batch_size)
        if self.world_size <= 1:
            return global_batch_count
        if self.drop_last:
            return global_batch_count // self.world_size
        return math.ceil(global_batch_count / self.world_size)


def _collate_cached_features(samples: Sequence[Dict]) -> Dict:
    cached_payloads = [sample["cached_features"] for sample in samples]
    max_frames = max(int(payload["g11_raw"].shape[0]) for payload in cached_payloads)
    max_patches = max(int(payload["g11_raw"].shape[1]) for payload in cached_payloads)
    batch_size = len(cached_payloads)
    first_g11 = cached_payloads[0]["g11_raw"]
    feature_shape = (batch_size, max_frames, max_patches, first_g11.shape[-1])
    collated = {
        key: first_g11.new_zeros(feature_shape)
        for key in ("g11_raw", "g17_raw", "g23_raw")
    }
    token_counts = torch.zeros(batch_size, max_frames, dtype=torch.long)
    valid_patch_mask = torch.zeros(batch_size, max_frames, max_patches, dtype=torch.bool)
    frame_shapes: List[List[tuple[int, int]]] = []

    for batch_idx, payload in enumerate(cached_payloads):
        num_frames, num_patches = payload["g11_raw"].shape[:2]
        for key in ("g11_raw", "g17_raw", "g23_raw"):
            collated[key][batch_idx, :num_frames, :num_patches] = payload[key]
        token_counts[batch_idx, : len(payload["token_counts"])] = torch.tensor(payload["token_counts"], dtype=torch.long)
        frame_shape_list = [tuple(int(value) for value in shape) for shape in payload["frame_shapes"]]
        frame_shapes.append(frame_shape_list + [(0, 0)] * (max_frames - len(frame_shape_list)))

        sample_valid_frames = samples[batch_idx]["valid_frame_mask"]
        valid_patch_mask[batch_idx, :num_frames, :num_patches] = sample_valid_frames[:num_frames, None].expand(
            num_frames, num_patches
        )

    collated["token_counts"] = token_counts
    collated["frame_shapes"] = frame_shapes
    collated["valid_patch_mask"] = valid_patch_mask
    return collated


def _collate_geometry_inputs(samples: Sequence[Dict]) -> Dict:
    geometry_inputs = [sample["geometry_inputs"] for sample in samples]
    return {
        "geometry_inputs": geometry_inputs,
        "valid_patch_mask": None,
        "frame_shapes": [],
    }


def stage1_collate_fn(samples: Sequence[Dict]) -> Dict:
    batch_size = len(samples)
    max_frames = max(int(sample["valid_frame_mask"].shape[0]) for sample in samples)
    valid_frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    for batch_idx, sample in enumerate(samples):
        valid_frame_mask[batch_idx, : sample["valid_frame_mask"].shape[0]] = sample["valid_frame_mask"]

    batch = {
        "group_id": [sample["group_id"] for sample in samples],
        "source_dataset": [sample["source_dataset"] for sample in samples],
        "source_sample_id": [sample["source_sample_id"] for sample in samples],
        "frame_paths": [sample["frame_paths"] for sample in samples],
        "question_type": [sample["question_type"] for sample in samples],
        "window_id": [sample["window_id"] for sample in samples],
        "cache_window_mode": [sample["cache_window_mode"] for sample in samples],
        "valid_frame_mask": valid_frame_mask,
        "cached_features": None,
        "geometry_inputs": None,
        "valid_patch_mask": None,
        "frame_shapes": [],
    }

    if all(sample["cached_features"] is not None for sample in samples):
        collated_features = _collate_cached_features(samples)
        batch["cached_features"] = {
            key: value
            for key, value in collated_features.items()
            if key in {"g11_raw", "g17_raw", "g23_raw", "token_counts", "valid_patch_mask"}
        }
        batch["valid_patch_mask"] = collated_features["valid_patch_mask"]
        batch["frame_shapes"] = collated_features["frame_shapes"]
        return batch

    if all(sample["geometry_inputs"] is not None for sample in samples):
        collated_inputs = _collate_geometry_inputs(samples)
        batch["geometry_inputs"] = collated_inputs["geometry_inputs"]
        batch["valid_patch_mask"] = collated_inputs["valid_patch_mask"]
        batch["frame_shapes"] = collated_inputs["frame_shapes"]
        return batch

    raise ValueError("Stage 1 batch mixes cached and uncached samples, which is not supported")


def _dump_visualization(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def visualize_mgc(path: str, *, masked_ratio: float, loss_dict: Dict[str, float], question_type: str) -> None:
    _dump_visualization(
        path,
        {
            "task": "mgc",
            "masked_ratio": masked_ratio,
            "question_type": question_type,
            "loss": loss_dict,
        },
    )


def visualize_lov(path: str, *, heldout_frame: int, loss_dict: Dict[str, float], question_type: str) -> None:
    _dump_visualization(
        path,
        {
            "task": "lov",
            "heldout_frame": heldout_frame,
            "question_type": question_type,
            "loss": loss_dict,
        },
    )


def visualize_shuffle_ablation(path: str, *, original_loss: float, shuffled_loss: float, question_type: str) -> None:
    _dump_visualization(
        path,
        {
            "task": "shuffle_ablation",
            "question_type": question_type,
            "original_loss": original_loss,
            "shuffled_loss": shuffled_loss,
            "delta": shuffled_loss - original_loss,
        },
    )

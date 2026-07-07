"""Stage 1 v2 correspondence-aware dataset helpers."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from qwen_vl.data.geometry_cache import GeometryCacheIndex
from qwen_vl.data.utils import prepare_image_inputs
from qwen_vl.train.stage1_compact_cache import (
    extract_window_from_source_joint_pack,
    is_projected_source_cache,
    is_window_ready_source_joint_pack,
    slice_projected_source_cache,
)
from qwen_vl.train.stage1_geometry import (
    Stage1SourceGroupedBatchSampler,
    visualize_lov,
    visualize_mgc,
    visualize_shuffle_ablation,
)


@dataclass
class Stage1V2Sample:
    group_id: str
    source_dataset: str
    source_sample_id: str
    frame_paths: List[str]
    sampled_frame_indices: List[int]
    valid_frame_mask: List[bool]
    question_type: str
    cache_path: str
    cache_format: str
    joint_pack_path: str
    joint_pack_key: str
    joint_pack_format: str
    corr_cache_path: str
    corr_pack_path: str
    corr_pack_key: str
    window_id: str
    cache_window_mode: str


class Stage1GeometryDatasetV2(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_processor,
        *,
        geometry_cache_required: bool = True,
        corr_cache_required: bool = True,
        online_fallback: bool = False,
        max_groups: int = -1,
        memory_cache_size: int = 8,
    ):
        super().__init__()
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Stage 1 v2 manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        self.image_processor = image_processor
        self.geometry_cache_required = bool(geometry_cache_required)
        self.corr_cache_required = bool(corr_cache_required)
        self.online_fallback = bool(online_fallback)
        self.memory_cache_size = max(int(memory_cache_size), 0)
        self.cache_index = GeometryCacheIndex(manifest_path)
        self._payload_cache: "OrderedDict[str, Dict]" = OrderedDict()
        self._joint_pack_cache: "OrderedDict[str, Dict]" = OrderedDict()
        self._corr_cache: "OrderedDict[str, Dict]" = OrderedDict()
        self._corr_pack_cache: "OrderedDict[str, Dict]" = OrderedDict()

        entries = list(self.cache_index.items())
        if max_groups > 0:
            entries = entries[:max_groups]
        self.entries = [
            Stage1V2Sample(
                group_id=entry["group_id"],
                source_dataset=entry["source_dataset"],
                source_sample_id=entry.get("source_sample_id", entry["group_id"]),
                frame_paths=list(entry["frame_paths"]),
                sampled_frame_indices=list(entry.get("sampled_frame_indices", list(range(len(entry["frame_paths"]))))),
                valid_frame_mask=list(entry.get("valid_frame_mask", [True] * len(entry["frame_paths"]))),
                question_type=entry.get("question_type", "unknown"),
                cache_path=entry.get("cache_path", ""),
                cache_format=entry.get("cache_format", ""),
                joint_pack_path=entry.get("joint_pack_path", ""),
                joint_pack_key=entry.get("joint_pack_key", entry["group_id"]),
                joint_pack_format=entry.get("joint_pack_format", ""),
                corr_cache_path=entry.get("corr_cache_path", ""),
                corr_pack_path=entry.get("corr_pack_path", ""),
                corr_pack_key=entry.get("corr_pack_key", entry["group_id"]),
                window_id=entry.get("window_id", "window_0"),
                cache_window_mode=entry.get("cache_window_mode", "fixed8"),
            )
            for entry in entries
        ]
        self.uses_projected_cache = any(
            sample.cache_format.startswith("projected_") or bool(sample.joint_pack_path)
            for sample in self.entries
        )
        source_group_map: Dict[str, List[int]] = {}
        for index, sample in enumerate(self.entries):
            source_group_map.setdefault(sample.source_sample_id, []).append(index)
        self._source_groups = [
            sorted(indices, key=lambda item: self.entries[item].window_id)
            for _, indices in sorted(source_group_map.items())
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def get_source_groups(self) -> List[List[int]]:
        return [list(group) for group in self._source_groups]

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

    def _remember_payload(self, cache_path: str, payload: Dict) -> Dict:
        if self.memory_cache_size <= 0:
            return payload
        self._payload_cache[cache_path] = payload
        self._payload_cache.move_to_end(cache_path)
        while len(self._payload_cache) > self.memory_cache_size:
            self._payload_cache.popitem(last=False)
        return payload

    def _load_cached_payload(self, cache_path: str) -> Dict:
        if cache_path in self._payload_cache:
            payload = self._payload_cache.pop(cache_path)
            self._payload_cache[cache_path] = payload
            return payload
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return self._remember_payload(cache_path, payload)

    def _load_joint_pack(self, pack_path: str) -> Dict:
        if pack_path in self._joint_pack_cache:
            payload = self._joint_pack_cache.pop(pack_path)
            self._joint_pack_cache[pack_path] = payload
            return payload
        payload = torch.load(pack_path, map_location="cpu", weights_only=True)
        if self.memory_cache_size > 0:
            self._joint_pack_cache[pack_path] = payload
            self._joint_pack_cache.move_to_end(pack_path)
            while len(self._joint_pack_cache) > self.memory_cache_size:
                self._joint_pack_cache.popitem(last=False)
        return payload

    def _load_corr_payload(self, cache_path: str) -> Dict:
        if cache_path in self._corr_cache:
            payload = self._corr_cache.pop(cache_path)
            self._corr_cache[cache_path] = payload
            return payload
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if self.memory_cache_size > 0:
            self._corr_cache[cache_path] = payload
            self._corr_cache.move_to_end(cache_path)
            while len(self._corr_cache) > self.memory_cache_size:
                self._corr_cache.popitem(last=False)
        return payload

    def _load_corr_pack(self, pack_path: str) -> Dict:
        if pack_path in self._corr_pack_cache:
            payload = self._corr_pack_cache.pop(pack_path)
            self._corr_pack_cache[pack_path] = payload
            return payload
        payload = torch.load(pack_path, map_location="cpu", weights_only=True)
        if self.memory_cache_size > 0:
            self._corr_pack_cache[pack_path] = payload
            self._corr_pack_cache.move_to_end(pack_path)
            while len(self._corr_pack_cache) > self.memory_cache_size:
                self._corr_pack_cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> Dict:
        sample = self.entries[index]
        cache_payload = None
        corr_payload = None

        if sample.joint_pack_path and os.path.exists(sample.joint_pack_path):
            joint_pack = self._load_joint_pack(sample.joint_pack_path)
            if not is_window_ready_source_joint_pack(joint_pack):
                raise ValueError(f"Unexpected Stage 1 v2 joint pack format: {sample.joint_pack_path}")
            window_payload = extract_window_from_source_joint_pack(joint_pack, sample.joint_pack_key)
            cache_payload = window_payload.get("cached_features")
            corr_payload = window_payload.get("corr_graph")
        elif sample.cache_path and os.path.exists(sample.cache_path):
            loaded_payload = self._load_cached_payload(sample.cache_path)
            if is_projected_source_cache(loaded_payload):
                cache_payload = slice_projected_source_cache(loaded_payload, sample.sampled_frame_indices)
            else:
                cache_payload = loaded_payload
        elif self.geometry_cache_required and not self.online_fallback:
            raise FileNotFoundError(f"Stage 1 v2 cached feature missing: {sample.cache_path}")

        if corr_payload is None:
            if sample.corr_pack_path and os.path.exists(sample.corr_pack_path):
                corr_pack = self._load_corr_pack(sample.corr_pack_path)
                windows = corr_pack.get("windows", {})
                corr_payload = windows.get(sample.corr_pack_key)
                if corr_payload is None and self.corr_cache_required:
                    raise KeyError(
                        f"Stage 1 v2 corr pack key missing: key={sample.corr_pack_key} path={sample.corr_pack_path}"
                    )
            elif sample.corr_cache_path and os.path.exists(sample.corr_cache_path):
                corr_payload = self._load_corr_payload(sample.corr_cache_path)
        if corr_payload is None and self.corr_cache_required:
            missing_target = sample.joint_pack_path or sample.corr_pack_path or sample.corr_cache_path
            raise FileNotFoundError(f"Stage 1 v2 corr graph missing: {missing_target}")

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
            "corr_graph": corr_payload,
            "geometry_inputs": geometry_inputs,
            "window_id": sample.window_id,
            "cache_window_mode": sample.cache_window_mode,
        }


def _collate_cached_features(samples: Sequence[Dict]) -> Dict:
    cached_payloads = [sample["cached_features"] for sample in samples]
    corr_payloads = [sample["corr_graph"] for sample in samples]
    feature_space = cached_payloads[0].get("feature_space", "raw")
    layer_names = tuple(cached_payloads[0].get("layer_names", ("g11", "g17", "g23")))
    if feature_space == "raw":
        feature_keys = tuple(f"{name}_raw" for name in layer_names)
    elif feature_space == "projected_quantized":
        feature_keys = tuple(f"{name}_q" for name in layer_names) + tuple(f"{name}_scale" for name in layer_names)
    else:
        feature_keys = layer_names
    max_frames = max(int(payload[feature_keys[0]].shape[0]) for payload in cached_payloads)
    max_patches = max(int(payload[feature_keys[0]].shape[1]) for payload in cached_payloads)
    max_neighbors = max(int(payload["neighbor_indices"].shape[2]) for payload in corr_payloads)
    batch_size = len(samples)
    first_g11 = cached_payloads[0][feature_keys[0]]
    collated = {}
    for key in feature_keys:
        value = cached_payloads[0][key]
        if value.dim() == 2:
            target_shape = (batch_size, max_frames, max_patches)
        elif value.dim() == 3:
            target_shape = (batch_size, max_frames, max_patches, value.shape[-1])
        else:
            target_shape = (batch_size, max_frames, max_patches, value.shape[-1])
        collated[key] = value.new_zeros(target_shape)
    token_counts = torch.zeros(batch_size, max_frames, dtype=torch.long)
    valid_patch_mask = torch.zeros(batch_size, max_frames, max_patches, dtype=torch.bool)
    frame_shapes: List[List[tuple[int, int]]] = []
    neighbor_indices = torch.full((batch_size, max_frames, max_patches, max_neighbors, 2), -1, dtype=torch.long)
    neighbor_scores = torch.zeros(batch_size, max_frames, max_patches, max_neighbors, dtype=torch.float32)
    neighbor_valid_mask = torch.zeros(batch_size, max_frames, max_patches, max_neighbors, dtype=torch.bool)

    for batch_idx, (payload, corr_payload) in enumerate(zip(cached_payloads, corr_payloads)):
        num_frames, num_patches = payload[feature_keys[0]].shape[:2]
        for key in feature_keys:
            if payload[key].dim() in (2, 3):
                collated[key][batch_idx, :num_frames, :num_patches] = payload[key]
            else:
                collated[key][batch_idx, :num_frames, :num_patches] = payload[key]
        token_counts[batch_idx, : len(payload["token_counts"])] = torch.tensor(payload["token_counts"], dtype=torch.long)
        frame_shape_list = [tuple(int(value) for value in shape) for shape in payload["frame_shapes"]]
        frame_shapes.append(frame_shape_list + [(0, 0)] * (max_frames - len(frame_shape_list)))
        sample_valid_frames = samples[batch_idx]["valid_frame_mask"]
        valid_patch_mask[batch_idx, :num_frames, :num_patches] = sample_valid_frames[:num_frames, None].expand(
            num_frames, num_patches
        )

        current_neighbors = corr_payload["neighbor_indices"]
        current_scores = corr_payload["neighbor_scores"]
        current_valid = current_neighbors[..., 0] >= 0
        neighbor_indices[batch_idx, : current_neighbors.shape[0], : current_neighbors.shape[1], : current_neighbors.shape[2]] = current_neighbors
        neighbor_scores[batch_idx, : current_scores.shape[0], : current_scores.shape[1], : current_scores.shape[2]] = current_scores.float()
        neighbor_valid_mask[batch_idx, : current_valid.shape[0], : current_valid.shape[1], : current_valid.shape[2]] = current_valid

    return {
        "cached_features": {
            **{key: collated[key] for key in feature_keys},
            "token_counts": token_counts,
            "valid_patch_mask": valid_patch_mask,
            "feature_space": feature_space,
            "layer_names": list(layer_names),
        },
        "frame_shapes": frame_shapes,
        "corr_graph": {
            "neighbor_indices": neighbor_indices,
            "neighbor_scores": neighbor_scores,
            "neighbor_valid_mask": neighbor_valid_mask,
        },
        "valid_patch_mask": valid_patch_mask,
    }


def _collate_corr_graph_only(samples: Sequence[Dict]) -> Dict:
    corr_payloads = [sample["corr_graph"] for sample in samples]
    max_frames = max(int(payload["neighbor_indices"].shape[0]) for payload in corr_payloads)
    max_patches = max(int(payload["neighbor_indices"].shape[1]) for payload in corr_payloads)
    max_neighbors = max(int(payload["neighbor_indices"].shape[2]) for payload in corr_payloads)
    batch_size = len(samples)
    neighbor_indices = torch.full((batch_size, max_frames, max_patches, max_neighbors, 2), -1, dtype=torch.long)
    neighbor_scores = torch.zeros(batch_size, max_frames, max_patches, max_neighbors, dtype=torch.float32)
    neighbor_valid_mask = torch.zeros(batch_size, max_frames, max_patches, max_neighbors, dtype=torch.bool)

    for batch_idx, corr_payload in enumerate(corr_payloads):
        current_neighbors = corr_payload["neighbor_indices"]
        current_scores = corr_payload["neighbor_scores"]
        current_valid = current_neighbors[..., 0] >= 0
        neighbor_indices[
            batch_idx,
            : current_neighbors.shape[0],
            : current_neighbors.shape[1],
            : current_neighbors.shape[2],
        ] = current_neighbors
        neighbor_scores[
            batch_idx,
            : current_scores.shape[0],
            : current_scores.shape[1],
            : current_scores.shape[2],
        ] = current_scores.float()
        neighbor_valid_mask[
            batch_idx,
            : current_valid.shape[0],
            : current_valid.shape[1],
            : current_valid.shape[2],
        ] = current_valid

    return {
        "corr_graph": {
            "neighbor_indices": neighbor_indices,
            "neighbor_scores": neighbor_scores,
            "neighbor_valid_mask": neighbor_valid_mask,
        }
    }


def stage1_v2_collate_fn(samples: Sequence[Dict]) -> Dict:
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
        "corr_graph": None,
        "geometry_inputs": None,
        "valid_patch_mask": None,
        "frame_shapes": [],
    }

    if all(sample["cached_features"] is not None and sample["corr_graph"] is not None for sample in samples):
        collated = _collate_cached_features(samples)
        batch["cached_features"] = collated["cached_features"]
        batch["corr_graph"] = collated["corr_graph"]
        batch["valid_patch_mask"] = collated["valid_patch_mask"]
        batch["frame_shapes"] = collated["frame_shapes"]
        return batch

    if all(sample["geometry_inputs"] is not None and sample["corr_graph"] is not None for sample in samples):
        collated = _collate_corr_graph_only(samples)
        batch["geometry_inputs"] = [sample["geometry_inputs"] for sample in samples]
        batch["corr_graph"] = collated["corr_graph"]
        return batch

    raise ValueError("Stage 1 v2 batch mixes unsupported cached/uncached samples")


__all__ = [
    "Stage1GeometryDatasetV2",
    "Stage1SourceGroupedBatchSampler",
    "stage1_v2_collate_fn",
    "visualize_lov",
    "visualize_mgc",
    "visualize_shuffle_ablation",
]

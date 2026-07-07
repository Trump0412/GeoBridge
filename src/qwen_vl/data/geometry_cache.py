"""Shared helpers for continuity-bank frame grouping and cache manifests."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


def sample_frame_indices(total_frames: int, target_frames: int = 8, min_frames: int = 4) -> List[int]:
    if total_frames < min_frames:
        return []
    if total_frames <= target_frames:
        return list(range(total_frames))
    if target_frames <= 1:
        return [0]
    return sorted({round(i * (total_frames - 1) / (target_frames - 1)) for i in range(target_frames)})


def stable_group_id(source_dataset: str, frame_paths: Sequence[str]) -> str:
    digest = hashlib.sha1()
    digest.update(source_dataset.encode("utf-8"))
    digest.update(b"\0")
    for path in frame_paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_source_sample_id(source_dataset: str, sample: Dict) -> str:
    digest = hashlib.sha1()
    digest.update(source_dataset.encode("utf-8"))
    digest.update(b"\0")
    if "video" in sample and isinstance(sample["video"], str):
        digest.update(b"video\0")
        digest.update(sample["video"].encode("utf-8"))
    elif "images" in sample or "image" in sample:
        digest.update(b"images\0")
        images = sample.get("images", sample.get("image"))
        if isinstance(images, list):
            for image_path in images:
                digest.update(str(image_path).encode("utf-8"))
                digest.update(b"\0")
    else:
        digest.update(json.dumps(sample, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def infer_question_type(sample: Dict) -> str:
    if isinstance(sample.get("question_type"), str) and sample["question_type"]:
        return sample["question_type"]
    spar_info = sample.get("spar_info")
    if isinstance(spar_info, str):
        try:
            parsed = json.loads(spar_info)
            value = parsed.get("type")
            if isinstance(value, str) and value:
                return value
        except json.JSONDecodeError:
            pass
    return "unknown"


def _collect_image_indices(value: Any) -> List[int]:
    indices: List[int] = []
    if isinstance(value, list):
        for item in value:
            indices.extend(_collect_image_indices(item))
    elif isinstance(value, int):
        indices.append(value)
    return indices


def extract_required_marker_indices(spar_info: Any) -> List[int]:
    if not isinstance(spar_info, str) or not spar_info:
        return []
    try:
        parsed = json.loads(spar_info)
    except json.JSONDecodeError:
        return []

    required = set()

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("_img_idx"):
                    required.update(index for index in _collect_image_indices(value) if index >= 0)
                else:
                    _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(parsed)
    return sorted(required)


def remap_spar_info_image_indices(spar_info: Any, sampled_frame_indices: Sequence[int]) -> Optional[str]:
    if not isinstance(spar_info, str) or not spar_info:
        return spar_info
    try:
        parsed = json.loads(spar_info)
    except json.JSONDecodeError:
        return spar_info

    index_map = {int(frame_index): offset for offset, frame_index in enumerate(sampled_frame_indices)}

    def _remap_value(value: Any) -> Any:
        if isinstance(value, list):
            return [_remap_value(item) for item in value]
        if isinstance(value, int):
            if value not in index_map:
                raise KeyError(value)
            return index_map[value]
        return value

    def _visit(node: Any) -> Any:
        if isinstance(node, dict):
            remapped = {}
            for key, value in node.items():
                if key.endswith("_img_idx"):
                    remapped[key] = _remap_value(value)
                else:
                    remapped[key] = _visit(value)
            return remapped
        if isinstance(node, list):
            return [_visit(item) for item in node]
        return node

    try:
        remapped = _visit(parsed)
    except KeyError:
        return None
    return json.dumps(remapped, ensure_ascii=False)


@dataclass
class GeometryCacheEntry:
    group_id: str
    source_dataset: str
    source_sample_id: str
    frame_paths: List[str]
    sampled_frame_indices: List[int]
    valid_frame_mask: List[bool]
    cache_path: str
    question_type: str = "unknown"
    window_id: str = "window_0"
    cache_window_mode: str = "fixed8"

    def to_json(self) -> Dict:
        return {
            "group_id": self.group_id,
            "source_dataset": self.source_dataset,
            "source_sample_id": self.source_sample_id,
            "frame_paths": self.frame_paths,
            "sampled_frame_indices": self.sampled_frame_indices,
            "valid_frame_mask": self.valid_frame_mask,
            "cache_path": self.cache_path,
            "question_type": self.question_type,
            "window_id": self.window_id,
            "cache_window_mode": self.cache_window_mode,
        }


class GeometryCacheIndex:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self._entries: Dict[str, Dict] = {}
        self._entries_by_source_sample_id: Dict[str, List[Dict]] = {}
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._entries[record["group_id"]] = record
                    source_sample_id = record.get("source_sample_id", "")
                    if source_sample_id:
                        self._entries_by_source_sample_id.setdefault(source_sample_id, []).append(record)
        for records in self._entries_by_source_sample_id.values():
            records.sort(key=lambda item: item.get("window_id", "window_0"))

    def __contains__(self, group_id: str) -> bool:
        return group_id in self._entries

    def get(self, group_id: str) -> Optional[Dict]:
        return self._entries.get(group_id)

    def get_by_source_sample_id(self, source_sample_id: str) -> List[Dict]:
        return list(self._entries_by_source_sample_id.get(source_sample_id, []))

    def items(self) -> Iterable[Dict]:
        return self._entries.values()


def _resolve_frame_paths(sample: Dict, data_path: str) -> Optional[List[str]]:
    if "video" in sample:
        video_root = os.path.join(data_path, sample["video"])
        if not os.path.isdir(video_root):
            return None
        return sorted(
            os.path.join(video_root, filename)
            for filename in os.listdir(video_root)
            if os.path.isfile(os.path.join(video_root, filename))
        )
    if "images" in sample or "image" in sample:
        images = sample.get("images", sample.get("image"))
        if not isinstance(images, list) or not images:
            return None
        return [os.path.join(data_path, image_path) for image_path in images]
    return None


def _make_entry(
    cache_group_name: str,
    question_type: str,
    source_sample_id: str,
    frame_paths: Sequence[str],
    sampled_indices: Sequence[int],
    cache_window_mode: str,
    window_id: str,
) -> GeometryCacheEntry:
    resolved_frame_paths = [frame_paths[index] for index in sampled_indices]
    if window_id == "window_0":
        group_id = stable_group_id(cache_group_name, resolved_frame_paths)
    else:
        group_id = stable_group_id(f"{cache_group_name}:{source_sample_id}:{window_id}", resolved_frame_paths)
    return GeometryCacheEntry(
        group_id=group_id,
        source_dataset=cache_group_name,
        source_sample_id=source_sample_id,
        frame_paths=list(resolved_frame_paths),
        sampled_frame_indices=list(sampled_indices),
        valid_frame_mask=[True] * len(resolved_frame_paths),
        cache_path="",
        question_type=question_type,
        window_id=window_id,
        cache_window_mode=cache_window_mode,
    )


def build_geometry_cache_entries(
    sample: Dict,
    data_path: str,
    *,
    cache_window_mode: str = "fixed8",
    num_windows_per_sample: int = 4,
    target_frames: int = 8,
    min_frames: int = 4,
    stride_min: int = 4,
    stride_max: int = 12,
) -> List[GeometryCacheEntry]:
    dataset_name = sample.get("dataset_name") or sample.get("tag") or "unknown"
    question_type = infer_question_type(sample)
    cache_group_name = sample.get("dataset_name", dataset_name)
    source_sample_id = stable_source_sample_id(cache_group_name, sample)
    frame_paths = _resolve_frame_paths(sample, data_path)
    if not frame_paths:
        return []

    total_frames = len(frame_paths)
    fixed_indices = sample_frame_indices(total_frames, target_frames=target_frames, min_frames=min_frames)
    if not fixed_indices:
        return []

    entries = [
        _make_entry(
            cache_group_name,
            question_type,
            source_sample_id,
            frame_paths,
            fixed_indices,
            cache_window_mode="fixed8" if cache_window_mode == "fixed8" else "multi_window",
            window_id="window_0",
        )
    ]
    if cache_window_mode != "multi_window" or total_frames < target_frames or num_windows_per_sample <= 1:
        return entries

    max_stride_fit = 1
    if target_frames > 1:
        max_stride_fit = max(1, (total_frames - 1) // (target_frames - 1))
    if max_stride_fit < 1:
        return entries

    rng = random.Random(source_sample_id)
    seen = {tuple(fixed_indices)}
    desired_total = max(1, int(num_windows_per_sample))
    attempts = 0
    max_attempts = desired_total * 16
    while len(entries) < desired_total and attempts < max_attempts:
        attempts += 1
        stride_upper = min(int(stride_max), max_stride_fit)
        stride_lower = min(max(int(stride_min), 1), stride_upper)
        if stride_upper < 1:
            break
        stride = rng.randint(stride_lower, stride_upper)
        span = 1 + (target_frames - 1) * stride
        if span > total_frames:
            continue
        start_max = total_frames - span
        start = rng.randint(0, start_max) if start_max > 0 else 0
        window_indices = tuple(start + step * stride for step in range(target_frames))
        if window_indices in seen:
            continue
        seen.add(window_indices)
        entries.append(
            _make_entry(
                cache_group_name,
                question_type,
                source_sample_id,
                frame_paths,
                window_indices,
                cache_window_mode="multi_window",
                window_id=f"window_{len(entries)}",
            )
        )
    return entries


def build_sampled_frame_paths(sample: Dict, data_path: str, target_frames: int = 8, min_frames: int = 4) -> Optional[GeometryCacheEntry]:
    entries = build_geometry_cache_entries(
        sample,
        data_path,
        cache_window_mode="fixed8",
        num_windows_per_sample=1,
        target_frames=target_frames,
        min_frames=min_frames,
    )
    return entries[0] if entries else None

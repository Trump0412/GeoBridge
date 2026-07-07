#!/usr/bin/env python3
"""Shared helpers for Stage1/VGGT similarity visualization scripts."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

DEFAULT_PROCESSOR_CANDIDATES = (
    os.environ.get("MODEL_NAME_OR_PATH"),
    os.environ.get("QWEN_MODEL_PATH"),
    "/data3/yeyuanhao/sp_re_cbp/thirdparty/models/Qwen2.5-VL-7B-Instruct",
)
SUPPORTED_LAYER_NAMES = ("g5", "g11", "g17", "g23", "stage1_c")
UPSAMPLE_MODES = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


@dataclass
class SimilarityFeatureBundle:
    """Feature bundle used by the visualization scripts."""

    processor_path: str
    frame_paths: List[str]
    rgb_frames: List[Image.Image]
    feature_maps: Dict[str, torch.Tensor]
    frame_shapes: List[Tuple[int, int]]
    token_counts: List[int]
    patch_grid: Tuple[int, int]
    merged_grid: Tuple[int, int]
    selector_probs: torch.Tensor | None
    selector_stats: torch.Tensor | None
    valid_patch_mask: torch.Tensor


def parse_int_csv(value: str) -> List[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(item) for item in items]


def parse_query_coord(value: str | None) -> Tuple[int, int] | None:
    if value is None:
        return None
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        return None
    if len(parts) != 2:
        raise ValueError(f"query_coord must be 'row,col', got {value!r}")
    return int(parts[0]), int(parts[1])


def parse_layer_ids(value: str) -> List[int]:
    return parse_int_csv(value)


def parse_layer_names(value: str) -> List[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("layers must not be empty")
    invalid = [item for item in items if item not in SUPPORTED_LAYER_NAMES]
    if invalid:
        raise ValueError(f"Unsupported layers: {invalid}. Supported values: {list(SUPPORTED_LAYER_NAMES)}")
    return items


def feature_key(layer_id: int) -> str:
    return f"g{int(layer_id)}_raw"


def stage1_checkpoint_label(checkpoint_path: str | Path) -> str:
    digits = "".join(ch for ch in Path(checkpoint_path).stem if ch.isdigit())
    return f"c{digits}" if digits else Path(checkpoint_path).stem.replace("checkpoint-", "c")


def layer_file_stem(layer_name: str, stage1_label: str | None = None) -> str:
    if layer_name == "stage1_c":
        return f"stage1_{stage1_label}" if stage1_label else "stage1_c"
    return layer_name


def layer_display_name(layer_name: str, stage1_label: str | None = None) -> str:
    if layer_name == "stage1_c":
        return f"Stage1-{stage1_label}" if stage1_label else "Stage1-c"
    return f"VGGT-{layer_name}"


def delta_stem() -> str:
    return "delta_stage1_c_minus_g11"


def resolve_frame_index(frame_count: int, requested_index: int | None) -> int:
    if requested_index is None:
        return frame_count // 2
    if not (0 <= requested_index < frame_count):
        raise ValueError(f"frame_index={requested_index} is out of range for {frame_count} frames")
    return int(requested_index)


def resolve_project_path(project_root: str | Path, path_value: str, *, required: bool = True) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root).expanduser() / candidate
    candidate = candidate.resolve(strict=False)
    if required and not candidate.exists():
        raise FileNotFoundError(f"Required path does not exist: {candidate}")
    return candidate


def resolve_processor_path(project_root: str | Path, processor_path: str | None = None) -> str:
    candidates: List[Path] = []
    raw_candidates: List[str] = []
    if processor_path:
        raw_candidates.append(processor_path)
    for candidate in DEFAULT_PROCESSOR_CANDIDATES:
        if candidate:
            raw_candidates.append(candidate)
    for candidate in raw_candidates:
        path_obj = Path(candidate).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path(project_root).expanduser() / path_obj
        candidates.append(path_obj.resolve(strict=False))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not resolve an image processor path. "
        f"Tried: {[str(candidate) for candidate in candidates]}"
    )


def load_manifest_row(path: str | Path, sample_index: int) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if index == sample_index:
                return json.loads(line)
    raise IndexError(f"sample_index={sample_index} is out of range for {path}")


def load_geometry_inputs(frame_paths: Sequence[str], image_processor) -> Tuple[List[Image.Image], torch.Tensor]:
    from qwen_vl.data.utils import prepare_image_inputs

    base_size = None
    rgb_frames: List[Image.Image] = []
    prepared = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            if base_size is None:
                base_size = rgb.size
            elif rgb.size != base_size:
                rgb = rgb.resize(base_size, Image.BILINEAR)
            rgb_frames.append(rgb.copy())
            prepared.append(prepare_image_inputs(rgb, image_processor)["geometry_encoder_inputs"])
    return rgb_frames, torch.stack(prepared, dim=0)


def build_valid_patch_mask(token_counts: Sequence[int], max_tokens: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, len(token_counts), max_tokens), dtype=torch.bool, device=device)
    for frame_idx, token_count in enumerate(token_counts):
        if token_count < 0 or token_count > max_tokens:
            raise ValueError(f"Invalid token_count={token_count}; expected 0 <= token_count <= {max_tokens}")
        mask[0, frame_idx, : int(token_count)] = True
    return mask


def reshape_frame_tokens(tokens: torch.Tensor, frame_shapes: Sequence[Tuple[int, int]], token_counts: Sequence[int]) -> torch.Tensor:
    if tokens.dim() == 4:
        if tokens.shape[0] != 1:
            raise ValueError(f"Expected batch dimension 1 for 4D tokens, got shape {tuple(tokens.shape)}")
        tokens = tokens.squeeze(0)
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens with shape [T, P, D], got {tuple(tokens.shape)}")
    if tokens.shape[0] != len(frame_shapes):
        raise ValueError(
            f"Token/frame count mismatch: tokens has {tokens.shape[0]} frames, "
            f"but frame_shapes has {len(frame_shapes)} entries."
        )
    feature_maps: List[torch.Tensor] = []
    for frame_idx, (frame_shape, token_count) in enumerate(zip(frame_shapes, token_counts)):
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        expected = frame_h * frame_w
        if int(token_count) != expected:
            raise ValueError(
                f"Frame {frame_idx} token_count={token_count} does not match frame_shape={frame_shape} "
                f"(expected {expected})."
            )
        frame_tokens = tokens[frame_idx, :expected]
        feature_maps.append(frame_tokens.reshape(frame_h, frame_w, frame_tokens.shape[-1]))
    return torch.stack(feature_maps, dim=0)


def reshape_frame_scores(scores: torch.Tensor, frame_shapes: Sequence[Tuple[int, int]], token_counts: Sequence[int]) -> torch.Tensor:
    if scores.dim() == 3:
        if scores.shape[0] != 1:
            raise ValueError(f"Expected batch dimension 1 for 3D scores, got {tuple(scores.shape)}")
        scores = scores.squeeze(0)
    if scores.dim() != 2:
        raise ValueError(f"Expected scores with shape [T, P], got {tuple(scores.shape)}")
    if scores.shape[0] != len(frame_shapes):
        raise ValueError(
            f"Score/frame count mismatch: scores has {scores.shape[0]} frames, "
            f"but frame_shapes has {len(frame_shapes)} entries."
        )
    score_maps: List[torch.Tensor] = []
    for frame_idx, (frame_shape, token_count) in enumerate(zip(frame_shapes, token_counts)):
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        expected = frame_h * frame_w
        if int(token_count) != expected:
            raise ValueError(
                f"Frame {frame_idx} token_count={token_count} does not match frame_shape={frame_shape} "
                f"(expected {expected})."
            )
        score_maps.append(scores[frame_idx, :expected].reshape(frame_h, frame_w))
    return torch.stack(score_maps, dim=0)


def resolve_query_coord(
    frame_shape: Tuple[int, int],
    *,
    query_mode: str,
    query_coord: Tuple[int, int] | None,
    reference_tokens: torch.Tensor,
    selector_probs: torch.Tensor | None = None,
    selector_stats: torch.Tensor | None = None,
) -> Tuple[int, int]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if query_mode == "coord":
        if query_coord is None:
            raise ValueError("query_mode='coord' requires --query_coord")
        row, col = int(query_coord[0]), int(query_coord[1])
    elif query_mode == "center":
        row, col = height // 2, width // 2
    elif query_mode == "max_cus":
        if selector_probs is None:
            raise ValueError("query_mode='max_cus' requires selector probabilities, but none are available.")
        flat_index = int(selector_probs.reshape(-1).argmax().item())
        row, col = divmod(flat_index, width)
    elif query_mode == "max_contrast":
        if selector_stats is not None:
            contrast_map = selector_stats[..., 0]
        else:
            contrast_map = torch.linalg.norm(reference_tokens.float() - reference_tokens.float().mean(dim=(0, 1)), dim=-1)
        flat_index = int(contrast_map.reshape(-1).argmax().item())
        row, col = divmod(flat_index, width)
    else:
        raise ValueError(f"Unsupported query_mode={query_mode!r}")
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"query_coord {(row, col)} is outside frame_shape={(height, width)}")
    return row, col


def resolve_query_coord_legacy(
    frame_shape: Tuple[int, int],
    *,
    query_mode: str,
    query_row: int,
    query_col: int,
    reference_tokens: torch.Tensor,
) -> Tuple[int, int]:
    explicit = None if query_row < 0 or query_col < 0 else (query_row, query_col)
    compat_mode = "coord" if explicit is not None else ("max_contrast" if query_mode == "max_norm" else query_mode)
    return resolve_query_coord(
        frame_shape,
        query_mode=compat_mode,
        query_coord=explicit,
        reference_tokens=reference_tokens,
        selector_probs=None,
        selector_stats=None,
    )


def cosine_similarity_map(query_frame: torch.Tensor, target_frame: torch.Tensor, query_coord: Tuple[int, int]) -> np.ndarray:
    query_h, query_w = int(query_coord[0]), int(query_coord[1])
    query_token = query_frame[query_h, query_w]
    query_token = F.normalize(query_token.float(), dim=-1)
    normalized_target = F.normalize(target_frame.float(), dim=-1)
    similarity = torch.einsum("d,hwd->hw", query_token, normalized_target)
    return similarity.detach().cpu().numpy().astype(np.float32)


def tokens_to_similarity_map(tokens: torch.Tensor, frame_shape: Tuple[int, int], query_coord: Tuple[int, int]) -> np.ndarray:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    frame_map = reshape_frame_tokens(tokens.unsqueeze(0), [frame_shape], [height * width])[0]
    return cosine_similarity_map(frame_map, frame_map, query_coord)


def describe_array(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).any():
        raise ValueError("Array stats require at least one finite value.")
    finite = array[np.isfinite(array)]
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p2": float(np.percentile(finite, 2)),
        "p98": float(np.percentile(finite, 98)),
    }


def robust_normalize(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    normalized, _ = normalize_similarity_map(values, mode="percentile", p_low=low, p_high=high)
    return normalized


def normalize_similarity_map(
    sim: np.ndarray,
    mode: str = "percentile",
    p_low: float = 2.0,
    p_high: float = 98.0,
    temperature: float = 0.05,
    reference_values: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, float | str | bool]]:
    sim = np.asarray(sim, dtype=np.float32)
    if not np.isfinite(sim).all():
        raise ValueError("Similarity map contains NaN or Inf values.")
    ref = np.asarray(reference_values, dtype=np.float32) if reference_values is not None else sim
    if ref.size == 0:
        raise ValueError("reference_values must not be empty.")

    info: Dict[str, float | str | bool] = {
        "mode": mode,
        "temperature": float(temperature),
        "p_low": float(p_low),
        "p_high": float(p_high),
    }
    if mode == "raw":
        return sim.astype(np.float32), info

    if mode == "minmax":
        lo = float(ref.min())
        hi = float(ref.max())
        info.update({"lo": lo, "hi": hi})
        if abs(hi - lo) < 1e-6:
            return np.zeros_like(sim, dtype=np.float32), info
        return ((sim - lo) / (hi - lo)).astype(np.float32), info

    if mode == "percentile":
        lo = float(np.percentile(ref, p_low))
        hi = float(np.percentile(ref, p_high))
        if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-6:
            lo = float(ref.min())
            hi = float(ref.max())
        info.update({"lo": lo, "hi": hi})
        if abs(hi - lo) < 1e-6:
            return np.zeros_like(sim, dtype=np.float32), info
        clipped = np.clip(sim, lo, hi)
        return ((clipped - lo) / (hi - lo)).astype(np.float32), info

    if mode == "softmax":
        if temperature <= 0:
            raise ValueError("temperature must be positive for softmax visualization.")
        flat = (sim.reshape(-1).astype(np.float64) / float(temperature))
        flat = flat - flat.max()
        weights = np.exp(flat)
        weights_sum = weights.sum()
        if not math.isfinite(weights_sum) or weights_sum <= 0:
            return np.zeros_like(sim, dtype=np.float32), info
        weights = weights / weights_sum
        weights = weights.reshape(sim.shape)
        max_value = float(weights.max())
        info["hi"] = max_value
        if max_value <= 0:
            return np.zeros_like(sim, dtype=np.float32), info
        return (weights / max_value).astype(np.float32), info

    raise ValueError(f"Unsupported normalization mode: {mode}")


def ensure_unit_interval(sim_vis: np.ndarray) -> Tuple[np.ndarray, bool]:
    vis = np.asarray(sim_vis, dtype=np.float32)
    clipped = bool((vis < 0).any() or (vis > 1).any())
    return np.clip(vis, 0.0, 1.0).astype(np.float32), clipped


def maybe_smooth_map(sim_vis: np.ndarray, sigma: float) -> np.ndarray:
    vis = np.asarray(sim_vis, dtype=np.float32)
    if sigma <= 0:
        return vis
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError("smooth_sigma > 0 requires scipy to be installed.") from exc
    smoothed = gaussian_filter(vis, sigma=float(sigma))
    min_value = float(smoothed.min())
    max_value = float(smoothed.max())
    if abs(max_value - min_value) < 1e-6:
        return np.zeros_like(vis, dtype=np.float32)
    return ((smoothed - min_value) / (max_value - min_value)).astype(np.float32)


def resize_similarity_map(sim_vis: np.ndarray, output_size: Tuple[int, int], upsample: str) -> np.ndarray:
    if upsample not in UPSAMPLE_MODES:
        raise ValueError(f"Unsupported upsample mode: {upsample}")
    vis_uint8 = np.clip(np.asarray(sim_vis, dtype=np.float32), 0.0, 1.0)
    vis_uint8 = (vis_uint8 * 255.0).round().astype(np.uint8)
    resized = Image.fromarray(vis_uint8, mode="L").resize(output_size, UPSAMPLE_MODES[upsample])
    return np.asarray(resized, dtype=np.float32) / 255.0


def colorize_similarity_map(
    similarity_map: np.ndarray,
    output_size: Tuple[int, int],
    cmap: str = "viridis",
    upsample: str = "bicubic",
) -> Tuple[Image.Image, np.ndarray]:
    normalized = robust_normalize(similarity_map)
    colorized = render_heatmap_image(normalized, output_size=output_size, cmap=cmap, upsample=upsample)
    return colorized, normalized


def render_heatmap_image(
    sim_vis: np.ndarray,
    *,
    output_size: Tuple[int, int] | None = None,
    cmap: str = "viridis",
    upsample: str = "bicubic",
) -> Image.Image:
    vis = np.asarray(sim_vis, dtype=np.float32)
    vis, _ = ensure_unit_interval(vis)
    if output_size is not None:
        vis = resize_similarity_map(vis, output_size=output_size, upsample=upsample)
    rgb = apply_colormap(vis, cmap)
    rgb_uint8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgb_uint8, mode="RGB")


def apply_colormap(vis: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        return matplotlib.colormaps.get_cmap(cmap)(vis)[..., :3]
    except ImportError:
        value = np.asarray(vis, dtype=np.float32)
        value, _ = ensure_unit_interval(value)
        if cmap in {"gray", "grey"}:
            return np.stack([value, value, value], axis=-1)
        return np.stack(
            [
                np.clip(1.5 * value - 0.25, 0.0, 1.0),
                np.clip(1.5 - np.abs(2.0 * value - 1.0) * 1.5, 0.0, 1.0),
                np.clip(1.25 - 1.5 * value, 0.0, 1.0),
            ],
            axis=-1,
        )


def save_heatmap_only(
    sim_vis: np.ndarray,
    out_path: str | Path,
    cmap: str = "viridis",
    dpi: int = 200,
    output_size: Tuple[int, int] | None = None,
    upsample: str = "bicubic",
) -> None:
    del dpi  # The image writer path does not use figure DPI.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_heatmap_image(sim_vis, output_size=output_size, cmap=cmap, upsample=upsample).save(out_path)


def make_overlay(
    base_image: Image.Image,
    similarity_map: np.ndarray,
    alpha: float,
    *,
    cmap: str = "viridis",
    upsample: str = "bicubic",
) -> Image.Image:
    normalized, _ = normalize_similarity_map(similarity_map, mode="percentile")
    return save_overlay_image(
        base_image=base_image,
        sim_vis=normalized,
        cmap=cmap,
        alpha=alpha,
        upsample=upsample,
    )


def save_overlay_image(
    base_image: Image.Image,
    sim_vis: np.ndarray,
    *,
    cmap: str = "viridis",
    alpha: float = 0.55,
    upsample: str = "bicubic",
    topk_entries: Sequence[Mapping[str, float | int]] | None = None,
    grid_shape: Tuple[int, int] | None = None,
    outline_color: str = "#ffffff",
    output_size: Tuple[int, int] | None = None,
) -> Image.Image:
    if output_size is None:
        output_size = base_image.size
    heatmap = render_heatmap_image(sim_vis, output_size=output_size, cmap=cmap, upsample=upsample)
    overlay = Image.blend(base_image.convert("RGB"), heatmap, float(alpha))
    if topk_entries and grid_shape is not None:
        overlay = draw_patch_boxes(overlay, grid_shape=grid_shape, entries=topk_entries, outline=outline_color)
    return overlay


def save_overlay(
    image: Image.Image,
    sim_vis: np.ndarray,
    out_path: str | Path,
    cmap: str = "viridis",
    alpha: float = 0.55,
    upsample: str = "bicubic",
    topk_entries: Sequence[Mapping[str, float | int]] | None = None,
    grid_shape: Tuple[int, int] | None = None,
    outline_color: str = "#ffffff",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_overlay_image(
        base_image=image,
        sim_vis=sim_vis,
        cmap=cmap,
        alpha=alpha,
        upsample=upsample,
        topk_entries=topk_entries,
        grid_shape=grid_shape,
        outline_color=outline_color,
    ).save(out_path)


def draw_query_box(image: Image.Image, frame_shape: Tuple[int, int], query_coord: Tuple[int, int]) -> Image.Image:
    row, col = int(query_coord[0]), int(query_coord[1])
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    left, top, right, bottom = patch_box_xyxy(image.size, frame_shape, row, col)
    draw.rectangle((left, top, right, bottom), outline="#ff2d2d", width=4)
    return annotated


def patch_box_xyxy(image_size: Tuple[int, int], grid_shape: Tuple[int, int], row: int, col: int) -> Tuple[int, int, int, int]:
    width, height = int(image_size[0]), int(image_size[1])
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    patch_w = width / float(grid_w)
    patch_h = height / float(grid_h)
    left = int(round(col * patch_w))
    top = int(round(row * patch_h))
    right = max(int(round((col + 1) * patch_w)) - 1, left)
    bottom = max(int(round((row + 1) * patch_h)) - 1, top)
    return left, top, right, bottom


def draw_patch_boxes(
    image: Image.Image,
    *,
    grid_shape: Tuple[int, int],
    entries: Sequence[Mapping[str, float | int]],
    outline: str = "#ffffff",
    width: int = 2,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for entry in entries:
        left, top, right, bottom = patch_box_xyxy(
            annotated.size,
            grid_shape,
            int(entry["row"]),
            int(entry["col"]),
        )
        draw.rectangle((left, top, right, bottom), outline=outline, width=width)
    return annotated


def topk_patch_entries(sim_map: np.ndarray, topk: int) -> List[Dict[str, float | int]]:
    if topk <= 0:
        return []
    array = np.asarray(sim_map, dtype=np.float32)
    flat = array.reshape(-1)
    actual_topk = min(int(topk), int(flat.size))
    indices = np.argpartition(-flat, actual_topk - 1)[:actual_topk]
    indices = indices[np.argsort(-flat[indices])]
    width = array.shape[1]
    results: List[Dict[str, float | int]] = []
    for rank, flat_index in enumerate(indices, start=1):
        row, col = divmod(int(flat_index), width)
        results.append(
            {
                "rank": rank,
                "row": int(row),
                "col": int(col),
                "score": float(flat[flat_index]),
            }
        )
    return results


def draw_caption(image: Image.Image, caption: str) -> Image.Image:
    font = ImageFont.load_default()
    margin = 8
    caption_height = 24
    canvas = Image.new("RGB", (image.width, image.height + caption_height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, image.height + 5), caption, fill="black", font=font)
    return canvas


def build_contact_sheet(panels: Sequence[Tuple[str, Image.Image]], columns: int = 3, background: str = "white") -> Image.Image:
    if not panels:
        raise ValueError("panels must not be empty")
    captioned = [draw_caption(image, caption) for caption, image in panels]
    cell_w = max(image.width for image in captioned)
    cell_h = max(image.height for image in captioned)
    rows = math.ceil(len(captioned) / columns)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), background)
    for idx, image in enumerate(captioned):
        row = idx // columns
        col = idx % columns
        x = col * cell_w
        y = row * cell_h
        sheet.paste(image, (x, y))
    return sheet


def build_grid_image(rows: Sequence[Sequence[Tuple[str, Image.Image]]], background: str = "white") -> Image.Image:
    if not rows or not rows[0]:
        raise ValueError("rows must not be empty")
    captioned_rows = [[draw_caption(image, caption) for caption, image in row] for row in rows]
    cell_w = max(image.width for row in captioned_rows for image in row)
    cell_h = max(image.height for row in captioned_rows for image in row)
    num_rows = len(captioned_rows)
    num_cols = max(len(row) for row in captioned_rows)
    canvas = Image.new("RGB", (num_cols * cell_w, num_rows * cell_h), background)
    for row_idx, row in enumerate(captioned_rows):
        for col_idx, image in enumerate(row):
            canvas.paste(image, (col_idx * cell_w, row_idx * cell_h))
    return canvas


def git_commit_hash(project_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def validate_frame_indices(num_frames: int, query_frame_index: int, target_frame_indices: Sequence[int]) -> None:
    if not (0 <= int(query_frame_index) < num_frames):
        raise ValueError(f"query_frame_index={query_frame_index} is outside [0, {num_frames - 1}]")
    invalid = [int(index) for index in target_frame_indices if not (0 <= int(index) < num_frames)]
    if invalid:
        raise ValueError(f"target_frame_indices contain out-of-range values: {invalid}")


def validate_feature_alignment(bundle: SimilarityFeatureBundle, *, require_g11: bool = True) -> None:
    if require_g11 and "g11" not in bundle.feature_maps:
        raise ValueError("Feature bundle is missing g11 features.")
    reference_shape = None
    for layer_name, feature_map in bundle.feature_maps.items():
        if feature_map.dim() != 4:
            raise ValueError(f"{layer_name} feature_map must have shape [T, H, W, D], got {tuple(feature_map.shape)}")
        layer_shapes = [tuple(int(v) for v in feature_map[frame_idx].shape[:2]) for frame_idx in range(feature_map.shape[0])]
        if layer_shapes != list(bundle.frame_shapes):
            raise ValueError(
                f"Feature shape mismatch for {layer_name}: {layer_shapes} vs frame_shapes={bundle.frame_shapes}"
            )
        if reference_shape is None:
            reference_shape = layer_shapes
        elif layer_shapes != reference_shape:
            raise ValueError(f"Layer {layer_name} does not align with other feature maps.")
    if "stage1_c" in bundle.feature_maps and "g11" in bundle.feature_maps:
        stage1_shape = tuple(int(v) for v in bundle.feature_maps["stage1_c"].shape[1:3])
        g11_shape = tuple(int(v) for v in bundle.feature_maps["g11"].shape[1:3])
        if stage1_shape != g11_shape:
            raise ValueError(f"Stage1-c grid {stage1_shape} does not match VGGT-g11 grid {g11_shape}.")


def extract_stage1_vggt_feature_bundle(
    *,
    frame_paths: Sequence[str],
    project_root: str | Path,
    vggt_model_path: str,
    stage1_checkpoint: str,
    layer_names: Sequence[str],
    processor_path: str | None = None,
    device: str = "cuda",
    d_geom: int = 1024,
    temporal_radius: int = 2,
    topk_neighbors: int = 8,
    continuity_mlp_hidden_ratio: float = 2.0,
    continuity_attention_heads: int = 4,
    corr_score_beta: float = 1.0,
    time_bias_init: float = -0.10,
) -> SimilarityFeatureBundle:
    from transformers import AutoProcessor

    from qwen_vl.model.geometry_bank import VGGTBankExtractor
    from qwen_vl.model.geometry_bank.corr_graph_utils import build_feature_knn_corr_graph_batch
    from qwen_vl.train.train_stage1_continuity_v2 import Stage1ContinuityModelV2

    resolved_processor_path = resolve_processor_path(project_root, processor_path)
    processor = AutoProcessor.from_pretrained(resolved_processor_path).image_processor
    rgb_frames, geometry_inputs = load_geometry_inputs(frame_paths, processor)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    if torch_device.type == "cuda":
        geometry_inputs = geometry_inputs.to(torch_device)

    required_vggt_layers = sorted(
        {
            int(layer_name.removeprefix("g"))
            for layer_name in layer_names
            if layer_name.startswith("g")
        }
        | {11}
    )
    extractor = VGGTBankExtractor(
        model_path=vggt_model_path,
        layer_ids=tuple(required_vggt_layers),
        spatial_merge_size=getattr(processor, "merge_size", 2),
        freeze_encoder=True,
    ).to(torch_device).eval()

    stage1_model = Stage1ContinuityModelV2(
        geometry_encoder_path=vggt_model_path,
        feature_layers=("g11",),
        d_geom=d_geom,
        continuity_radius=temporal_radius,
        continuity_use_spatial_neighbors=False,
        continuity_mlp_hidden_ratio=continuity_mlp_hidden_ratio,
        continuity_attention_heads=continuity_attention_heads,
        corr_score_beta=corr_score_beta,
        time_bias_init=time_bias_init,
    ).to(torch_device)
    stage1_model.initialize_from_checkpoint(stage1_checkpoint, torch_device)
    stage1_model.eval()

    with torch.no_grad():
        extracted = extractor.extract(geometry_inputs)
        projected = stage1_model.geo_projector({"g11_raw": extracted.layer_tokens["g11_raw"]})
        z = projected["g11"]
        token_counts = [int(value) for value in extracted.frame_layout.token_counts]
        frame_shapes = [tuple(int(v) for v in shape) for shape in extracted.frame_layout.frame_shapes]
        valid_patch_mask = build_valid_patch_mask(token_counts, int(z.shape[1]), device=torch_device)
        corr_graph = build_feature_knn_corr_graph_batch(
            z,
            token_counts,
            temporal_radius=temporal_radius,
            topk_neighbors=topk_neighbors,
        )
        neighbor_indices = corr_graph["neighbor_indices"].unsqueeze(0)
        neighbor_scores = corr_graph["neighbor_scores"].unsqueeze(0)
        z_batch = z.unsqueeze(0)
        selector_output = stage1_model.continuity_selector(z_batch, neighbor_indices, neighbor_scores, valid_patch_mask)
        edge_activation = stage1_model.activated_corr_graph(
            neighbor_indices,
            neighbor_scores,
            selector_output["probs"],
        )["activation"]
        continuity, _ = stage1_model.continuity_builder.forward_from_fused(
            z_batch,
            mode="corr_graph",
            corr_neighbor_indices=neighbor_indices,
            corr_neighbor_scores=neighbor_scores,
            corr_edge_activation=edge_activation,
            return_aux=True,
        )

    feature_maps: Dict[str, torch.Tensor] = {}
    for layer_id in required_vggt_layers:
        raw_key = feature_key(layer_id)
        feature_maps[f"g{layer_id}"] = reshape_frame_tokens(
            extracted.layer_tokens[raw_key].detach().cpu(),
            frame_shapes=frame_shapes,
            token_counts=token_counts,
        )
    feature_maps["stage1_c"] = reshape_frame_tokens(
        continuity.detach().cpu(),
        frame_shapes=frame_shapes,
        token_counts=token_counts,
    )
    selector_probs = reshape_frame_scores(
        selector_output["probs"].detach().cpu(),
        frame_shapes=frame_shapes,
        token_counts=token_counts,
    )
    selector_stats = selector_output["stats"].detach().cpu()
    selector_stat_maps = []
    for stat_idx in range(selector_stats.shape[-1]):
        selector_stat_maps.append(
            reshape_frame_scores(
                selector_stats[..., stat_idx],
                frame_shapes=frame_shapes,
                token_counts=token_counts,
            )
        )
    selector_stats = torch.stack(selector_stat_maps, dim=-1)

    bundle = SimilarityFeatureBundle(
        processor_path=resolved_processor_path,
        frame_paths=[str(path) for path in frame_paths],
        rgb_frames=rgb_frames,
        feature_maps=feature_maps,
        frame_shapes=frame_shapes,
        token_counts=token_counts,
        patch_grid=tuple(int(v) for v in extracted.patch_grid),
        merged_grid=tuple(int(v) for v in extracted.merged_grid),
        selector_probs=selector_probs,
        selector_stats=selector_stats,
        valid_patch_mask=valid_patch_mask.detach().cpu(),
    )
    validate_feature_alignment(bundle)
    return bundle

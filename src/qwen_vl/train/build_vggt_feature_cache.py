"""Build continuity-bank cache manifests and optional VGGT feature shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qwen_vl.data import data_list
from qwen_vl.data.geometry_cache import build_geometry_cache_entries
from qwen_vl.data.utils import prepare_image_inputs
from qwen_vl.model.geometry_bank import VGGTBankExtractor


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def iter_unique_groups(
    dataset_use: str,
    max_groups: int,
    *,
    cache_window_mode: str,
    num_windows_per_sample: int,
    num_frames: int,
    stride_min: int,
    stride_max: int,
):
    seen = set()
    count = 0
    for config in data_list(dataset_use.split(",")):
        with open(config["annotation_path"], "r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        for annotation in annotations:
            annotation["data_path"] = config["data_path"]
            annotation["tag"] = config["tag"]
            annotation["dataset_name"] = config.get("dataset_name", config["tag"])
            entries = build_geometry_cache_entries(
                annotation,
                config["data_path"],
                cache_window_mode=cache_window_mode,
                num_windows_per_sample=num_windows_per_sample,
                target_frames=num_frames,
                min_frames=4,
                stride_min=stride_min,
                stride_max=stride_max,
            )
            for entry in entries:
                if entry.group_id in seen:
                    continue
                seen.add(entry.group_id)
                yield entry
                count += 1
                if max_groups > 0 and count >= max_groups:
                    return


def load_geometry_inputs(frame_paths, image_processor):
    resized_paths = list(frame_paths)
    if len(resized_paths) > 1:
        from PIL import Image

        with Image.open(resized_paths[0]) as first_image:
            base_size = first_image.size
        prepared_inputs = []
        for frame_path in resized_paths:
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB")
                if rgb.size != base_size:
                    rgb = rgb.resize(base_size, Image.BILINEAR)
                prepared_inputs.append(prepare_image_inputs(rgb, image_processor))
        return torch.stack([prepared["geometry_encoder_inputs"] for prepared in prepared_inputs])

    inputs = []
    for frame_path in resized_paths:
        prepared = prepare_image_inputs(frame_path, image_processor)
        inputs.append(prepared["geometry_encoder_inputs"])
    return torch.stack(inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_use", type=str, default="llava_hound_64k,spar_234k")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--geometry_encoder_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--write_features", type=str, default="False")
    parser.add_argument("--max_groups", type=int, default=-1)
    parser.add_argument("--cache_window_mode", type=str, default="fixed8", choices=("fixed8", "multi_window"))
    parser.add_argument("--num_windows_per_sample", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--stride_min", type=int, default=4)
    parser.add_argument("--stride_max", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    manifest_path = args.manifest_path or os.path.join(args.cache_dir, "manifest.jsonl")
    complete_flag = os.path.join(args.cache_dir, ".complete")
    write_features = parse_bool(args.write_features)

    image_processor = AutoProcessor.from_pretrained(args.model_name_or_path).image_processor
    extractor = None
    if write_features:
        extractor = VGGTBankExtractor(
            model_path=args.geometry_encoder_path,
            layer_ids=(11, 17, 23),
            spatial_merge_size=getattr(image_processor, "merge_size", 2),
            freeze_encoder=True,
        ).eval()
        if torch.cuda.is_available():
            extractor = extractor.cuda()

    with open(manifest_path, "w", encoding="utf-8") as manifest_handle:
        for index, entry in enumerate(
            iter_unique_groups(
                args.dataset_use,
                args.max_groups,
                cache_window_mode=args.cache_window_mode,
                num_windows_per_sample=args.num_windows_per_sample,
                num_frames=args.num_frames,
                stride_min=args.stride_min,
                stride_max=args.stride_max,
            ),
            start=1,
        ):
            feature_path = os.path.join(args.cache_dir, "features", f"{entry.group_id}.pt")
            entry.cache_path = feature_path if write_features else ""
            if write_features and not os.path.exists(feature_path):
                os.makedirs(os.path.dirname(feature_path), exist_ok=True)
                geometry_inputs = load_geometry_inputs(entry.frame_paths, image_processor)
                if torch.cuda.is_available():
                    geometry_inputs = geometry_inputs.cuda()
                extracted = extractor.extract(geometry_inputs)
                payload = {
                    "group_id": entry.group_id,
                    "source_dataset": entry.source_dataset,
                    "source_sample_id": entry.source_sample_id,
                    "window_id": entry.window_id,
                    "cache_window_mode": entry.cache_window_mode,
                    "frame_paths": entry.frame_paths,
                    "sampled_frame_indices": entry.sampled_frame_indices,
                    "valid_frame_mask": entry.valid_frame_mask,
                    "g11_raw": extracted.layer_tokens["g11_raw"].cpu().to(torch.bfloat16),
                    "g17_raw": extracted.layer_tokens["g17_raw"].cpu().to(torch.bfloat16),
                    "g23_raw": extracted.layer_tokens["g23_raw"].cpu().to(torch.bfloat16),
                    "token_counts": extracted.frame_layout.token_counts,
                    "frame_shapes": extracted.frame_layout.frame_shapes,
                    "patch_grid": extracted.patch_grid,
                    "merged_grid": extracted.merged_grid,
                }
                torch.save(payload, feature_path)

            manifest_handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
            if index % 100 == 0:
                manifest_handle.flush()
                print(f"[cache] processed_groups={index}", flush=True)

    with open(complete_flag, "w", encoding="utf-8") as handle:
        handle.write("done\n")
    print(f"[cache] manifest_path={manifest_path}")
    print(f"[cache] complete_flag={complete_flag}")


if __name__ == "__main__":
    main()

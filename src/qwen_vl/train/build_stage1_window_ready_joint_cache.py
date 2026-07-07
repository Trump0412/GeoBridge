"""Build source-level window-ready joint cache for Stage 1 v2."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch

from qwen_vl.train.stage1_compact_cache import (
    WINDOW_READY_SOURCE_JOINT_PACK_FORMAT,
    materialize_projected_source_cache,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_path", type=str, required=True)
    parser.add_argument("--output_cache_dir", type=str, required=True)
    parser.add_argument("--output_manifest_path", type=str, required=True)
    parser.add_argument("--feature_dtype", type=str, default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--score_dtype", type=str, default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max_sources", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--overwrite", type=str, default="False")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def group_records(records: Sequence[Dict]) -> List[Tuple[str, List[Dict]]]:
    grouped = defaultdict(list)
    for record in records:
        source_sample_id = record.get("source_sample_id", record["group_id"])
        grouped[source_sample_id].append(record)
    return sorted(grouped.items(), key=lambda item: item[0])


def pack_path(cache_dir: str, source_sample_id: str) -> str:
    prefix = source_sample_id[:2] if source_sample_id else "xx"
    return os.path.join(cache_dir, "source_window_ready_packs", prefix, f"{source_sample_id}.pt")


def build_window_payload(
    record: Dict,
    source_payload: Dict,
    corr_pack: Dict,
    *,
    feature_dtype: torch.dtype,
    score_dtype: torch.dtype,
) -> Dict:
    sampled_frame_indices = record.get("sampled_frame_indices", list(range(len(record.get("frame_paths", [])))))
    cached_features = materialize_projected_source_cache(
        source_payload,
        sampled_frame_indices,
        output_dtype=feature_dtype,
    )
    corr_graph = corr_pack.get("windows", {}).get(record["group_id"])
    if corr_graph is None:
        raise KeyError(
            f"corr window missing for group_id={record['group_id']} "
            f"source_sample_id={record.get('source_sample_id', record['group_id'])}"
        )
    return {
        "cached_features": cached_features,
        "corr_graph": {
            "neighbor_indices": corr_graph["neighbor_indices"].cpu(),
            "neighbor_scores": corr_graph["neighbor_scores"].to(dtype=score_dtype).cpu(),
        },
    }


def build_pack(
    source_sample_id: str,
    source_records: Sequence[Dict],
    output_path: str,
    *,
    feature_dtype: torch.dtype,
    score_dtype: torch.dtype,
) -> None:
    cache_path = source_records[0].get("cache_path", "")
    corr_pack_path = source_records[0].get("corr_pack_path", "")
    if not cache_path or not os.path.exists(cache_path):
        raise FileNotFoundError(f"source cache missing for source_sample_id={source_sample_id}: {cache_path}")
    if not corr_pack_path or not os.path.exists(corr_pack_path):
        raise FileNotFoundError(f"corr pack missing for source_sample_id={source_sample_id}: {corr_pack_path}")

    source_payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    corr_pack = torch.load(corr_pack_path, map_location="cpu", weights_only=True)
    layer_names = list(source_payload.get("layer_names", ("g11", "g17", "g23")))
    windows = {
        record["group_id"]: build_window_payload(
            record,
            source_payload,
            corr_pack,
            feature_dtype=feature_dtype,
            score_dtype=score_dtype,
        )
        for record in source_records
    }

    payload = {
        "pack_format": WINDOW_READY_SOURCE_JOINT_PACK_FORMAT,
        "source_sample_id": source_sample_id,
        "source_dataset": source_records[0].get("source_dataset", "unknown"),
        "layer_names": layer_names,
        "feature_dtype": str(feature_dtype).removeprefix("torch."),
        "score_dtype": str(score_dtype).removeprefix("torch."),
        "windows": windows,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)


def main():
    args = parse_args()
    overwrite = parse_bool(args.overwrite)
    feature_dtype = resolve_dtype(args.feature_dtype)
    score_dtype = resolve_dtype(args.score_dtype)

    with open(args.input_manifest_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    grouped_records = group_records(records)
    if args.max_sources > 0:
        grouped_records = grouped_records[: args.max_sources]
    if args.num_shards > 1:
        if not (0 <= args.shard_rank < args.num_shards):
            raise ValueError(f"Invalid shard_rank={args.shard_rank} for num_shards={args.num_shards}")
        grouped_records = [
            item for index, item in enumerate(grouped_records) if index % args.num_shards == args.shard_rank
        ]

    os.makedirs(args.output_cache_dir, exist_ok=True)
    output_manifest_dir = os.path.dirname(args.output_manifest_path)
    if output_manifest_dir:
        os.makedirs(output_manifest_dir, exist_ok=True)

    processed_sources = 0
    processed_rows = 0
    with open(args.output_manifest_path, "w", encoding="utf-8") as manifest_handle:
        for source_sample_id, source_records in grouped_records:
            output_path = pack_path(args.output_cache_dir, source_sample_id)
            if overwrite or not os.path.exists(output_path):
                build_pack(
                    source_sample_id,
                    source_records,
                    output_path,
                    feature_dtype=feature_dtype,
                    score_dtype=score_dtype,
                )

            for record in source_records:
                updated = dict(record)
                updated["joint_pack_path"] = output_path
                updated["joint_pack_key"] = record["group_id"]
                updated["joint_pack_format"] = WINDOW_READY_SOURCE_JOINT_PACK_FORMAT
                manifest_handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
                processed_rows += 1
            processed_sources += 1
            if processed_sources % 100 == 0:
                manifest_handle.flush()
                print(
                    f"[window-ready-cache] processed_sources={processed_sources} processed_rows={processed_rows}",
                    flush=True,
                )

    print(f"[window-ready-cache] output_manifest_path={args.output_manifest_path}")
    print(f"[window-ready-cache] processed_sources={processed_sources}")
    print(f"[window-ready-cache] processed_rows={processed_rows}")


if __name__ == "__main__":
    main()

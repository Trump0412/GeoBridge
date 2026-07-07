"""Pack per-window corr graph files into source-level corr packs for Stage 1 v2."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch


CORR_SOURCE_PACK_FORMAT = "corr_source_pack_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_path", type=str, required=True)
    parser.add_argument("--output_cache_dir", type=str, required=True)
    parser.add_argument("--output_manifest_path", type=str, required=True)
    parser.add_argument("--max_sources", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--overwrite", type=str, default="False")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def group_records(records: Sequence[Dict]) -> List[Tuple[str, List[Dict]]]:
    grouped = defaultdict(list)
    for record in records:
        source_sample_id = record.get("source_sample_id", record["group_id"])
        grouped[source_sample_id].append(record)
    return sorted(grouped.items(), key=lambda item: item[0])


def pack_path(cache_dir: str, source_sample_id: str) -> str:
    prefix = source_sample_id[:2] if source_sample_id else "xx"
    return os.path.join(cache_dir, "corr_source_packs", prefix, f"{source_sample_id}.pt")


def build_pack(source_sample_id: str, source_records: Sequence[Dict], output_path: str) -> None:
    windows = {}
    for record in source_records:
        corr_cache_path = record.get("corr_cache_path", "")
        if not corr_cache_path or not os.path.exists(corr_cache_path):
            raise FileNotFoundError(f"corr_cache_path missing for group_id={record['group_id']}: {corr_cache_path}")
        corr_payload = torch.load(corr_cache_path, map_location="cpu", weights_only=True)
        windows[record["group_id"]] = {
            "neighbor_indices": corr_payload["neighbor_indices"],
            "neighbor_scores": corr_payload["neighbor_scores"],
        }

    payload = {
        "pack_format": CORR_SOURCE_PACK_FORMAT,
        "source_sample_id": source_sample_id,
        "windows": windows,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)


def main():
    args = parse_args()
    overwrite = parse_bool(args.overwrite)

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
                build_pack(source_sample_id, source_records, output_path)

            for record in source_records:
                updated = dict(record)
                updated["corr_pack_path"] = output_path
                updated["corr_pack_key"] = record["group_id"]
                updated["corr_pack_format"] = CORR_SOURCE_PACK_FORMAT
                manifest_handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
                processed_rows += 1
            processed_sources += 1
            if processed_sources % 100 == 0:
                manifest_handle.flush()
                print(
                    f"[corr-pack] processed_sources={processed_sources} processed_rows={processed_rows}",
                    flush=True,
                )

    print(f"[corr-pack] output_manifest_path={args.output_manifest_path}")
    print(f"[corr-pack] processed_sources={processed_sources}")
    print(f"[corr-pack] processed_rows={processed_rows}")


if __name__ == "__main__":
    main()

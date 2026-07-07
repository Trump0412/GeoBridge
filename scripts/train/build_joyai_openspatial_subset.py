#!/usr/bin/env python3
"""Build a filtered JoyAI-OpenSpatial subset into GeoThinker-style jsonl + image files."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
import subprocess

if os.environ.get("GEOBRIDGE_ALLOW_HF_TRANSFER") != "1":
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


REPO_BASE_URL = "https://hf-mirror.com/datasets/jdopensource/JoyAI-Image-OpenSpatial/resolve/main"
DEFAULT_ALLOWED_SOURCES = [
    "arkitscenes",
    "hypersim",
    "matterport3d",
    "wildrgbd",
    "ego-exo4d",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--target-samples", type=int, default=100000)
    parser.add_argument("--shard-count", type=int, default=7569)
    parser.add_argument("--start-shard", type=int, default=1)
    parser.add_argument("--base-url", default=REPO_BASE_URL)
    parser.add_argument("--tmp-dir", default="")
    parser.add_argument("--annotation-path", default="data/train/joyai_openspatial_100k.jsonl")
    parser.add_argument("--image-dir", default="data/media/JoyAI-OpenSpatial-100k")
    parser.add_argument("--summary-path", default="data/train/joyai_openspatial_100k.summary.json")
    parser.add_argument("--allow-source", action="append", dest="allow_sources", default=[])
    parser.add_argument("--keep-shards", action="store_true")
    return parser.parse_args()


def normalize_source(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def now() -> str:
    return time.strftime("%F %T")


def remote_size(url: str) -> int | None:
    return None


def download_file(url: str, dest_path: Path) -> int:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path.stat().st_size

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if "jdopensource/JoyAI-Image-OpenSpatial/resolve/main/" in url:
        filename = url.split("/resolve/main/", 1)[1]
        local_dir = dest_path.parent / ".hf_download"
        last_error: Exception | None = None
        for attempt in range(1, 7):
            try:
                downloaded = Path(
                    hf_hub_download(
                        repo_id="jdopensource/JoyAI-Image-OpenSpatial",
                        repo_type="dataset",
                        filename=filename,
                        endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                        local_dir=str(local_dir),
                    )
                )
                tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
                tmp_path.write_bytes(downloaded.read_bytes())
                os.replace(tmp_path, dest_path)
                return dest_path.stat().st_size
            except Exception as exc:
                last_error = exc
                if attempt == 6:
                    break
                wait_seconds = min(60, 2 * attempt)
                print(
                    json.dumps(
                        {
                            "ts": now(),
                            "event": "hf_download_retry",
                            "file": filename,
                            "attempt": attempt,
                            "wait_seconds": wait_seconds,
                            "error": repr(exc)[:500],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise RuntimeError(f"failed to download {filename} after retries") from last_error

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--speed-time",
        "30",
        "--speed-limit",
        "1024",
        "-C",
        "-",
        "-o",
        str(tmp_path),
        url,
    ]
    subprocess.run(command, check=True)
    os.replace(tmp_path, dest_path)
    return dest_path.stat().st_size


def image_extension(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            fmt = (image.format or "png").lower()
    except Exception:
        fmt = "png"
    if fmt == "jpeg":
        return ".jpg"
    if not fmt.startswith("."):
        return f".{fmt}"
    return fmt


def ensure_conversations(raw_value):
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    raise TypeError(f"unsupported conversations type: {type(raw_value)!r}")


def ensure_images(raw_value):
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    raise TypeError(f"unsupported images type: {type(raw_value)!r}")


def save_record_images(record_id: str, source: str, images, image_root: Path, media_root: Path) -> tuple[list[str], int]:
    rel_paths = []
    total_bytes = 0
    source_dir = image_root / source
    source_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(images):
        if not isinstance(item, dict):
            continue
        image_bytes = item.get("bytes")
        if not isinstance(image_bytes, (bytes, bytearray)):
            continue
        suffix = image_extension(bytes(image_bytes))
        file_name = f"{record_id}_{idx:02d}{suffix}"
        out_path = source_dir / file_name
        if not out_path.exists():
            out_path.write_bytes(bytes(image_bytes))
        total_bytes += out_path.stat().st_size
        rel_paths.append(os.path.relpath(out_path, media_root))
    return rel_paths, total_bytes


def write_summary(summary_path: Path, payload: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    media_root = repo_root / "data" / "media"
    image_root = repo_root / args.image_dir
    annotation_path = repo_root / args.annotation_path
    summary_path = repo_root / args.summary_path
    tmp_dir = Path(args.tmp_dir).resolve() if args.tmp_dir else repo_root / "tmp" / "joyai_openspatial_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    allowed = args.allow_sources or DEFAULT_ALLOWED_SOURCES
    allowed_norm = {normalize_source(item): item for item in allowed}

    started_at = now()
    started_epoch = time.time()
    open_mode = "w"
    next_shard = args.start_shard
    last_completed_shard = args.start_shard - 1

    saved_samples = 0
    saved_images = 0
    downloaded_bytes = 0
    saved_image_bytes = 0
    processed_shards = 0
    seen_source_counts: Counter[str] = Counter()
    saved_source_counts: Counter[str] = Counter()

    if summary_path.exists() != annotation_path.exists():
        raise FileExistsError(
            "joyai progress files are inconsistent; annotation and summary must either both exist or both be absent"
        )
    resume_summary = load_summary(summary_path)
    if resume_summary:
        last_completed_shard = int(resume_summary.get("last_completed_shard", args.start_shard - 1))
        next_shard = max(args.start_shard, last_completed_shard + 1)
        saved_samples = int(resume_summary.get("saved_samples", 0))
        saved_images = int(resume_summary.get("saved_images", 0))
        downloaded_bytes = int(resume_summary.get("downloaded_bytes", 0))
        saved_image_bytes = int(resume_summary.get("saved_image_bytes", 0))
        processed_shards = int(resume_summary.get("processed_shards", 0))
        seen_source_counts = Counter(resume_summary.get("seen_source_counts", {}))
        saved_source_counts = Counter(resume_summary.get("saved_source_counts", {}))
        started_at = str(resume_summary.get("started_at", started_at))
        started_epoch = float(resume_summary.get("started_epoch", started_epoch))
        open_mode = "a"

    print(
        json.dumps(
            {
                "ts": started_at,
                "repo_root": str(repo_root),
                "target_samples": args.target_samples,
                "allowed_sources": allowed,
                "shard_range": [args.start_shard, args.shard_count],
                "resume_mode": open_mode == "a",
                "resume_next_shard": next_shard,
                "resume_saved_samples": saved_samples,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with annotation_path.open(open_mode, encoding="utf-8") as out_handle:
        for shard_idx in range(next_shard, args.shard_count + 1):
            if saved_samples >= args.target_samples:
                break
            shard_name = f"data/{shard_idx:06d}.parquet"
            shard_url = f"{args.base_url.rstrip('/')}/{shard_name}"
            shard_path = tmp_dir / f"{shard_idx:06d}.parquet"
            expected_size = remote_size(shard_url)
            downloaded_bytes += download_file(shard_url, shard_path)
            processed_shards += 1
            last_completed_shard = shard_idx

            table = pq.read_table(shard_path)
            shard_rows = table.to_pylist()
            kept_in_shard = 0
            for record in shard_rows:
                source_raw = str(record.get("data_source", "")).strip()
                source_key = normalize_source(source_raw)
                seen_source_counts[source_raw] += 1
                if source_key not in allowed_norm:
                    continue

                conversations = ensure_conversations(record["conversations"])
                images = ensure_images(record["images"])
                rel_paths, bytes_written = save_record_images(
                    record_id=str(record["id"]),
                    source=source_raw,
                    images=images,
                    image_root=image_root,
                    media_root=media_root,
                )
                if not rel_paths:
                    continue

                item = {
                    "id": str(record["id"]),
                    "data_source": source_raw,
                    "conversations": conversations,
                    "images": rel_paths,
                }
                out_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                saved_samples += 1
                kept_in_shard += 1
                saved_source_counts[source_raw] += 1
                saved_images += len(rel_paths)
                saved_image_bytes += bytes_written
                if saved_samples >= args.target_samples:
                    break

            out_handle.flush()
            if not args.keep_shards:
                shard_path.unlink(missing_ok=True)

            accepted_ratio = saved_samples / max(sum(seen_source_counts.values()), 1)
            avg_shard_bytes = downloaded_bytes / processed_shards
            avg_image_bytes = saved_image_bytes / max(saved_images, 1)
            elapsed_seconds = max(time.time() - started_epoch, 1.0)
            samples_per_second = saved_samples / elapsed_seconds
            estimated_shards = (
                int(processed_shards * args.target_samples / max(saved_samples, 1))
                if saved_samples
                else None
            )
            estimate = {
                "ts": now(),
                "processed_shards": processed_shards,
                "last_shard": shard_name,
                "expected_size": expected_size,
                "saved_samples": saved_samples,
                "saved_images": saved_images,
                "accepted_ratio": round(accepted_ratio, 4),
                "avg_shard_mib": round(avg_shard_bytes / 1024**2, 2),
                "avg_image_kib": round(avg_image_bytes / 1024, 2),
                "saved_source_counts": dict(saved_source_counts),
                "estimated_raw_download_gib_for_target": round(
                    ((downloaded_bytes / max(saved_samples, 1)) * args.target_samples) / 1024**3,
                    2,
                ),
                "estimated_saved_image_gib_for_target": round(
                    (avg_image_bytes * args.target_samples) / 1024**3,
                    2,
                ),
                "elapsed_hours": round(elapsed_seconds / 3600, 2),
                "estimated_total_hours_for_target": round(
                    (args.target_samples / max(samples_per_second, 1e-9)) / 3600,
                    2,
                ),
                "estimated_remaining_hours": round(
                    (max(args.target_samples - saved_samples, 0) / max(samples_per_second, 1e-9)) / 3600,
                    2,
                ),
            }
            if estimated_shards is not None:
                estimate["estimated_shards_from_avg_keep_rate"] = estimated_shards
            write_summary(
                summary_path,
                {
                    "started_at": started_at,
                    "started_epoch": started_epoch,
                    "last_updated_at": estimate["ts"],
                    "target_samples": args.target_samples,
                    "saved_samples": saved_samples,
                    "saved_images": saved_images,
                    "processed_shards": processed_shards,
                    "last_completed_shard": last_completed_shard,
                    "next_shard": shard_idx + 1,
                    "downloaded_bytes": downloaded_bytes,
                    "saved_image_bytes": saved_image_bytes,
                    "allowed_sources": allowed,
                    "seen_source_counts": dict(seen_source_counts),
                    "saved_source_counts": dict(saved_source_counts),
                    "annotation_path": str(annotation_path),
                    "image_root": str(image_root),
                    "tmp_dir": str(tmp_dir),
                    "complete": False,
                },
            )
            print(json.dumps(estimate, ensure_ascii=False), flush=True)

    finished_at = now()
    payload = {
        "started_at": started_at,
        "started_epoch": started_epoch,
        "finished_at": finished_at,
        "last_updated_at": finished_at,
        "target_samples": args.target_samples,
        "saved_samples": saved_samples,
        "saved_images": saved_images,
        "processed_shards": processed_shards,
        "last_completed_shard": last_completed_shard,
        "next_shard": min(last_completed_shard + 1, args.shard_count + 1),
        "downloaded_bytes": downloaded_bytes,
        "saved_image_bytes": saved_image_bytes,
        "allowed_sources": allowed,
        "seen_source_counts": dict(seen_source_counts),
        "saved_source_counts": dict(saved_source_counts),
        "annotation_path": str(annotation_path),
        "image_root": str(image_root),
        "tmp_dir": str(tmp_dir),
        "complete": saved_samples >= args.target_samples,
    }
    write_summary(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

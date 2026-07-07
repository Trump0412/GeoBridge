#!/usr/bin/env python3
"""Download a Hugging Face dataset incrementally until a local size target is reached."""

from __future__ import annotations

import argparse
import fnmatch
import os
import time
from typing import Iterable, List

from huggingface_hub import hf_hub_download, repo_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--target-gib", type=float, required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--include-pattern", action="append", default=[])
    parser.add_argument("--exclude-pattern", action="append", default=[])
    parser.add_argument("--scan-interval-files", type=int, default=1)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%F %T")


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            full_path = os.path.join(root, name)
            try:
                total += os.path.getsize(full_path)
            except OSError:
                continue
    return total


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def filter_files(paths: Iterable[str], include_patterns: List[str], exclude_patterns: List[str]) -> List[str]:
    filtered = []
    for path in paths:
        if include_patterns and not matches_any(path, include_patterns):
            continue
        if exclude_patterns and matches_any(path, exclude_patterns):
            continue
        filtered.append(path)
    return sorted(filtered)


def main() -> int:
    args = parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)
    target_bytes = int(args.target_gib * 1024**3)

    print(
        {
            "ts": now(),
            "repo_id": args.repo_id,
            "repo_type": args.repo_type,
            "dest_dir": args.dest_dir,
            "target_gib": args.target_gib,
            "include_patterns": args.include_pattern,
            "exclude_patterns": args.exclude_pattern,
        },
        flush=True,
    )

    info = repo_info(args.repo_id, repo_type=args.repo_type, files_metadata=True)
    all_paths = [sibling.rfilename for sibling in info.siblings]
    filtered_paths = filter_files(all_paths, args.include_pattern, args.exclude_pattern)
    file_sizes = {sibling.rfilename: getattr(sibling, "size", None) or 0 for sibling in info.siblings}
    print({"ts": now(), "candidate_files": len(filtered_paths)}, flush=True)

    downloaded_bytes = dir_size(args.dest_dir)
    print({"ts": now(), "existing_gib": round(downloaded_bytes / 1024**3, 3)}, flush=True)
    if downloaded_bytes >= target_bytes:
        print({"ts": now(), "status": "already_at_target"}, flush=True)
        return 0

    for index, path in enumerate(filtered_paths, start=1):
        if index == 1 or index % max(args.scan_interval_files, 1) == 0:
            downloaded_bytes = dir_size(args.dest_dir)
        if downloaded_bytes >= target_bytes:
            print(
                {
                    "ts": now(),
                    "status": "target_reached",
                    "downloaded_gib": round(downloaded_bytes / 1024**3, 3),
                    "files_seen": index - 1,
                },
                flush=True,
            )
            return 0

        local_path = os.path.join(args.dest_dir, path)
        remote_size = file_sizes.get(path, 0)
        if os.path.exists(local_path):
            try:
                local_size = os.path.getsize(local_path)
            except OSError:
                local_size = -1
            if remote_size > 0 and local_size == remote_size:
                continue

        print(
            {
                "ts": now(),
                "download": path,
                "remote_gib": round(remote_size / 1024**3, 3),
                "downloaded_gib": round(downloaded_bytes / 1024**3, 3),
            },
            flush=True,
        )
        hf_hub_download(
            repo_id=args.repo_id,
            filename=path,
            repo_type=args.repo_type,
            local_dir=args.dest_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        downloaded_bytes = dir_size(args.dest_dir)
        print({"ts": now(), "done": path, "downloaded_gib": round(downloaded_bytes / 1024**3, 3)}, flush=True)

    downloaded_bytes = dir_size(args.dest_dir)
    print({"ts": now(), "status": "repo_exhausted", "downloaded_gib": round(downloaded_bytes / 1024**3, 3)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

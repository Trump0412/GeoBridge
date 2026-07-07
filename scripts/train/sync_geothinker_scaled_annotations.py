#!/usr/bin/env python3
"""Sync official GeoThinker scaled-regime train annotations via hf-mirror."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


BASE_URL = "https://hf-mirror.com/lihy285/GeoThinker/resolve/main"
DEFAULT_FILES = [
    "train/mindcube_10k.json",
    "train/vlm3r_vsi_205k.json",
    "train/vlm3r_vsi_205k_16frames.json",
    "train/vlm3r_vsi_205k_32frames.json",
    "train/vlm3r_vst_132k.json",
    "train/vlm3r_vst_132k_16frames.json",
    "train/vlm3r_vst_132k_32frames.json",
    "train/vsi_590k.json",
    "train/vsi_590k_16frame.json",
    "train/vsi_590k_32frame.json",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--file", action="append", dest="files", default=[])
    parser.add_argument("--skip-vsi-symlink", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(base_url: str, rel_path: str, dest_path: Path) -> tuple[int, int | None, bool]:
    url = f"{base_url.rstrip('/')}/{rel_path}"
    expected = None
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path.stat().st_size, expected, False

    ensure_parent(dest_path)
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

    return dest_path.stat().st_size, expected, True


def ensure_vsi_symlink(repo_root: Path) -> None:
    link_path = repo_root / "data" / "media" / "VSI-590K"
    target_path = repo_root / "data" / "VSI-590K"
    if link_path.exists() or link_path.is_symlink():
        print(f"[sync-annotations] keep existing VSI-590K entry: {link_path}")
        return
    if not target_path.exists():
        print(f"[sync-annotations] skip VSI-590K symlink; target missing: {target_path}")
        return
    link_path.symlink_to(Path("..") / "VSI-590K")
    print(f"[sync-annotations] created symlink: {link_path} -> ../VSI-590K")


def iter_files(args: argparse.Namespace) -> Iterable[str]:
    if args.files:
        return args.files
    return DEFAULT_FILES


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo root not found: {repo_root}")

    total_bytes = 0
    downloaded_bytes = 0
    updated = 0
    skipped = 0

    for rel_path in iter_files(args):
        dest_path = repo_root / "data" / rel_path.removeprefix("train/")
        if rel_path.startswith("train/"):
            dest_path = repo_root / "data" / "train" / rel_path.split("/", 1)[1]

        size, expected, changed = download_file(args.base_url, rel_path, dest_path)
        total_bytes += size
        downloaded_bytes += size if changed else 0
        updated += int(changed)
        skipped += int(not changed)
        print(
            f"[sync-annotations] {'downloaded' if changed else 'reused'} {rel_path} "
            f"size={size} expected={expected}",
            flush=True,
        )

    if not args.skip_vsi_symlink:
        ensure_vsi_symlink(repo_root)

    print(
        f"[sync-annotations] summary repo_root={repo_root} updated={updated} "
        f"reused={skipped} total_bytes={total_bytes} downloaded_bytes={downloaded_bytes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

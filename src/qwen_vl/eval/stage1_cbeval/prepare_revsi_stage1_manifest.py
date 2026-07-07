"""Prepare a ReVSI-derived Stage1 manifest by decoding sampled video frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence

import cv2


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_scene_rows(scene_json_path: Path) -> List[Dict]:
    payload = json.loads(scene_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {scene_json_path}, got {type(payload).__name__}")
    scene_rows: List[Dict] = []
    for item in payload:
        if isinstance(item, dict):
            scene_id = str(item["scene_id"])
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "dataset": str(item.get("dataset", "unknown")),
                    "question_count": int(item.get("question_count", 0)),
                }
            )
        else:
            scene_rows.append({"scene_id": str(item), "dataset": "unknown", "question_count": 0})
    return scene_rows


def load_sampled_frame_indices(metadata_path: Path, sampled_frame_key: str) -> Dict[str, List[int]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {metadata_path}, got {type(payload).__name__}")
    output: Dict[str, List[int]] = {}
    for scene_id, scene_payload in payload.items():
        if not isinstance(scene_payload, dict) or sampled_frame_key not in scene_payload:
            raise KeyError(f"Missing sampled_frame_key={sampled_frame_key} for scene_id={scene_id}")
        output[str(scene_id)] = [int(value) for value in scene_payload[sampled_frame_key]]
    return output


def ensure_extracted_video(
    archive: zipfile.ZipFile,
    *,
    subset_dir: str,
    scene_id: str,
    video_root: Path,
    overwrite: bool,
) -> tuple[Path, str]:
    member = f"{subset_dir}/{scene_id}.mp4"
    try:
        archive.getinfo(member)
    except KeyError as exc:
        raise FileNotFoundError(f"Missing ReVSI video member: {member}") from exc
    target_path = video_root / member
    if overwrite and target_path.exists():
        target_path.unlink()
    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        archive.extract(member, path=video_root)
    return target_path, member


def decode_video_frames(video_path: Path) -> List:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frames: List = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


def ensure_frame_images(
    *,
    frames,
    scene_id: str,
    frame_root: Path,
    image_format: str,
    overwrite: bool,
) -> List[str]:
    scene_dir = frame_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = [scene_dir / f"frame_{index:04d}.{image_format}" for index in range(len(frames))]
    if overwrite or any(not path.exists() for path in frame_paths):
        for frame_path, frame in zip(frame_paths, frames):
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to write frame image: {frame_path}")
    return [str(path.resolve()) for path in frame_paths]


def stable_group_id(scene_id: str, start_index: int, window_size: int, subset_dir: str) -> str:
    digest = hashlib.sha1()
    digest.update(f"revsi:{subset_dir}:{scene_id}:{start_index}:{window_size}".encode("utf-8"))
    return digest.hexdigest()


def build_window_records(
    *,
    scene_row: Dict,
    frame_paths: Sequence[str],
    sampled_frame_indices: Sequence[int],
    subset_dir: str,
    video_path: str,
    video_member: str,
    window_size: int,
    window_stride: int,
    max_windows_per_scene: int,
) -> List[Dict]:
    if len(frame_paths) != len(sampled_frame_indices):
        raise ValueError(
            f"scene_id={scene_row['scene_id']} frame_paths={len(frame_paths)} "
            f"!= sampled_frame_indices={len(sampled_frame_indices)}"
        )
    if len(frame_paths) < window_size:
        raise ValueError(
            f"scene_id={scene_row['scene_id']} has only {len(frame_paths)} decoded frames, "
            f"smaller than window_size={window_size}"
        )
    starts = list(range(0, len(frame_paths) - window_size + 1, window_stride))
    if max_windows_per_scene > 0:
        starts = starts[:max_windows_per_scene]
    records: List[Dict] = []
    for window_index, start_index in enumerate(starts):
        end_index = start_index + window_size
        records.append(
            {
                "group_id": stable_group_id(scene_row["scene_id"], start_index, window_size, subset_dir),
                "source_dataset": scene_row["dataset"],
                "source_sample_id": scene_row["scene_id"],
                "video_id": scene_row["scene_id"],
                "image_group_id": scene_row["scene_id"],
                "frame_paths": list(frame_paths[start_index:end_index]),
                "sampled_frame_indices": list(sampled_frame_indices[start_index:end_index]),
                "valid_frame_mask": [True] * window_size,
                "question_type": "revsi_eval_window",
                "window_id": f"window_{window_index}",
                "cache_window_mode": f"revsi_{subset_dir}_w{window_size}_s{window_stride}",
                "source_video_path": str(video_path),
                "source_video_member": video_member,
                "source_num_frames": len(frame_paths),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", type=str, required=True)
    parser.add_argument("--video_zip_path", type=str, required=True)
    parser.add_argument("--sampled_frame_indices_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subset_dir", type=str, default="32_frame")
    parser.add_argument("--sampled_frame_key", type=str, default="32-frame")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--max_windows_per_scene", type=int, default=4)
    parser.add_argument("--max_scenes", type=int, default=-1)
    parser.add_argument("--image_format", type=str, default="png")
    parser.add_argument("--overwrite", type=str, default="False")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overwrite = parse_bool(args.overwrite)
    scene_json = Path(args.scene_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = output_dir / "videos"
    frame_root = output_dir / "frames"

    scene_rows = load_scene_rows(scene_json)
    if args.max_scenes > 0:
        scene_rows = scene_rows[: args.max_scenes]
    sampled_frame_indices_map = load_sampled_frame_indices(
        Path(args.sampled_frame_indices_json),
        args.sampled_frame_key,
    )

    manifest_rows: List[Dict] = []
    decode_report_rows: List[Dict] = []
    with zipfile.ZipFile(args.video_zip_path) as archive:
        for scene_row in scene_rows:
            scene_id = scene_row["scene_id"]
            if scene_id not in sampled_frame_indices_map:
                raise KeyError(f"Missing sampled frame indices for scene_id={scene_id}")
            video_path, video_member = ensure_extracted_video(
                archive,
                subset_dir=args.subset_dir,
                scene_id=scene_id,
                video_root=video_root,
                overwrite=overwrite,
            )
            frames = decode_video_frames(video_path)
            sampled_frame_indices = sampled_frame_indices_map[scene_id]
            if len(frames) != len(sampled_frame_indices):
                raise ValueError(
                    f"scene_id={scene_id} decoded_frames={len(frames)} "
                    f"!= sampled_frame_indices={len(sampled_frame_indices)}"
                )
            frame_paths = ensure_frame_images(
                frames=frames,
                scene_id=scene_id,
                frame_root=frame_root,
                image_format=args.image_format,
                overwrite=overwrite,
            )
            manifest_rows.extend(
                build_window_records(
                    scene_row=scene_row,
                    frame_paths=frame_paths,
                    sampled_frame_indices=sampled_frame_indices,
                    subset_dir=args.subset_dir,
                    video_path=str(video_path.resolve()),
                    video_member=video_member,
                    window_size=args.window_size,
                    window_stride=args.window_stride,
                    max_windows_per_scene=args.max_windows_per_scene,
                )
            )
            decode_report_rows.append(
                {
                    "scene_id": scene_id,
                    "dataset": scene_row["dataset"],
                    "decoded_frames": len(frames),
                    "question_count": scene_row.get("question_count", 0),
                    "video_path": str(video_path.resolve()),
                    "video_member": video_member,
                }
            )

    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(output_dir / "scene_decode_report.jsonl", decode_report_rows)
    write_json(
        output_dir / "manifest_meta.json",
        {
            "scene_json": str(scene_json),
            "video_zip_path": str(Path(args.video_zip_path)),
            "sampled_frame_indices_json": str(Path(args.sampled_frame_indices_json)),
            "subset_dir": args.subset_dir,
            "sampled_frame_key": args.sampled_frame_key,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "max_windows_per_scene": args.max_windows_per_scene,
            "max_scenes": args.max_scenes,
            "num_scenes": len(scene_rows),
            "num_manifest_rows": len(manifest_rows),
        },
    )


if __name__ == "__main__":
    main()

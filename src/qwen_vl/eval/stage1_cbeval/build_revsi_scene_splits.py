"""Build scene-level held-out splits for ReVSI-backed Stage1 evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pyarrow.parquet as pq


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_scene_id(value) -> str:
    return str(value)


def load_parquet_rows(parquet_path: Path) -> List[Dict]:
    table = pq.read_table(parquet_path)
    return table.to_pylist()


def load_video_scene_ids(video_zip_path: Path | None, subset_dir: str) -> set[str] | None:
    if video_zip_path is None:
        return None
    with zipfile.ZipFile(video_zip_path) as handle:
        return {
            Path(member).stem
            for member in handle.namelist()
            if member.startswith(f"{subset_dir}/") and member.endswith(".mp4")
        }


def load_sampled_frame_keys(metadata_path: Path | None) -> set[str] | None:
    if metadata_path is None:
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in sampled frame metadata, got {type(payload).__name__}")
    return {str(key) for key in payload}


def build_scene_inventory(
    rows: Sequence[Dict],
    *,
    available_video_scene_ids: set[str] | None,
    sampled_frame_scene_ids: set[str] | None,
) -> tuple[List[Dict], Dict[str, List[Dict]]]:
    question_rows_by_scene: Dict[str, List[Dict]] = defaultdict(list)
    scene_meta: Dict[str, Dict] = {}
    for row in rows:
        scene_id = normalize_scene_id(row["scene_id"])
        dataset = str(row.get("dataset", "unknown"))
        question_type = str(row.get("question_type", "unknown"))
        scene_row = scene_meta.setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "dataset": dataset,
                "question_count": 0,
                "question_type_histogram": Counter(),
                "num_frames_values": set(),
            },
        )
        if scene_row["dataset"] != dataset:
            raise ValueError(f"Conflicting dataset for scene_id={scene_id}: {scene_row['dataset']} vs {dataset}")
        scene_row["question_count"] += 1
        scene_row["question_type_histogram"][question_type] += 1
        scene_row["num_frames_values"].add(str(row.get("num_frames", "")))
        question_rows_by_scene[scene_id].append(dict(row))

    inventory: List[Dict] = []
    for scene_id, scene_row in scene_meta.items():
        has_video = scene_id in available_video_scene_ids if available_video_scene_ids is not None else True
        has_sampled_frame_indices = (
            scene_id in sampled_frame_scene_ids if sampled_frame_scene_ids is not None else True
        )
        inventory.append(
            {
                "scene_id": scene_id,
                "dataset": scene_row["dataset"],
                "question_count": scene_row["question_count"],
                "question_type_histogram": dict(sorted(scene_row["question_type_histogram"].items())),
                "num_frames_values": sorted(scene_row["num_frames_values"]),
                "has_video": bool(has_video),
                "has_sampled_frame_indices": bool(has_sampled_frame_indices),
            }
        )
    inventory.sort(key=lambda row: (row["dataset"], row["scene_id"]))
    return inventory, question_rows_by_scene


def filter_inventory_for_runtime(inventory: Sequence[Dict]) -> List[Dict]:
    filtered = [row for row in inventory if row["has_video"] and row["has_sampled_frame_indices"]]
    if not filtered:
        raise ValueError("No ReVSI scenes remain after requiring both video and sampled-frame metadata.")
    return filtered


def allocate_counts(available_counts: Dict[str, int], total: int) -> Dict[str, int]:
    if total < 0:
        raise ValueError(f"Requested total must be non-negative, got {total}")
    available_total = sum(int(value) for value in available_counts.values())
    if total > available_total:
        raise ValueError(f"Requested {total} scenes but only {available_total} are available")
    if total == 0:
        return {key: 0 for key in sorted(available_counts)}

    exact = {
        key: (total * int(count) / available_total if available_total > 0 else 0.0)
        for key, count in available_counts.items()
    }
    allocation = {key: min(int(math.floor(value)), int(available_counts[key])) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    ranked_keys = sorted(
        available_counts,
        key=lambda key: (exact[key] - allocation[key], available_counts[key], key),
        reverse=True,
    )
    while remaining > 0:
        progress = False
        for key in ranked_keys:
            if allocation[key] >= int(available_counts[key]):
                continue
            allocation[key] += 1
            remaining -= 1
            progress = True
            if remaining == 0:
                break
        if not progress:
            raise RuntimeError("Unable to allocate the requested ReVSI scene counts")
    return {key: allocation.get(key, 0) for key in sorted(available_counts)}


def sample_scenes_by_dataset(
    inventory: Sequence[Dict],
    *,
    selection_scenes: int,
    locked_test_scenes: int,
    seed: int,
) -> tuple[List[Dict], List[Dict], List[Dict], Dict[str, Dict[str, int]]]:
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for row in inventory:
        by_dataset[str(row["dataset"])].append(dict(row))

    for dataset, items in by_dataset.items():
        items.sort(key=lambda row: row["scene_id"])
        random.Random(f"{seed}:{dataset}").shuffle(items)

    available_counts = {dataset: len(items) for dataset, items in by_dataset.items()}
    selection_counts = allocate_counts(available_counts, selection_scenes)
    remaining_counts = {
        dataset: available_counts[dataset] - selection_counts[dataset]
        for dataset in sorted(available_counts)
    }
    locked_counts = allocate_counts(remaining_counts, locked_test_scenes)

    selection: List[Dict] = []
    locked_test: List[Dict] = []
    reserve: List[Dict] = []
    for dataset in sorted(by_dataset):
        items = by_dataset[dataset]
        select_end = selection_counts[dataset]
        test_end = select_end + locked_counts[dataset]
        selection.extend(items[:select_end])
        locked_test.extend(items[select_end:test_end])
        reserve.extend(items[test_end:])

    selection.sort(key=lambda row: (row["dataset"], row["scene_id"]))
    locked_test.sort(key=lambda row: (row["dataset"], row["scene_id"]))
    reserve.sort(key=lambda row: (row["dataset"], row["scene_id"]))
    counts_by_split = {
        "available": available_counts,
        "selection_dev": selection_counts,
        "locked_test": locked_counts,
        "reserve": {
            dataset: available_counts[dataset] - selection_counts[dataset] - locked_counts[dataset]
            for dataset in sorted(available_counts)
        },
    }
    return selection, locked_test, reserve, counts_by_split


def subset_question_rows(scene_rows: Sequence[Dict], question_rows_by_scene: Dict[str, List[Dict]]) -> List[Dict]:
    output: List[Dict] = []
    for scene_row in scene_rows:
        output.extend(question_rows_by_scene.get(scene_row["scene_id"], []))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--video_zip_path", type=str, default="")
    parser.add_argument("--sampled_frame_indices_json", type=str, default="")
    parser.add_argument("--subset_dir", type=str, default="32_frame")
    parser.add_argument("--selection_scenes", type=int, default=128)
    parser.add_argument("--locked_test_scenes", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_zip_path = Path(args.video_zip_path) if args.video_zip_path else None
    sampled_frame_indices_json = Path(args.sampled_frame_indices_json) if args.sampled_frame_indices_json else None

    rows = load_parquet_rows(parquet_path)
    available_video_scene_ids = load_video_scene_ids(video_zip_path, args.subset_dir)
    sampled_frame_scene_ids = load_sampled_frame_keys(sampled_frame_indices_json)
    inventory, question_rows_by_scene = build_scene_inventory(
        rows,
        available_video_scene_ids=available_video_scene_ids,
        sampled_frame_scene_ids=sampled_frame_scene_ids,
    )
    runtime_inventory = filter_inventory_for_runtime(inventory)
    selection, locked_test, reserve, counts_by_split = sample_scenes_by_dataset(
        runtime_inventory,
        selection_scenes=args.selection_scenes,
        locked_test_scenes=args.locked_test_scenes,
        seed=args.seed,
    )

    write_jsonl(output_dir / "scene_inventory.jsonl", inventory)
    write_json(output_dir / "selection_dev_scenes.json", selection)
    write_json(output_dir / "locked_test_scenes.json", locked_test)
    write_json(output_dir / "reserve_scenes.json", reserve)
    (output_dir / "selection_dev_scene_ids.txt").write_text(
        "\n".join(row["scene_id"] for row in selection) + "\n",
        encoding="utf-8",
    )
    (output_dir / "locked_test_scene_ids.txt").write_text(
        "\n".join(row["scene_id"] for row in locked_test) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reserve_scene_ids.txt").write_text(
        "\n".join(row["scene_id"] for row in reserve) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "selection_dev_questions.jsonl", subset_question_rows(selection, question_rows_by_scene))
    write_jsonl(output_dir / "locked_test_questions.jsonl", subset_question_rows(locked_test, question_rows_by_scene))
    write_jsonl(output_dir / "reserve_questions.jsonl", subset_question_rows(reserve, question_rows_by_scene))
    write_json(
        output_dir / "split_meta.json",
        {
            "parquet_path": str(parquet_path),
            "video_zip_path": str(video_zip_path) if video_zip_path else "",
            "sampled_frame_indices_json": str(sampled_frame_indices_json) if sampled_frame_indices_json else "",
            "subset_dir": args.subset_dir,
            "seed": args.seed,
            "selection_scenes": args.selection_scenes,
            "locked_test_scenes": args.locked_test_scenes,
            "reserve_scenes": len(reserve),
            "num_question_rows": len(rows),
            "num_inventory_scenes": len(inventory),
            "num_runtime_scenes": len(runtime_inventory),
            "counts_by_split": counts_by_split,
        },
    )


if __name__ == "__main__":
    main()

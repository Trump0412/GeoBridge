"""Build source-level held-out splits for Stage1 CBEval."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_GROUP_PRIORITY = (
    "source_sample_id",
    "source_id",
    "video_id",
    "image_group_id",
)


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def frame_paths_hash(entry: Dict) -> str:
    frame_paths = entry.get("frame_paths") or entry.get("image_paths") or []
    return stable_hash(sorted(str(path) for path in frame_paths))


def stable_group_key(entry: Dict, priority: Sequence[str] = DEFAULT_GROUP_PRIORITY) -> str:
    for key in priority:
        value = entry.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    frame_hash = frame_paths_hash(entry)
    if frame_hash:
        return frame_hash
    if entry.get("group_id"):
        return str(entry["group_id"])
    return stable_hash(json.dumps(entry, ensure_ascii=False, sort_keys=True))


def filter_rows(rows: Sequence[Dict], include_datasets: set[str], exclude_datasets: set[str]) -> List[Dict]:
    filtered: List[Dict] = []
    for row in rows:
        dataset_name = str(row.get("source_dataset", "unknown"))
        if include_datasets and dataset_name not in include_datasets:
            continue
        if exclude_datasets and dataset_name in exclude_datasets:
            continue
        filtered.append(dict(row))
    return filtered


def group_rows(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        source_key = stable_group_key(row)
        enriched = dict(row)
        enriched["source_group_key"] = source_key
        grouped[source_key].append(enriched)
    for items in grouped.values():
        items.sort(key=lambda item: (str(item.get("window_id", "")), str(item.get("group_id", ""))))
    return grouped


def choose_windows(source_key: str, rows: Sequence[Dict], *, max_windows_per_source: int, seed: int) -> List[Dict]:
    if max_windows_per_source <= 0 or len(rows) <= max_windows_per_source:
        return list(rows)
    local_seed = int(hashlib.sha1(f"{seed}:{source_key}:windows".encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(local_seed)
    chosen = rng.sample(list(rows), k=max_windows_per_source)
    chosen.sort(key=lambda item: (str(item.get("window_id", "")), str(item.get("group_id", ""))))
    return chosen


def collect_train_seen_sources(train_manifest: Path | None) -> set[str]:
    if train_manifest is None:
        return set()
    return {stable_group_key(row) for row in load_jsonl(train_manifest)}


def build_split_payload(
    grouped_rows: Dict[str, List[Dict]],
    source_keys: Sequence[str],
    *,
    max_windows_per_source: int,
    seed: int,
) -> List[Dict]:
    payload: List[Dict] = []
    for source_key in source_keys:
        payload.extend(
            choose_windows(
                source_key,
                grouped_rows[source_key],
                max_windows_per_source=max_windows_per_source,
                seed=seed,
            )
        )
    return payload


def build_leakage_report(
    train_seen_sources: set[str],
    selection_rows: Sequence[Dict],
    locked_rows: Sequence[Dict],
) -> Dict:
    selection_sources = {row["source_group_key"] for row in selection_rows}
    locked_sources = {row["source_group_key"] for row in locked_rows}
    selection_windows = {str(row.get("group_id", "")) for row in selection_rows}
    locked_windows = {str(row.get("group_id", "")) for row in locked_rows}
    selection_media = {frame_paths_hash(row) for row in selection_rows}
    locked_media = {frame_paths_hash(row) for row in locked_rows}

    report = {
        "num_train_sources": len(train_seen_sources),
        "num_selection_sources": len(selection_sources),
        "num_locked_test_sources": len(locked_sources),
        "train_selection_overlap": len(train_seen_sources & selection_sources),
        "train_test_overlap": len(train_seen_sources & locked_sources),
        "selection_test_overlap": len(selection_sources & locked_sources),
        "window_overlap": len(selection_windows & locked_windows),
        "media_path_hash_overlap": len(selection_media & locked_media),
    }
    report["ok"] = all(report[key] == 0 for key in (
        "train_selection_overlap",
        "train_test_overlap",
        "selection_test_overlap",
        "window_overlap",
        "media_path_hash_overlap",
    ))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, default="")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--selection_sources", type=int, required=True)
    parser.add_argument("--test_sources", type=int, required=True)
    parser.add_argument("--max_windows_per_source", type=int, default=4)
    parser.add_argument("--include_source_datasets", type=str, default="")
    parser.add_argument("--exclude_source_datasets", type=str, default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_datasets = set(parse_csv_list(args.include_source_datasets))
    exclude_datasets = set(parse_csv_list(args.exclude_source_datasets))
    rows = filter_rows(load_jsonl(manifest_path), include_datasets, exclude_datasets)
    grouped_rows = group_rows(rows)

    train_manifest = Path(args.train_manifest) if args.train_manifest else None
    train_seen_sources = collect_train_seen_sources(train_manifest)
    eligible_sources = [key for key in grouped_rows if key not in train_seen_sources]
    if args.selection_sources + args.test_sources > len(eligible_sources):
        raise ValueError(
            "Not enough held-out sources after train overlap removal: "
            f"need {args.selection_sources + args.test_sources}, got {len(eligible_sources)}"
        )

    rng = random.Random(args.seed)
    shuffled_sources = list(eligible_sources)
    rng.shuffle(shuffled_sources)
    selection_sources = sorted(shuffled_sources[: args.selection_sources])
    locked_sources = sorted(
        shuffled_sources[args.selection_sources : args.selection_sources + args.test_sources]
    )

    selection_rows = build_split_payload(
        grouped_rows,
        selection_sources,
        max_windows_per_source=args.max_windows_per_source,
        seed=args.seed,
    )
    locked_rows = build_split_payload(
        grouped_rows,
        locked_sources,
        max_windows_per_source=args.max_windows_per_source,
        seed=args.seed + 1,
    )
    leakage_report = build_leakage_report(train_seen_sources, selection_rows, locked_rows)

    write_json(output_dir / "split_meta.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "train_manifest": str(train_manifest) if train_manifest else "",
        "seed": args.seed,
        "selection_sources": args.selection_sources,
        "test_sources": args.test_sources,
        "max_windows_per_source": args.max_windows_per_source,
        "include_source_datasets": sorted(include_datasets),
        "exclude_source_datasets": sorted(exclude_datasets),
        "num_manifest_rows": len(rows),
        "num_unique_sources": len(grouped_rows),
        "num_eligible_sources": len(eligible_sources),
        "num_selection_rows": len(selection_rows),
        "num_locked_rows": len(locked_rows),
    })
    write_json(output_dir / "train_seen_sources.json", sorted(train_seen_sources))
    write_json(output_dir / "selection_dev_sources.json", selection_sources)
    write_json(output_dir / "locked_test_sources.json", locked_sources)
    write_jsonl(output_dir / "selection_dev_manifest.jsonl", selection_rows)
    write_jsonl(output_dir / "locked_test_manifest.jsonl", locked_rows)
    write_json(output_dir / "leakage_report.json", leakage_report)
    if not leakage_report["ok"]:
        raise SystemExit("Leakage report failed; refusing to keep an overlapping split.")


if __name__ == "__main__":
    main()

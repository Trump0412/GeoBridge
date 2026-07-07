"""Filter Stage 1 geometry manifests by valid frame count."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def count_valid_frames(record: dict) -> int:
    valid_frame_mask = record.get("valid_frame_mask")
    if valid_frame_mask is None:
        return len(record.get("frame_paths", []))
    return sum(1 for value in valid_frame_mask if bool(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Stage 1 manifest by minimum valid frames.")
    parser.add_argument("--input", required=True, help="Input manifest jsonl path.")
    parser.add_argument("--output", required=True, help="Output manifest jsonl path.")
    parser.add_argument("--min-valid-frames", type=int, default=5, help="Minimum valid frame count to keep.")
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional summary json path. If empty, summary is only printed.",
    )
    args = parser.parse_args()

    if args.min_valid_frames <= 0:
        raise ValueError("--min-valid-frames must be positive")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input manifest not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.summary_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_output)), exist_ok=True)

    total = 0
    kept = 0
    dropped = 0
    kept_counts: Counter[int] = Counter()
    dropped_counts: Counter[int] = Counter()

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            valid_frames = count_valid_frames(record)
            total += 1
            if valid_frames >= args.min_valid_frames:
                kept += 1
                kept_counts[valid_frames] += 1
                fout.write(json.dumps(record, ensure_ascii=True) + "\n")
            else:
                dropped += 1
                dropped_counts[valid_frames] += 1

    summary = {
        "input_manifest": os.path.abspath(args.input),
        "output_manifest": os.path.abspath(args.output),
        "min_valid_frames": int(args.min_valid_frames),
        "total_records": int(total),
        "kept_records": int(kept),
        "dropped_records": int(dropped),
        "kept_valid_frame_distribution": {str(key): int(value) for key, value in sorted(kept_counts.items())},
        "dropped_valid_frame_distribution": {str(key): int(value) for key, value in sorted(dropped_counts.items())},
    }

    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, ensure_ascii=True)
            fout.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

"""Split a JSONL eval manifest into round-robin shards."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="shard")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [output_dir / f"{args.prefix}_{index:02d}.jsonl" for index in range(args.num_shards)]
    counts = [0 for _ in range(args.num_shards)]
    handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    try:
        with open(args.manifest, "r", encoding="utf-8") as handle:
            row_index = 0
            for line in handle:
                if not line.strip():
                    continue
                shard_index = row_index % args.num_shards
                handles[shard_index].write(line)
                counts[shard_index] += 1
                row_index += 1
    finally:
        for handle in handles:
            handle.close()

    for shard_path, count in zip(shard_paths, counts):
        print(f"{shard_path}\t{count}")


if __name__ == "__main__":
    main()

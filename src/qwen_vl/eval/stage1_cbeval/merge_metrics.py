"""Merge sharded Stage1 CBEval metrics by window-weighted averaging."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_metrics(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_metrics", nargs="+", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    payloads = [load_metrics(path) for path in args.input_metrics]
    if not payloads:
        raise ValueError("input_metrics must not be empty")

    merged_methods: dict[str, dict] = {}
    for metrics_path, payload in zip(args.input_metrics, payloads):
        methods = payload.get("methods", {})
        for method_name, method_payload in methods.items():
            num_windows = int(method_payload.get("num_windows", 0))
            merged = merged_methods.setdefault(
                method_name,
                {
                    "kind": method_payload.get("kind", ""),
                    "checkpoint_path": method_payload.get("checkpoint_path", ""),
                    "num_windows": 0,
                    "metric_sums": {},
                    "sources": [],
                },
            )
            if merged["kind"] != method_payload.get("kind", ""):
                raise ValueError(f"kind mismatch for method {method_name}")
            if merged["checkpoint_path"] != method_payload.get("checkpoint_path", ""):
                raise ValueError(f"checkpoint_path mismatch for method {method_name}")
            merged["num_windows"] += num_windows
            merged["sources"].append(metrics_path)
            for key, value in method_payload.get("metrics", {}).items():
                merged["metric_sums"][key] = merged["metric_sums"].get(key, 0.0) + float(value) * num_windows

    merged_payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "merged_from": list(args.input_metrics),
            "num_inputs": len(args.input_metrics),
        },
        "methods": {},
    }
    for method_name, merged in merged_methods.items():
        denom = max(int(merged["num_windows"]), 1)
        merged_payload["methods"][method_name] = {
            "kind": merged["kind"],
            "checkpoint_path": merged["checkpoint_path"],
            "num_windows": int(merged["num_windows"]),
            "metrics": {
                key: value / denom
                for key, value in sorted(merged["metric_sums"].items())
            },
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

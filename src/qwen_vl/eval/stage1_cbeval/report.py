"""Render Stage1 CBEval tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


CSV_COLUMNS = [
    "method",
    "kind",
    "num_windows",
    "mccr_cos_primary",
    "mccr_l1_primary",
    "mccr_cos_corr_tube",
    "mccr_cos_frame_block",
    "corr_recall@5",
    "tcs_margin",
    "tcs",
    "shuffle_gap",
    "cfo_gap",
    "normal_score",
    "cbeval_score",
    "guard",
]


def load_payload(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_methods(metrics_payload: Dict, score_payload: Dict | None = None) -> List[Dict]:
    ranked_lookup = {}
    if score_payload:
        for row in score_payload.get("ranked_checkpoints", []):
            ranked_lookup[row["method"]] = row
    rows: List[Dict] = []
    for method_name, method_payload in metrics_payload["methods"].items():
        metrics = dict(method_payload["metrics"])
        ranked = ranked_lookup.get(method_name, {})
        rows.append({
            "method": method_name,
            "kind": method_payload["kind"],
            "num_windows": method_payload["num_windows"],
            "mccr_cos_primary": metrics.get("mccr_cos_primary", 0.0),
            "mccr_l1_primary": metrics.get("mccr_l1_primary", 0.0),
            "mccr_cos_corr_tube": metrics.get("mccr_cos_corr_tube", 0.0),
            "mccr_cos_frame_block": metrics.get("mccr_cos_frame_block", 0.0),
            "corr_recall@5": metrics.get("corr_recall@5", 0.0),
            "tcs_margin": metrics.get("tcs_margin", 0.0),
            "tcs": metrics.get("tcs", 0.0),
            "shuffle_gap": metrics.get("shuffle_gap", 0.0),
            "cfo_gap": metrics.get("cfo_gap", 0.0),
            "normal_score": metrics.get("normal_score", 0.0),
            "cbeval_score": ranked.get("cbeval_score", ""),
            "guard": ranked.get("guard", ""),
        })
    return rows


def write_metrics_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_main_table(path: Path, rows: List[Dict]) -> None:
    header = (
        "| Method | MCCR-Cos | Corr@5 | TCS-Margin | ShuffleGap | CFOGap | CBEval | Guard |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {method} | {mccr_cos_primary:.4f} | {corr_recall@5:.4f} | {tcs_margin:.4f} | "
            "{shuffle_gap:.4f} | {cfo_gap:.4f} | {cbeval_score} | {guard} |\n".format(**row)
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_json", type=str, required=True)
    parser.add_argument("--score_json", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = load_payload(Path(args.metrics_json))
    score_payload = load_payload(Path(args.score_json)) if args.score_json else None
    rows = flatten_methods(metrics_payload, score_payload)
    write_metrics_csv(output_dir / "metrics.csv", rows)
    write_main_table(output_dir / "main_table.md", rows)


if __name__ == "__main__":
    main()

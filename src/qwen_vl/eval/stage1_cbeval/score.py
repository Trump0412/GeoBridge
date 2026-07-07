"""Score and rank Stage1 CBEval checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from qwen_vl.eval.stage1_cbeval.report import main as render_report


def checkpoint_label_from_path(path: str) -> str:
    match = re.search(r"checkpoint-(\d+)", Path(path).name)
    if match:
        return f"FCP-c{match.group(1)}"
    return Path(path).stem


def _minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum - minimum < 1e-8:
        return {key: 0.5 for key in values}
    return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}


def guard_from_metrics(method_name: str, method_metrics: Dict[str, float], knn_metrics: Dict[str, float]) -> str:
    failures: List[str] = []
    suspicious: List[str] = []
    if method_metrics["mccr_cos_primary"] < knn_metrics["mccr_cos_primary"]:
        failures.append("mccr<g11_knn")
    if method_metrics["cos_rep_g11"] > 0.98:
        suspicious.append("cos_rep_g11_high")
    if method_metrics["cos_rep_g11"] < 0.10:
        suspicious.append("cos_rep_g11_low")
    if not failures and not suspicious:
        return "pass"
    if failures:
        return "fail:" + ",".join(failures + suspicious)
    return "warn:" + ",".join(suspicious)


def score_metrics_payload(metrics_payload: Dict) -> Dict:
    methods = metrics_payload["methods"]
    if "g11_knn" not in methods:
        raise ValueError("metrics.json is missing baseline method g11_knn")
    if "current_frame_only" not in methods:
        raise ValueError("metrics.json is missing baseline method current_frame_only")

    knn_metrics = methods["g11_knn"]["metrics"]
    checkpoint_methods = {
        method_name: method_payload["metrics"]
        for method_name, method_payload in methods.items()
        if method_payload["kind"] == "checkpoint"
    }
    raw_values = {}
    guards = {}
    for method_name, method_metrics in checkpoint_methods.items():
        raw_values[method_name] = {
            "mccr_gain": method_metrics["mccr_cos_primary"] - knn_metrics["mccr_cos_primary"],
            "tcs_gain": method_metrics["tcs"] - knn_metrics["tcs"],
            "shuffle_gap": method_metrics["shuffle_gap"],
            "cfo_gap": method_metrics["cfo_gap"],
        }
        guards[method_name] = guard_from_metrics(method_name, method_metrics, knn_metrics)

    normalized_mccr = _minmax_normalize({key: row["mccr_gain"] for key, row in raw_values.items()})
    normalized_tcs = _minmax_normalize({key: row["tcs_gain"] for key, row in raw_values.items()})
    normalized_shuffle = _minmax_normalize({key: row["shuffle_gap"] for key, row in raw_values.items()})
    normalized_cfo = _minmax_normalize({key: row["cfo_gap"] for key, row in raw_values.items()})

    ranked = []
    for method_name, method_metrics in checkpoint_methods.items():
        cbeval_score = (
            0.40 * normalized_mccr.get(method_name, 0.0)
            + 0.30 * normalized_tcs.get(method_name, 0.0)
            + 0.20 * normalized_shuffle.get(method_name, 0.0)
            + 0.10 * normalized_cfo.get(method_name, 0.0)
        )
        ranked.append({
            "method": method_name,
            "checkpoint_path": methods[method_name]["checkpoint_path"],
            "label": checkpoint_label_from_path(methods[method_name]["checkpoint_path"]),
            "guard": guards[method_name],
            "cbeval_score": cbeval_score,
            "raw_metrics": raw_values[method_name],
            "normalization": {
                "mccr_gain": normalized_mccr.get(method_name, 0.0),
                "tcs_gain": normalized_tcs.get(method_name, 0.0),
                "shuffle_gap": normalized_shuffle.get(method_name, 0.0),
                "cfo_gap": normalized_cfo.get(method_name, 0.0),
            },
        })

    ranked.sort(key=lambda row: row["cbeval_score"], reverse=True)
    return {
        "meta": metrics_payload.get("meta", {}),
        "ranked_checkpoints": ranked,
        "top2": ranked[:2],
        "guard_report": guards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    score_payload = score_metrics_payload(metrics_payload)
    (output_dir / "cbeval_score.json").write_text(
        json.dumps(score_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ranked_checkpoints.json").write_text(
        json.dumps(score_payload["ranked_checkpoints"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "guard_report.json").write_text(
        json.dumps(score_payload["guard_report"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    render_report_args = [
        "--metrics_json",
        str(metrics_path),
        "--score_json",
        str(output_dir / "cbeval_score.json"),
        "--output_dir",
        str(output_dir),
    ]
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = ["report.py", *render_report_args]
        render_report()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

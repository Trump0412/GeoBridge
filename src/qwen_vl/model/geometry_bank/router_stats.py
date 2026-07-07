"""Utilities for recording geometry-bank routing stats during evaluation."""

from __future__ import annotations

import atexit
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


CANDIDATE_NAMES = ("g11", "g17", "g23", "cont")


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class _LayerAccumulator:
    selection_histogram: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    drop_histogram: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    raw_candidate_norm_sum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    projected_candidate_norm_sum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    gate_mean_sum: float = 0.0
    gate_std_sum: float = 0.0
    count: int = 0
    hgb_gate_histogram: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hgb_gate_prob_sum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hgb_local_selection_histogram: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hgb_layer_scale_tanh_sum: float = 0.0
    hgb_hidden_norm_sum: float = 0.0
    hgb_local_delta_norm_sum: float = 0.0
    hgb_cont_delta_norm_sum: float = 0.0
    hgb_mixed_norm_sum: float = 0.0
    hgb_saliency_mean_sum: float = 0.0
    hgb_saliency_std_sum: float = 0.0
    hgb_local_entropy_mean_sum: float = 0.0
    hgb_layout_fallback_count: float = 0.0
    hgb_token_mismatch_count: float = 0.0
    hgb_overlap_ratio_sum: float = 0.0
    hgb_overlap_ratio_count: int = 0
    hgb_min_overlap_ratio: float = 1.0
    hgb_available_frames_sum: float = 0.0
    hgb_count: int = 0

    def update(
        self,
        selection_histogram: List[float],
        drop_histogram: List[float],
        raw_candidate_norm: List[float],
        projected_candidate_norm: List[float],
        gate_mean: float,
        gate_std: float,
        extra_metrics: Optional[Dict[str, object]] = None,
    ) -> None:
        self.selection_histogram = [
            current + value for current, value in zip(self.selection_histogram, selection_histogram)
        ]
        self.drop_histogram = [current + value for current, value in zip(self.drop_histogram, drop_histogram)]
        self.raw_candidate_norm_sum = [
            current + value for current, value in zip(self.raw_candidate_norm_sum, raw_candidate_norm)
        ]
        self.projected_candidate_norm_sum = [
            current + value for current, value in zip(self.projected_candidate_norm_sum, projected_candidate_norm)
        ]
        self.gate_mean_sum += gate_mean
        self.gate_std_sum += gate_std
        self.count += 1
        if extra_metrics:
            self._update_hgb(extra_metrics)

    def _update_hgb(self, extra_metrics: Dict[str, object]) -> None:
        def _as_list(key: str, expected_len: int) -> List[float]:
            values = extra_metrics.get(key)
            if values is None:
                return [0.0] * expected_len
            items = [float(value) for value in values]
            if len(items) < expected_len:
                items.extend([0.0] * (expected_len - len(items)))
            return items[:expected_len]

        self.hgb_gate_histogram = [
            current + value for current, value in zip(self.hgb_gate_histogram, _as_list("gate_histogram", 3))
        ]
        self.hgb_gate_prob_sum = [
            current + value for current, value in zip(self.hgb_gate_prob_sum, _as_list("gate_prob_mean", 3))
        ]
        self.hgb_local_selection_histogram = [
            current + value
            for current, value in zip(self.hgb_local_selection_histogram, _as_list("local_selection_histogram", 3))
        ]
        self.hgb_layer_scale_tanh_sum += float(extra_metrics.get("layer_scale_tanh", 0.0))
        self.hgb_hidden_norm_sum += float(extra_metrics.get("hidden_norm_mean", 0.0))
        self.hgb_local_delta_norm_sum += float(extra_metrics.get("local_delta_norm_mean", 0.0))
        self.hgb_cont_delta_norm_sum += float(extra_metrics.get("cont_delta_norm_mean", 0.0))
        self.hgb_mixed_norm_sum += float(extra_metrics.get("mixed_norm_mean", 0.0))
        self.hgb_saliency_mean_sum += float(extra_metrics.get("saliency_mean", 0.0))
        self.hgb_saliency_std_sum += float(extra_metrics.get("saliency_std", 0.0))
        self.hgb_local_entropy_mean_sum += float(extra_metrics.get("local_entropy_mean", 0.0))
        self.hgb_layout_fallback_count += float(extra_metrics.get("layout_fallback_count", 0.0))
        self.hgb_token_mismatch_count += float(extra_metrics.get("token_mismatch_count", 0.0))
        overlap_count = int(extra_metrics.get("overlap_ratio_count", 0) or 0)
        overlap_sum = float(extra_metrics.get("overlap_ratio_sum", 0.0))
        self.hgb_overlap_ratio_count += overlap_count
        self.hgb_overlap_ratio_sum += overlap_sum
        if overlap_count > 0:
            self.hgb_min_overlap_ratio = min(
                self.hgb_min_overlap_ratio,
                float(extra_metrics.get("min_overlap_ratio", 1.0)),
            )
        self.hgb_available_frames_sum += float(extra_metrics.get("available_frames", 0.0))
        self.hgb_count += 1

    def to_dict(self) -> Dict[str, object]:
        selection_total = sum(self.selection_histogram)
        drop_total = sum(self.drop_histogram)
        divisor = max(self.count, 1)
        payload = {
            "count": self.count,
            "selection_histogram": self.selection_histogram,
            "selection_ratio": [
                value / selection_total if selection_total > 0 else 0.0 for value in self.selection_histogram
            ],
            "drop_histogram": self.drop_histogram,
            "drop_ratio": [value / drop_total if drop_total > 0 else 0.0 for value in self.drop_histogram],
            "raw_candidate_norm_mean": [value / divisor for value in self.raw_candidate_norm_sum],
            "projected_candidate_norm_mean": [value / divisor for value in self.projected_candidate_norm_sum],
            "gate_mean": self.gate_mean_sum / divisor,
            "gate_std": self.gate_std_sum / divisor,
        }
        if self.hgb_count > 0:
            gate_total = sum(self.hgb_gate_histogram)
            local_total = sum(self.hgb_local_selection_histogram)
            hgb_divisor = max(self.hgb_count, 1)
            payload["hgb"] = {
                "count": self.hgb_count,
                "gate_histogram": self.hgb_gate_histogram,
                "gate_ratio": [value / gate_total if gate_total > 0 else 0.0 for value in self.hgb_gate_histogram],
                "gate_prob_mean": [value / hgb_divisor for value in self.hgb_gate_prob_sum],
                "local_selection_histogram": self.hgb_local_selection_histogram,
                "local_selection_ratio": [
                    value / local_total if local_total > 0 else 0.0 for value in self.hgb_local_selection_histogram
                ],
                "layer_scale_tanh": self.hgb_layer_scale_tanh_sum / hgb_divisor,
                "hidden_norm_mean": self.hgb_hidden_norm_sum / hgb_divisor,
                "local_delta_norm_mean": self.hgb_local_delta_norm_sum / hgb_divisor,
                "cont_delta_norm_mean": self.hgb_cont_delta_norm_sum / hgb_divisor,
                "mixed_norm_mean": self.hgb_mixed_norm_sum / hgb_divisor,
                "saliency_mean": self.hgb_saliency_mean_sum / hgb_divisor,
                "saliency_std": self.hgb_saliency_std_sum / hgb_divisor,
                "local_entropy_mean": self.hgb_local_entropy_mean_sum / hgb_divisor,
                "layout_fallback_count": self.hgb_layout_fallback_count,
                "token_mismatch_count": self.hgb_token_mismatch_count,
                "min_overlap_ratio": self.hgb_min_overlap_ratio if self.hgb_overlap_ratio_count > 0 else 1.0,
                "mean_overlap_ratio": (
                    self.hgb_overlap_ratio_sum / self.hgb_overlap_ratio_count
                    if self.hgb_overlap_ratio_count > 0
                    else 1.0
                ),
                "available_frames_mean": self.hgb_available_frames_sum / hgb_divisor,
            }
        return payload


class RouterStatsCollector:
    def __init__(self, output_dir: str, tag: str):
        self.output_dir = Path(output_dir)
        self.tag = tag
        self.rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
        self.pid = os.getpid()
        self.flush_every = max(int(os.getenv("ZENVIEW_ROUTER_STATS_FLUSH_EVERY", "20")), 1)
        self.records = 0
        self.variant_name: Optional[str] = None
        self.layers: Dict[str, _LayerAccumulator] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.flush)

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"router_stats_{self.tag}_rank{self.rank}_pid{self.pid}.json"

    def record(
        self,
        variant_name: str,
        layer_idx: int,
        selection_histogram: List[float],
        drop_histogram: List[float],
        raw_candidate_norm: List[float],
        projected_candidate_norm: List[float],
        gate_mean: float,
        gate_std: float,
        extra_metrics: Optional[Dict[str, object]] = None,
    ) -> None:
        self.variant_name = variant_name
        layer_key = str(int(layer_idx))
        if layer_key not in self.layers:
            self.layers[layer_key] = _LayerAccumulator()
        self.layers[layer_key].update(
            selection_histogram=selection_histogram,
            drop_histogram=drop_histogram,
            raw_candidate_norm=raw_candidate_norm,
            projected_candidate_norm=projected_candidate_norm,
            gate_mean=gate_mean,
            gate_std=gate_std,
            extra_metrics=extra_metrics,
        )
        self.records += 1
        if self.records % self.flush_every == 0:
            self.flush()

    def _overall(self) -> Dict[str, object]:
        overall = _LayerAccumulator()
        for layer in self.layers.values():
            overall.selection_histogram = [
                current + value for current, value in zip(overall.selection_histogram, layer.selection_histogram)
            ]
            overall.drop_histogram = [
                current + value for current, value in zip(overall.drop_histogram, layer.drop_histogram)
            ]
            overall.raw_candidate_norm_sum = [
                current + value for current, value in zip(overall.raw_candidate_norm_sum, layer.raw_candidate_norm_sum)
            ]
            overall.projected_candidate_norm_sum = [
                current + value
                for current, value in zip(overall.projected_candidate_norm_sum, layer.projected_candidate_norm_sum)
            ]
            overall.gate_mean_sum += layer.gate_mean_sum
            overall.gate_std_sum += layer.gate_std_sum
            overall.count += layer.count
            overall.hgb_gate_histogram = [
                current + value for current, value in zip(overall.hgb_gate_histogram, layer.hgb_gate_histogram)
            ]
            overall.hgb_gate_prob_sum = [
                current + value for current, value in zip(overall.hgb_gate_prob_sum, layer.hgb_gate_prob_sum)
            ]
            overall.hgb_local_selection_histogram = [
                current + value
                for current, value in zip(overall.hgb_local_selection_histogram, layer.hgb_local_selection_histogram)
            ]
            overall.hgb_layer_scale_tanh_sum += layer.hgb_layer_scale_tanh_sum
            overall.hgb_hidden_norm_sum += layer.hgb_hidden_norm_sum
            overall.hgb_local_delta_norm_sum += layer.hgb_local_delta_norm_sum
            overall.hgb_cont_delta_norm_sum += layer.hgb_cont_delta_norm_sum
            overall.hgb_mixed_norm_sum += layer.hgb_mixed_norm_sum
            overall.hgb_saliency_mean_sum += layer.hgb_saliency_mean_sum
            overall.hgb_saliency_std_sum += layer.hgb_saliency_std_sum
            overall.hgb_local_entropy_mean_sum += layer.hgb_local_entropy_mean_sum
            overall.hgb_layout_fallback_count += layer.hgb_layout_fallback_count
            overall.hgb_token_mismatch_count += layer.hgb_token_mismatch_count
            overall.hgb_overlap_ratio_sum += layer.hgb_overlap_ratio_sum
            overall.hgb_overlap_ratio_count += layer.hgb_overlap_ratio_count
            if layer.hgb_overlap_ratio_count > 0:
                overall.hgb_min_overlap_ratio = min(overall.hgb_min_overlap_ratio, layer.hgb_min_overlap_ratio)
            overall.hgb_available_frames_sum += layer.hgb_available_frames_sum
            overall.hgb_count += layer.hgb_count
        return overall.to_dict()

    def flush(self) -> None:
        if not self.layers:
            return
        payload = {
            "tag": self.tag,
            "variant_name": self.variant_name,
            "rank": int(self.rank),
            "pid": self.pid,
            "candidate_names": list(CANDIDATE_NAMES),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "records": self.records,
            "overall": self._overall(),
            "per_layer": {layer_key: self.layers[layer_key].to_dict() for layer_key in sorted(self.layers, key=int)},
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


_GLOBAL_COLLECTOR: Optional[RouterStatsCollector] = None


def get_router_stats_collector() -> Optional[RouterStatsCollector]:
    global _GLOBAL_COLLECTOR

    output_dir = os.getenv("ZENVIEW_ROUTER_STATS_DIR", "").strip()
    if not output_dir:
        return None
    if not _env_flag("ZENVIEW_ROUTER_STATS_ENABLE") and not _env_flag("ZENVIEW_ROUTER_STATS_FORCE"):
        return None

    if _GLOBAL_COLLECTOR is None:
        tag = os.getenv("ZENVIEW_ROUTER_STATS_TAG", "eval")
        _GLOBAL_COLLECTOR = RouterStatsCollector(output_dir=output_dir, tag=tag)
    return _GLOBAL_COLLECTOR

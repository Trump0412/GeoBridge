"""Run Stage1 CBEval for g11-only baselines and checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from qwen_vl.eval.stage1_cbeval.baselines import (
    Stage1CBEvalModel,
    current_frame_only_context,
    current_frame_only_recovery,
    fcp_context,
    fcp_recovery,
    g11_knn_context,
    g11_knn_recovery,
    load_projected_g11,
    move_batch_to_device,
    same_position_context,
    same_position_recovery,
)
from qwen_vl.eval.stage1_cbeval.masks import build_eval_masks, build_frame_permutations
from qwen_vl.eval.stage1_cbeval.metrics import (
    MetricAccumulator,
    compute_cosine_alignment,
    compute_mccr,
    compute_norm_metric,
    compute_tcs,
    enrich_method_metrics,
)
from qwen_vl.eval.stage1_cbeval.score import checkpoint_label_from_path
from qwen_vl.train.stage1_geometry_v2 import Stage1GeometryDatasetV2, stage1_v2_collate_fn


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def build_method_specs(args, device: torch.device) -> Dict[str, Dict]:
    baselines = {}
    if not args.disable_default_baselines:
        baselines = {
            "current_frame_only": {"kind": "baseline"},
            "g11_samepos": {"kind": "baseline"},
            "g11_knn": {"kind": "baseline"},
            "g11_knn_shuffle": {"kind": "baseline"},
        }
    methods = dict(baselines)
    labels = args.checkpoint_labels or []
    if labels and len(labels) != len(args.checkpoint_paths):
        raise ValueError("checkpoint_labels must match checkpoint_paths length")
    for index, checkpoint_path in enumerate(args.checkpoint_paths):
        method_name = labels[index] if labels else checkpoint_label_from_path(checkpoint_path)
        model = Stage1CBEvalModel(
            d_geom=args.d_geom,
            continuity_radius=args.temporal_radius,
            continuity_mlp_hidden_ratio=args.continuity_mlp_hidden_ratio,
            continuity_attention_heads=args.continuity_attention_heads,
            corr_score_beta=args.corr_score_beta,
            time_bias_init=args.time_bias_init,
        )
        model.load_checkpoint(checkpoint_path, device)
        methods[method_name] = {
            "kind": "checkpoint",
            "checkpoint_path": checkpoint_path,
            "model": model,
        }
    return methods


def evaluate_single_method(
    method_name: str,
    method_spec: Dict,
    *,
    z: torch.Tensor,
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    permutations,
    group_ids: List[str],
    args,
) -> Dict[str, float]:
    if method_name == "current_frame_only":
        rep = current_frame_only_context(z)
        rep_shuffled = rep
        prediction_fn = lambda mask, perms=None: current_frame_only_recovery(z, valid_token_mask, mask)
    elif method_name == "g11_samepos":
        rep = same_position_context(z, valid_token_mask, radius=args.temporal_radius)
        rep_shuffled = same_position_context(
            z,
            valid_token_mask,
            radius=args.temporal_radius,
            permutations=permutations,
        )
        prediction_fn = lambda mask, perms=None: same_position_recovery(
            z,
            valid_token_mask,
            mask,
            radius=args.temporal_radius,
            permutations=perms,
        )
    elif method_name == "g11_knn":
        rep = g11_knn_context(
            z,
            valid_token_mask,
            neighbor_indices,
            neighbor_scores,
            topk=args.topk_neighbors,
        )
        rep_shuffled = g11_knn_context(
            z,
            valid_token_mask,
            neighbor_indices,
            neighbor_scores,
            topk=args.topk_neighbors,
            permutations=permutations,
        )
        prediction_fn = lambda mask, perms=None: g11_knn_recovery(
            z,
            valid_token_mask,
            mask,
            neighbor_indices,
            neighbor_scores,
            topk=args.topk_neighbors,
            permutations=perms,
        )
    elif method_name == "g11_knn_shuffle":
        rep = g11_knn_context(
            z,
            valid_token_mask,
            neighbor_indices,
            neighbor_scores,
            topk=args.topk_neighbors,
            permutations=permutations,
        )
        rep_shuffled = rep
        prediction_fn = lambda mask, perms=None: g11_knn_recovery(
            z,
            valid_token_mask,
            mask,
            neighbor_indices,
            neighbor_scores,
            topk=args.topk_neighbors,
            permutations=permutations,
        )
    else:
        model = method_spec["model"]
        rep = fcp_context(
            model,
            z,
            valid_token_mask,
            neighbor_indices,
            neighbor_scores,
            use_continuity_selector=parse_bool(args.use_continuity_selector),
            use_activated_corr_graph=parse_bool(args.use_activated_corr_graph),
        )
        rep_shuffled = fcp_context(
            model,
            z,
            valid_token_mask,
            neighbor_indices,
            neighbor_scores,
            use_continuity_selector=parse_bool(args.use_continuity_selector),
            use_activated_corr_graph=parse_bool(args.use_activated_corr_graph),
            permutations=permutations,
        )
        prediction_fn = lambda mask, perms=None: fcp_recovery(
            model,
            z,
            valid_token_mask,
            mask,
            neighbor_indices,
            neighbor_scores,
            use_continuity_selector=parse_bool(args.use_continuity_selector),
            use_activated_corr_graph=parse_bool(args.use_activated_corr_graph),
            permutations=perms,
        )

    raw_metrics = {}
    shuffled_raw_metrics = {}
    for mask_name, mask_tensor in masks.items():
        prediction = prediction_fn(mask_tensor, None)
        shuffled_prediction = prediction if method_name in {"current_frame_only", "g11_knn_shuffle"} else prediction_fn(mask_tensor, permutations)
        metric = compute_mccr(prediction, z, mask_tensor)
        shuffled_metric = compute_mccr(shuffled_prediction, z, mask_tensor)
        raw_metrics[f"mccr_cos_{mask_name}"] = metric["cos"]
        raw_metrics[f"mccr_l1_{mask_name}"] = metric["l1"]
        shuffled_raw_metrics[f"mccr_cos_{mask_name}"] = shuffled_metric["cos"]
        shuffled_raw_metrics[f"mccr_l1_{mask_name}"] = shuffled_metric["l1"]

    tcs_metrics = compute_tcs(
        rep,
        valid_token_mask,
        neighbor_indices,
        sample_keys=group_ids,
        seed=args.tcs_seed,
        positive_topk=args.positive_topk,
        max_anchors=args.tcs_anchors,
        num_negatives=args.tcs_negatives,
    )
    shuffled_tcs_metrics = tcs_metrics if method_name in {"current_frame_only", "g11_knn_shuffle"} else compute_tcs(
        rep_shuffled,
        valid_token_mask,
        neighbor_indices,
        sample_keys=group_ids,
        seed=args.tcs_seed,
        positive_topk=args.positive_topk,
        max_anchors=args.tcs_anchors,
        num_negatives=args.tcs_negatives,
    )

    raw_metrics.update(tcs_metrics)
    raw_metrics["norm_rep"] = compute_norm_metric(rep, valid_token_mask)
    raw_metrics["cos_rep_g11"] = compute_cosine_alignment(rep, z, valid_token_mask)

    current_frame_only_score = None
    if method_name != "current_frame_only":
        current_rep = current_frame_only_context(z)
        current_mask_metrics = {
            name: compute_mccr(current_frame_only_recovery(z, valid_token_mask, tensor), z, tensor)
            for name, tensor in masks.items()
        }
        current_tcs = compute_tcs(
            current_rep,
            valid_token_mask,
            neighbor_indices,
            sample_keys=group_ids,
            seed=args.tcs_seed,
            positive_topk=args.positive_topk,
            max_anchors=args.tcs_anchors,
            num_negatives=args.tcs_negatives,
        )
        current_frame_only_score = 0.5 * (
            0.5 * (current_mask_metrics["corr_tube"]["cos"] + current_mask_metrics["frame_block"]["cos"])
            + current_tcs["tcs"]
        )

    return enrich_method_metrics(
        raw_metrics,
        shuffled_metrics={**shuffled_raw_metrics, **shuffled_tcs_metrics},
        current_frame_only_score=current_frame_only_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_manifest", type=str, required=True)
    parser.add_argument("--cache_root", type=str, default="")
    parser.add_argument("--checkpoint_paths", nargs="*", default=[])
    parser.add_argument("--checkpoint_labels", nargs="*", default=[])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--disable_default_baselines", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--memory_cache_size", type=int, default=8)
    parser.add_argument("--max_windows", type=int, default=-1)
    parser.add_argument("--masked_ratio", type=float, default=0.20)
    parser.add_argument("--positive_topk", type=int, default=3)
    parser.add_argument("--topk_neighbors", type=int, default=8)
    parser.add_argument("--mask_seed", type=int, default=20260523)
    parser.add_argument("--tcs_seed", type=int, default=20260523)
    parser.add_argument("--shuffle_seed", type=int, default=20260523)
    parser.add_argument("--tcs_anchors", type=int, default=64)
    parser.add_argument("--tcs_negatives", type=int, default=64)
    parser.add_argument("--d_geom", type=int, default=1024)
    parser.add_argument("--temporal_radius", type=int, default=2)
    parser.add_argument("--continuity_mlp_hidden_ratio", type=float, default=2.0)
    parser.add_argument("--continuity_attention_heads", type=int, default=4)
    parser.add_argument("--corr_score_beta", type=float, default=1.0)
    parser.add_argument("--time_bias_init", type=float, default=-0.10)
    parser.add_argument("--use_continuity_selector", type=str, default="True")
    parser.add_argument("--use_activated_corr_graph", type=str, default="True")
    parser.add_argument("--log_every", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Stage1GeometryDatasetV2(
        args.eval_manifest,
        image_processor=None,
        geometry_cache_required=True,
        corr_cache_required=True,
        online_fallback=False,
        max_groups=args.max_windows,
        memory_cache_size=args.memory_cache_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=stage1_v2_collate_fn,
    )

    methods = build_method_specs(args, device)
    accumulators = {
        method_name: MetricAccumulator(
            kind=method_spec["kind"],
            checkpoint_path=method_spec.get("checkpoint_path", ""),
        )
        for method_name, method_spec in methods.items()
    }

    total_windows = len(dataset)
    total_batches = len(loader)
    method_names = list(methods.keys())
    print(
        (
            "[Stage1CBEval] start "
            f"windows={total_windows} "
            f"batches={total_batches} "
            f"methods={method_names} "
            f"device={device} "
            f"batch_size={args.batch_size}"
        ),
        flush=True,
    )
    start_time = time.perf_counter()
    processed_windows = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            group_ids = list(batch["group_id"])
            z = load_projected_g11(batch)
            valid_token_mask = batch["valid_patch_mask"]
            neighbor_indices = batch["corr_graph"]["neighbor_indices"]
            neighbor_scores = batch["corr_graph"]["neighbor_scores"]
            masks = build_eval_masks(
                group_ids,
                valid_token_mask,
                neighbor_indices,
                mask_seed=args.mask_seed,
                masked_ratio=args.masked_ratio,
                positive_topk=args.positive_topk,
            )
            permutations = build_frame_permutations(
                group_ids,
                batch["valid_frame_mask"],
                shuffle_seed=args.shuffle_seed,
            )

            batch_weight = float(len(group_ids))
            processed_windows += int(batch_weight)
            for method_name, method_spec in methods.items():
                method_metrics = evaluate_single_method(
                    method_name,
                    method_spec,
                    z=z,
                    valid_token_mask=valid_token_mask,
                    neighbor_indices=neighbor_indices,
                    neighbor_scores=neighbor_scores,
                    masks=masks,
                    permutations=permutations,
                    group_ids=group_ids,
                    args=args,
                )
                accumulators[method_name].update(method_metrics, batch_weight=batch_weight)

            should_log = args.log_every > 0 and (
                batch_index % args.log_every == 0 or batch_index == total_batches
            )
            if should_log:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                windows_per_sec = processed_windows / elapsed
                remaining_windows = max(total_windows - processed_windows, 0)
                eta_seconds = remaining_windows / max(windows_per_sec, 1e-6)
                print(
                    (
                        "[Stage1CBEval] progress "
                        f"batch={batch_index}/{total_batches} "
                        f"windows={processed_windows}/{total_windows} "
                        f"rate={windows_per_sec:.2f} win/s "
                        f"eta={eta_seconds/60.0:.1f} min"
                    ),
                    flush=True,
                )

    metrics_payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "eval_manifest": args.eval_manifest,
            "cache_root": args.cache_root,
            "mask_seed": args.mask_seed,
            "shuffle_seed": args.shuffle_seed,
            "tcs_seed": args.tcs_seed,
            "masked_ratio": args.masked_ratio,
            "positive_topk": args.positive_topk,
            "topk_neighbors": args.topk_neighbors,
            "temporal_radius": args.temporal_radius,
            "use_continuity_selector": parse_bool(args.use_continuity_selector),
            "use_activated_corr_graph": parse_bool(args.use_activated_corr_graph),
        },
        "methods": {
            method_name: accumulator.finalize()
            for method_name, accumulator in accumulators.items()
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[Stage1CBEval] finished output={output_dir / 'metrics.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

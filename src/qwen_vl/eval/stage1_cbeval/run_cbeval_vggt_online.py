"""Stage1 CBEval with online VGGT inference for raw and projected baselines.

Two new baselines compared to run_cbeval.py:
  vggt_raw_knn   (A): VGGT online -> raw 8192-d, L2-normalised -> cosine KNN
  vggt_proj_knn  (B): VGGT online -> GeoProjector -> 1024-d -> KNN

The corr_graph neighbor_indices/scores from the existing cache are reused for
both baselines so the KNN topology is identical to g11_knn.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qwen_vl.eval.stage1_cbeval.baselines import (
    g11_knn_context,
    g11_knn_recovery,
    load_projected_g11,
    move_batch_to_device,
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

from qwen_vl.data.utils import prepare_image_inputs
from qwen_vl.model.geometry_bank.geo_projector import GeoProjector
from qwen_vl.model.geometry_bank.vggt_bank_extractor import VGGTBankExtractor
from qwen_vl.train.stage1_geometry_v2 import Stage1GeometryDatasetV2, stage1_v2_collate_fn


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def load_geo_projector(checkpoint_path: str, device: torch.device, d_geom: int = 1024) -> GeoProjector:
    projector = GeoProjector(input_dims={"g11_raw": 8192}, d_geom=d_geom)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = payload.get("model", payload)
    proj_state = {
        key.removeprefix("geo_projector."): value
        for key, value in state_dict.items()
        if key.startswith("geo_projector.")
    }
    if not proj_state:
        raise ValueError(
            f"No geo_projector.* keys found in checkpoint {checkpoint_path}. "
            "Cannot run vggt_proj_knn baseline."
        )
    projector.load_state_dict(proj_state, strict=True)
    projector.to(device)
    projector.eval()
    return projector


_PATCH_SIZE = 14


def extract_vggt_raw_batch(
    extractor: VGGTBankExtractor,
    batch: Dict,
    device: torch.device,
    image_processor,
) -> torch.Tensor:
    """Run VGGT online for each sample in the batch, return [B, T, P, 8192] float32.

    Uses prepare_image_inputs (same as training cache build) so the output token
    count P matches the cache's projected g11 token count.
    """
    from PIL import Image

    frame_paths_list: List[List[str]] = batch["frame_paths"]
    valid_frame_mask: torch.Tensor = batch["valid_frame_mask"]
    batch_size = len(frame_paths_list)

    all_raw: List[torch.Tensor] = []
    for b in range(batch_size):
        paths = frame_paths_list[b]
        valid = valid_frame_mask[b]

        imgs_prepared: List[torch.Tensor] = []
        first_tensor: torch.Tensor | None = None
        for i, p in enumerate(paths):
            if valid[i]:
                with Image.open(p) as img:
                    rgb = img.convert("RGB")
                t = prepare_image_inputs(rgb, image_processor)["geometry_encoder_inputs"]
                if first_tensor is None:
                    first_tensor = t
                imgs_prepared.append(t)
            else:
                # placeholder: repeat first valid frame
                if first_tensor is not None:
                    imgs_prepared.append(first_tensor)
                else:
                    # fallback before any valid frame seen
                    imgs_prepared.append(torch.zeros(3, 518, 518))

        imgs_tensor = torch.stack(imgs_prepared).to(device)  # [T, 3, H, W]

        with torch.inference_mode():
            feat_out = extractor.extract(imgs_tensor)

        raw = feat_out.layer_tokens["g11_raw"].float()  # [T, P, 8192]
        all_raw.append(raw)

    # pad to same P if needed (should be identical within a batch)
    max_p = max(r.shape[1] for r in all_raw)
    num_t = all_raw[0].shape[0]
    padded = torch.zeros(batch_size, num_t, max_p, all_raw[0].shape[-1], device=device)
    for b, r in enumerate(all_raw):
        padded[b, :, : r.shape[1], :] = r
    return padded  # [B, T, P, 8192]


def evaluate_vggt_baselines(
    method_name: str,
    raw_features: torch.Tensor,
    geo_projector: GeoProjector | None,
    *,
    z_proj: torch.Tensor,
    valid_token_mask: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    permutations,
    group_ids: List[str],
    args,
) -> Dict[str, float]:
    """Evaluate one VGGT online baseline using the same metric pipeline as run_cbeval."""

    if method_name == "vggt_raw_knn":
        # L2-normalise raw features so cosine similarity is well-defined
        z = F.normalize(raw_features.float(), dim=-1)
    elif method_name == "vggt_proj_knn":
        assert geo_projector is not None
        B, T, P, D = raw_features.shape
        flat = raw_features.reshape(B * T * P, D)
        with torch.inference_mode():
            proj = geo_projector({"g11_raw": flat})["g11"]
        z = proj.reshape(B, T, P, -1).float()
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # context / recovery use the same KNN functions as g11_knn
    rep = g11_knn_context(z, valid_token_mask, neighbor_indices, neighbor_scores, topk=args.topk_neighbors)
    rep_shuffled = g11_knn_context(
        z, valid_token_mask, neighbor_indices, neighbor_scores,
        topk=args.topk_neighbors, permutations=permutations,
    )

    raw_metrics: Dict[str, float] = {}
    shuffled_raw_metrics: Dict[str, float] = {}
    for mask_name, mask_tensor in masks.items():
        pred = g11_knn_recovery(z, valid_token_mask, mask_tensor, neighbor_indices, neighbor_scores, topk=args.topk_neighbors)
        pred_shuffled = g11_knn_recovery(
            z, valid_token_mask, mask_tensor, neighbor_indices, neighbor_scores,
            topk=args.topk_neighbors, permutations=permutations,
        )
        m = compute_mccr(pred, z, mask_tensor)
        ms = compute_mccr(pred_shuffled, z, mask_tensor)
        raw_metrics[f"mccr_cos_{mask_name}"] = m["cos"]
        raw_metrics[f"mccr_l1_{mask_name}"] = m["l1"]
        shuffled_raw_metrics[f"mccr_cos_{mask_name}"] = ms["cos"]
        shuffled_raw_metrics[f"mccr_l1_{mask_name}"] = ms["l1"]

    tcs_metrics = compute_tcs(
        rep, valid_token_mask, neighbor_indices,
        sample_keys=group_ids, seed=args.tcs_seed,
        positive_topk=args.positive_topk,
        max_anchors=args.tcs_anchors, num_negatives=args.tcs_negatives,
    )
    shuffled_tcs_metrics = compute_tcs(
        rep_shuffled, valid_token_mask, neighbor_indices,
        sample_keys=group_ids, seed=args.tcs_seed,
        positive_topk=args.positive_topk,
        max_anchors=args.tcs_anchors, num_negatives=args.tcs_negatives,
    )

    raw_metrics.update(tcs_metrics)
    raw_metrics["norm_rep"] = compute_norm_metric(rep, valid_token_mask)
    # alignment against projected g11 — only meaningful when rep is 1024-d
    if rep.shape[-1] == z_proj.shape[-1]:
        raw_metrics["cos_rep_proj_g11"] = compute_cosine_alignment(rep, z_proj, valid_token_mask)

    # current_frame_only gap
    from qwen_vl.eval.stage1_cbeval.baselines import current_frame_only_context, current_frame_only_recovery
    current_rep = current_frame_only_context(z)
    current_mask_metrics = {
        name: compute_mccr(current_frame_only_recovery(z, valid_token_mask, tensor), z, tensor)
        for name, tensor in masks.items()
    }
    current_tcs = compute_tcs(
        current_rep, valid_token_mask, neighbor_indices,
        sample_keys=group_ids, seed=args.tcs_seed,
        positive_topk=args.positive_topk,
        max_anchors=args.tcs_anchors, num_negatives=args.tcs_negatives,
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
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vggt_model_path", type=str,
                        default="models/VGGT-1B",
                        help="Path to VGGT-1B model for online feature extraction")
    parser.add_argument("--qwen_model_path", type=str,
                        default="models/Qwen2.5-VL-7B-Instruct",
                        help="Path to Qwen2.5-VL model for image_processor (needed for prepare_image_inputs)")
    parser.add_argument("--stage1_checkpoint_path", type=str, default="",
                        help="Stage1 checkpoint to load GeoProjector weights for vggt_proj_knn")
    parser.add_argument("--run_raw_knn", type=str, default="True",
                        help="Run vggt_raw_knn baseline (A)")
    parser.add_argument("--run_proj_knn", type=str, default="True",
                        help="Run vggt_proj_knn baseline (B)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Keep small: each sample runs VGGT online")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--memory_cache_size", type=int, default=8)
    parser.add_argument("--max_windows", type=int, default=-1)
    parser.add_argument("--masked_ratio", type=float, default=0.20)
    parser.add_argument("--positive_topk", type=int, default=3)
    parser.add_argument("--topk_neighbors", type=int, default=8)
    parser.add_argument("--mask_seed", type=int, default=20260524)
    parser.add_argument("--tcs_seed", type=int, default=20260524)
    parser.add_argument("--shuffle_seed", type=int, default=20260524)
    parser.add_argument("--tcs_anchors", type=int, default=64)
    parser.add_argument("--tcs_negatives", type=int, default=64)
    parser.add_argument("--d_geom", type=int, default=1024)
    parser.add_argument("--log_every", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_raw = parse_bool(args.run_raw_knn)
    run_proj = parse_bool(args.run_proj_knn)
    if not run_raw and not run_proj:
        raise ValueError("At least one of --run_raw_knn or --run_proj_knn must be True")
    if run_proj and not args.stage1_checkpoint_path:
        raise ValueError("--stage1_checkpoint_path is required for vggt_proj_knn")

    print(f"[VGGTOnlineEval] loading VGGT from {args.vggt_model_path}", flush=True)
    extractor = VGGTBankExtractor(
        model_path=args.vggt_model_path,
        layer_ids=(11,),
        freeze_encoder=True,
    )
    extractor.to(device)
    extractor.eval()

    print(f"[VGGTOnlineEval] loading image_processor from {args.qwen_model_path}", flush=True)
    from transformers import AutoProcessor as _AutoProcessor
    image_processor = _AutoProcessor.from_pretrained(args.qwen_model_path).image_processor

    geo_projector: GeoProjector | None = None
    if run_proj:
        print(f"[VGGTOnlineEval] loading GeoProjector from {args.stage1_checkpoint_path}", flush=True)
        geo_projector = load_geo_projector(args.stage1_checkpoint_path, device, d_geom=args.d_geom)

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
        persistent_workers=False,
        collate_fn=stage1_v2_collate_fn,
    )

    method_names = []
    if run_raw:
        method_names.append("vggt_raw_knn")
    if run_proj:
        method_names.append("vggt_proj_knn")

    accumulators = {
        name: MetricAccumulator(kind="baseline", checkpoint_path="")
        for name in method_names
    }

    total_windows = len(dataset)
    total_batches = len(loader)
    print(
        f"[VGGTOnlineEval] start windows={total_windows} batches={total_batches} "
        f"methods={method_names} device={device} batch_size={args.batch_size}",
        flush=True,
    )
    start_time = time.perf_counter()
    processed_windows = 0

    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        group_ids = list(batch["group_id"])
        z_proj = load_projected_g11(batch)
        valid_token_mask = batch["valid_patch_mask"]
        neighbor_indices = batch["corr_graph"]["neighbor_indices"]
        neighbor_scores = batch["corr_graph"]["neighbor_scores"]

        masks = build_eval_masks(
            group_ids, valid_token_mask, neighbor_indices,
            mask_seed=args.mask_seed, masked_ratio=args.masked_ratio,
            positive_topk=args.positive_topk,
        )
        permutations = build_frame_permutations(
            group_ids, batch["valid_frame_mask"], shuffle_seed=args.shuffle_seed,
        )

        # run VGGT online once per batch, shared by both methods
        raw_features = extract_vggt_raw_batch(extractor, batch, device, image_processor)

        batch_weight = float(len(group_ids))
        processed_windows += int(batch_weight)

        for method_name in method_names:
            method_metrics = evaluate_vggt_baselines(
                method_name,
                raw_features,
                geo_projector,
                z_proj=z_proj,
                valid_token_mask=valid_token_mask,
                neighbor_indices=neighbor_indices,
                neighbor_scores=neighbor_scores,
                masks=masks,
                permutations=permutations,
                group_ids=group_ids,
                args=args,
            )
            accumulators[method_name].update(method_metrics, batch_weight=batch_weight)

        if args.log_every > 0 and (batch_index % args.log_every == 0 or batch_index == total_batches):
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            rate = processed_windows / elapsed
            eta = max(total_windows - processed_windows, 0) / max(rate, 1e-6)
            print(
                f"[VGGTOnlineEval] batch={batch_index}/{total_batches} "
                f"windows={processed_windows}/{total_windows} "
                f"rate={rate:.2f} win/s eta={eta/60:.1f} min",
                flush=True,
            )

    metrics_payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "eval_manifest": args.eval_manifest,
            "vggt_model_path": args.vggt_model_path,
            "stage1_checkpoint_path": args.stage1_checkpoint_path,
            "mask_seed": args.mask_seed,
            "shuffle_seed": args.shuffle_seed,
            "tcs_seed": args.tcs_seed,
            "masked_ratio": args.masked_ratio,
            "topk_neighbors": args.topk_neighbors,
            "tcs_anchors": args.tcs_anchors,
            "tcs_negatives": args.tcs_negatives,
        },
        "methods": {
            name: acc.finalize()
            for name, acc in accumulators.items()
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[VGGTOnlineEval] metrics written to {metrics_path}", flush=True)

    for name, result in metrics_payload["methods"].items():
        mccr = result.get("mccr_cos_primary", float("nan"))
        shuffle_gap = result.get("shuffle_gap", float("nan"))
        cfo_gap = result.get("current_frame_only_gap", float("nan"))
        print(
            f"[VGGTOnlineEval] {name}: "
            f"mccr_cos_primary={mccr:.4f} "
            f"shuffle_gap={shuffle_gap:.4f} "
            f"cfo_gap={cfo_gap:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

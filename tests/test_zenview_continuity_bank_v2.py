import os
import tempfile
import types
import json

import torch

from qwen_vl.data.geometry_cache import (
    build_sampled_frame_paths,
    extract_required_marker_indices,
    remap_spar_info_image_indices,
)
from qwen_vl.model.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from qwen_vl.model.geometry_bank import BaseGeometryFusion, BankRouter, ContinuityBuilder, GeometryDecoder
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
from qwen_vl.train.stage1_geometry import (
    Stage1GeometryDataset,
    Stage1SourceGroupedBatchSampler,
    stage1_collate_fn,
)


def test_base_fusion_and_decoder_shapes():
    fusion = BaseGeometryFusion(d_geom=16)
    decoder = GeometryDecoder(d_geom=16)
    g11 = torch.randn(3, 5, 16)
    g17 = torch.randn(3, 5, 16)
    g23 = torch.randn(3, 5, 16)
    z = fusion(g11, g17, g23)
    continuity = ContinuityBuilder(d_geom=16).forward_from_fused(z)
    decoded = decoder(continuity)

    assert z.shape == (3, 5, 16)
    assert continuity.shape == (3, 5, 16)
    assert set(decoded.keys()) == {"g11", "g17", "g23"}
    assert decoded["g11"].shape == (3, 5, 16)


def test_continuity_builder_supports_batched_inputs():
    builder = ContinuityBuilder(d_geom=16, radius_t=1, use_spatial_neighbors=False)
    z = torch.randn(2, 3, 5, 16)
    continuity = builder.forward_from_fused(z)
    assert continuity.shape == (2, 3, 5, 16)


def test_router_dropout_falls_back_when_too_many_candidates_drop():
    router = BankRouter(
        hidden_size=8,
        d_geom=8,
        num_layers=20,
        topk=2,
        normalize_query=True,
        normalize_bank=True,
        temperature=0.07,
        candidate_dropout_enabled=True,
        g11_drop_prob=1.0,
        g17_drop_prob=1.0,
        g23_drop_prob=1.0,
    )
    router.train()
    hidden = torch.randn(4, 8)
    bank = torch.randn(4, 4, 8)
    _, info = router(hidden, bank, layer_id=0)

    assert info["topk_idx"].shape == (4, 2)
    assert torch.allclose(info["topk_weight"].sum(dim=-1), torch.ones(4), atol=1e-5)
    # fallback keeps routing valid even when all three local candidates would have dropped
    assert info["drop_histogram"][:3].sum().item() == 0


def test_router_scores_use_normalized_bank_but_values_use_raw_bank():
    router = BankRouter(
        hidden_size=2,
        d_geom=2,
        num_layers=20,
        topk=2,
        normalize_query=True,
        normalize_bank=True,
        temperature=1.0,
        candidate_dropout_enabled=False,
    )
    with torch.no_grad():
        router.input_norm.weight.fill_(1.0)
        router.input_norm.bias.zero_()
        router.query_proj.weight.copy_(torch.eye(2))

    hidden = torch.tensor([[3.0, 4.0]])
    bank = torch.tensor([[[10.0, 0.0], [0.0, 5.0], [1.0, 1.0], [2.0, 2.0]]])
    selected, info = router(hidden, bank, layer_id=0)

    topk_idx = info["topk_idx"][0].tolist()
    gathered = bank[0, topk_idx]
    expected = (gathered * info["topk_weight"][0].unsqueeze(-1)).sum(dim=0)
    assert torch.allclose(selected[0], expected, atol=1e-5)
    assert torch.allclose(info["raw_candidate_norm"], bank.norm(dim=-1).mean(dim=0), atol=1e-5)


def test_new_decoder_layer_variant_can_use_sparse_fusion_layers():
    config = Qwen2_5_VLConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    setattr(config, "geo_cross_attn", True)
    setattr(config, "geo_inject_version", "zenview_continuity_bank_v2")
    setattr(config, "vggt_bank_num_layers", 20)
    setattr(config, "vggt_bank_fusion_layer_indices", "0,2,4,6")
    setattr(config, "vggt_bank_d_geom", 16)
    setattr(config, "vggt_bank_topk", 2)
    setattr(config, "bank_gate_mode", "scalar")
    setattr(config, "geo_importance_gate", False)
    setattr(config, "depart_smi_token", False)
    setattr(config, "_attn_implementation", "sdpa")

    layer_0 = Qwen2_5_VLDecoderLayer(config, 0)
    layer_1 = Qwen2_5_VLDecoderLayer(config, 1)
    layer_6 = Qwen2_5_VLDecoderLayer(config, 6)
    layer_7 = Qwen2_5_VLDecoderLayer(config, 7)

    assert layer_0.cross_flag is True
    assert layer_1.cross_flag is False
    assert layer_6.cross_flag is True
    assert layer_7.cross_flag is False


def test_build_sampled_frame_paths_for_small_image_group(tmp_path):
    images = []
    for index in range(5):
        image_path = tmp_path / f"frame_{index}.jpg"
        image_path.write_bytes(b"")
        images.append(image_path.name)
    sample = {"images": images, "dataset_name": "spar_234k", "spar_info": '{"type":"route_planning"}'}
    entry = build_sampled_frame_paths(sample, str(tmp_path), target_frames=8, min_frames=4)

    assert entry is not None
    assert len(entry.frame_paths) == 5
    assert entry.question_type == "route_planning"


def test_multi_window_cache_entries_include_metadata(tmp_path):
    from qwen_vl.data.geometry_cache import build_geometry_cache_entries

    images = []
    for index in range(40):
        image_path = tmp_path / f"frame_{index}.jpg"
        image_path.write_bytes(b"")
        images.append(image_path.name)
    sample = {"images": images, "dataset_name": "spar_234k", "spar_info": '{"type":"route_planning"}'}
    entries = build_geometry_cache_entries(
        sample,
        str(tmp_path),
        cache_window_mode="multi_window",
        num_windows_per_sample=4,
        target_frames=8,
        min_frames=4,
        stride_min=2,
        stride_max=4,
    )

    assert len(entries) >= 2
    assert entries[0].window_id == "window_0"
    assert all(entry.source_sample_id == entries[0].source_sample_id for entry in entries)
    assert all(len(entry.frame_paths) == 8 for entry in entries)


def test_extract_and_remap_marker_indices_for_cached_window():
    spar_info = json.dumps(
        {
            "type": "distance_infer_center_oo_video",
            "point_img_idx": [[4, 10]],
            "nested": {"bbox_img_idx": [[10, 15]]},
        }
    )

    assert extract_required_marker_indices(spar_info) == [4, 10, 15]

    remapped = remap_spar_info_image_indices(spar_info, [4, 10, 12, 15])
    parsed = json.loads(remapped)
    assert parsed["point_img_idx"] == [[0, 1]]
    assert parsed["nested"]["bbox_img_idx"] == [[1, 3]]


def test_remap_marker_indices_returns_none_when_window_misses_required_frame():
    spar_info = json.dumps({"type": "distance_infer_center_oo_video", "point_img_idx": [[2, 8]]})
    assert remap_spar_info_image_indices(spar_info, [0, 2, 4, 6]) is None


def test_stage1_collate_fn_pads_cached_features():
    sample_a = {
        "group_id": "a0",
        "source_dataset": "spar_234k",
        "source_sample_id": "source_a",
        "frame_paths": ["a"],
        "valid_frame_mask": torch.tensor([True, True], dtype=torch.bool),
        "question_type": "route_planning",
        "cached_features": {
            "g11_raw": torch.randn(2, 3, 4),
            "g17_raw": torch.randn(2, 3, 4),
            "g23_raw": torch.randn(2, 3, 4),
            "token_counts": [3, 3],
            "frame_shapes": [(1, 3), (1, 3)],
        },
        "geometry_inputs": None,
        "window_id": "window_0",
        "cache_window_mode": "multi_window",
    }
    sample_b = {
        "group_id": "b0",
        "source_dataset": "spar_234k",
        "source_sample_id": "source_b",
        "frame_paths": ["b"],
        "valid_frame_mask": torch.tensor([True], dtype=torch.bool),
        "question_type": "object_counting",
        "cached_features": {
            "g11_raw": torch.randn(1, 2, 4),
            "g17_raw": torch.randn(1, 2, 4),
            "g23_raw": torch.randn(1, 2, 4),
            "token_counts": [2],
            "frame_shapes": [(1, 2)],
        },
        "geometry_inputs": None,
        "window_id": "window_0",
        "cache_window_mode": "multi_window",
    }

    batch = stage1_collate_fn([sample_a, sample_b])
    assert batch["cached_features"]["g11_raw"].shape == (2, 2, 3, 4)
    assert batch["valid_frame_mask"].shape == (2, 2)
    assert batch["valid_patch_mask"].shape == (2, 2, 3)
    assert batch["valid_patch_mask"][1, 0, :2].all()
    assert not batch["valid_patch_mask"][1, 0, 2]


def test_source_grouped_batch_sampler_keeps_same_source_windows_together(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    records = [
        {"group_id": "a0", "source_dataset": "spar_234k", "source_sample_id": "source_a", "frame_paths": [], "valid_frame_mask": [True], "cache_path": "", "question_type": "unknown", "window_id": "window_0", "cache_window_mode": "multi_window"},
        {"group_id": "a1", "source_dataset": "spar_234k", "source_sample_id": "source_a", "frame_paths": [], "valid_frame_mask": [True], "cache_path": "", "question_type": "unknown", "window_id": "window_1", "cache_window_mode": "multi_window"},
        {"group_id": "a2", "source_dataset": "spar_234k", "source_sample_id": "source_a", "frame_paths": [], "valid_frame_mask": [True], "cache_path": "", "question_type": "unknown", "window_id": "window_2", "cache_window_mode": "multi_window"},
        {"group_id": "a3", "source_dataset": "spar_234k", "source_sample_id": "source_a", "frame_paths": [], "valid_frame_mask": [True], "cache_path": "", "question_type": "unknown", "window_id": "window_3", "cache_window_mode": "multi_window"},
        {"group_id": "b0", "source_dataset": "spar_234k", "source_sample_id": "source_b", "frame_paths": [], "valid_frame_mask": [True], "cache_path": "", "question_type": "unknown", "window_id": "window_0", "cache_window_mode": "multi_window"},
        {"group_id": "c0", "source_dataset": "spar_234k", "source_sample_id": "source_c", "frame_paths": [], "valid_frame_mask": [True], "cache_path": "", "question_type": "unknown", "window_id": "window_0", "cache_window_mode": "multi_window"},
    ]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    dataset = Stage1GeometryDataset(str(manifest_path), image_processor=None)
    sampler = Stage1SourceGroupedBatchSampler(dataset, batch_size=4, shuffle=False)
    batches = list(iter(sampler))

    assert batches[0] == [0, 1, 2, 3]
    assert batches[1] == [4, 5]

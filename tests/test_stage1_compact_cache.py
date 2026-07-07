import torch

from qwen_vl.train.stage1_compact_cache import (
    PROJECTED_SOURCE_CACHE_FORMAT,
    WINDOW_READY_SOURCE_JOINT_PACK_FORMAT,
    extract_window_from_source_joint_pack,
    materialize_projected_source_cache,
    quantize_projected_tokenwise,
    slice_projected_source_cache,
)


def test_projected_source_cache_slice_keeps_quantized_payload():
    feature = torch.randn(4, 7, 16, dtype=torch.float32)
    quantized, scales = quantize_projected_tokenwise(feature)
    payload = {
        "cache_format": PROJECTED_SOURCE_CACHE_FORMAT,
        "source_sample_id": "source_a",
        "source_frame_indices": [0, 3, 5, 7],
        "frame_shapes": [(2, 4), (2, 4), (2, 4), (2, 4)],
        "token_counts": [7, 7, 7, 7],
        "patch_grid": (4, 4),
        "merged_grid": (2, 2),
        "g11_q": quantized,
        "g11_scale": scales,
        "g17_q": quantized.clone(),
        "g17_scale": scales.clone(),
        "g23_q": quantized.clone(),
        "g23_scale": scales.clone(),
    }

    sliced = slice_projected_source_cache(payload, [3, 7])
    assert sliced["feature_space"] == "projected_quantized"
    assert sliced["g11_q"].shape == (2, 7, 16)
    assert sliced["token_counts"] == [7, 7]


def test_projected_source_cache_materialize_roundtrip():
    feature = torch.randn(4, 7, 16, dtype=torch.float32)
    quantized, scales = quantize_projected_tokenwise(feature)
    payload = {
        "cache_format": PROJECTED_SOURCE_CACHE_FORMAT,
        "source_sample_id": "source_a",
        "source_frame_indices": [0, 3, 5, 7],
        "frame_shapes": [(2, 4), (2, 4), (2, 4), (2, 4)],
        "token_counts": [7, 7, 7, 7],
        "patch_grid": (4, 4),
        "merged_grid": (2, 2),
        "g11_q": quantized,
        "g11_scale": scales,
        "g17_q": quantized.clone(),
        "g17_scale": scales.clone(),
        "g23_q": quantized.clone(),
        "g23_scale": scales.clone(),
    }

    materialized = materialize_projected_source_cache(payload, [3, 7], output_dtype=torch.float32)
    assert materialized["feature_space"] == "projected"
    assert materialized["g11"].shape == (2, 7, 16)
    reference = feature.index_select(0, torch.tensor([1, 3]))
    max_error = (materialized["g11"] - reference).abs().max().item()
    assert max_error < 0.1


def test_projected_source_cache_materialize_g11_only():
    feature = torch.randn(3, 5, 8, dtype=torch.float32)
    quantized, scales = quantize_projected_tokenwise(feature)
    payload = {
        "cache_format": PROJECTED_SOURCE_CACHE_FORMAT,
        "source_sample_id": "source_g11",
        "source_frame_indices": [2, 4, 6],
        "layer_names": ["g11"],
        "frame_shapes": [(1, 5), (1, 5), (1, 5)],
        "token_counts": [5, 5, 5],
        "patch_grid": (2, 5),
        "merged_grid": (1, 5),
        "g11_q": quantized,
        "g11_scale": scales,
    }

    materialized = materialize_projected_source_cache(payload, [4, 6], output_dtype=torch.float32)
    assert materialized["feature_space"] == "projected"
    assert materialized["layer_names"] == ["g11"]
    assert materialized["g11"].shape == (2, 5, 8)
    reference = feature.index_select(0, torch.tensor([1, 2]))
    max_error = (materialized["g11"] - reference).abs().max().item()
    assert max_error < 0.1


def test_extract_window_from_source_joint_pack():
    payload = {
        "pack_format": WINDOW_READY_SOURCE_JOINT_PACK_FORMAT,
        "source_sample_id": "source_joint",
        "windows": {
            "window_a": {
                "cached_features": {"feature_space": "projected", "layer_names": ["g11"], "g11": torch.ones(2, 3, 4)},
                "corr_graph": {"neighbor_indices": torch.zeros(2, 3, 1, 2, dtype=torch.long), "neighbor_scores": torch.zeros(2, 3, 1)},
            }
        },
    }

    extracted = extract_window_from_source_joint_pack(payload, "window_a")
    assert extracted["cached_features"]["feature_space"] == "projected"
    assert extracted["corr_graph"]["neighbor_indices"].shape == (2, 3, 1, 2)

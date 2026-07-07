from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from stage1_vggt_similarity_common import (  # noqa: E402
    cosine_similarity_map,
    ensure_unit_interval,
    normalize_similarity_map,
    save_heatmap_only,
    save_overlay,
    topk_patch_entries,
)


def test_similarity_render_pipeline(tmp_path: Path) -> None:
    features = torch.randn(3, 4, 5, 16)
    sim_map = cosine_similarity_map(features[0], features[2], (1, 3))
    assert sim_map.shape == (4, 5)

    sim_vis, norm_info = normalize_similarity_map(sim_map, mode="percentile", p_low=2, p_high=98)
    sim_vis, clipped = ensure_unit_interval(sim_vis)
    assert not clipped
    assert 0.0 <= float(sim_vis.min()) <= 1.0
    assert 0.0 <= float(sim_vis.max()) <= 1.0
    assert norm_info["mode"] == "percentile"

    rgb = Image.fromarray(np.random.randint(0, 255, size=(48, 64, 3), dtype=np.uint8), mode="RGB")
    topk = topk_patch_entries(sim_map, topk=3)
    assert len(topk) == 3
    assert topk[0]["score"] >= topk[-1]["score"]

    heatmap_path = tmp_path / "heatmap.png"
    overlay_path = tmp_path / "overlay.png"
    save_heatmap_only(sim_vis, heatmap_path, output_size=rgb.size, upsample="bicubic")
    save_overlay(
        rgb,
        sim_vis,
        overlay_path,
        alpha=0.55,
        upsample="bicubic",
        topk_entries=topk,
        grid_shape=sim_map.shape,
    )

    assert heatmap_path.exists()
    assert overlay_path.exists()
    assert Image.open(heatmap_path).size == rgb.size
    assert Image.open(overlay_path).size == rgb.size

    topk_json_path = tmp_path / "topk.json"
    topk_json_path.write_text(json.dumps({"g11": topk}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert topk_json_path.exists()
    payload = json.loads(topk_json_path.read_text(encoding="utf-8"))
    assert payload["g11"][0]["rank"] == 1


def test_softmax_normalization_stays_in_unit_interval() -> None:
    sim_map = np.array([[0.1, 0.3], [0.7, 1.0]], dtype=np.float32)
    sim_vis, _ = normalize_similarity_map(sim_map, mode="softmax", temperature=0.05)
    assert sim_vis.shape == sim_map.shape
    assert float(sim_vis.min()) >= 0.0
    assert float(sim_vis.max()) <= 1.0

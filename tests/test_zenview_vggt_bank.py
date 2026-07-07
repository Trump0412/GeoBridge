import sys
import types

import pytest
import torch

from qwen_vl.model.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from qwen_vl.model.geometry_bank import (
    BankFusionBlock,
    BankRouter,
    ContinuityBuilder,
    GeoProjector,
    GeometryBank,
    VGGTBankExtractor,
)
from qwen_vl.model.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLForConditionalGenerationWithVGGT,
)


class _DummyVGGT(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def aggregator(self, images):
        _, num_frames, _, _, _ = images.shape
        patch_start_idx = 5
        outputs = []
        for layer_idx in range(24):
            value = float(layer_idx + 1)
            tokens = torch.full((1, num_frames, patch_start_idx + 16, 8), value)
            outputs.append(tokens)
        return outputs, patch_start_idx


def test_vggt_bank_extractor_reads_target_layers(monkeypatch):
    fake_module = types.ModuleType("qwen_vl.model.vggt.models.vggt")
    fake_module.VGGT = _DummyVGGT
    monkeypatch.setitem(sys.modules, "qwen_vl.model.vggt.models.vggt", fake_module)
    extractor = VGGTBankExtractor(
        layer_ids=(11, 17, 23),
        spatial_merge_size=2,
        freeze_encoder=True,
    )
    images = torch.randn(2, 3, 56, 56)
    features = extractor.extract(images)

    assert set(features.layer_tokens.keys()) == {"g11_raw", "g17_raw", "g23_raw"}
    assert features.layer_tokens["g11_raw"].shape == (2, 4, 32)
    assert features.layer_tokens["g17_raw"].shape == (2, 4, 32)
    assert features.layer_tokens["g23_raw"].shape == (2, 4, 32)
    assert features.layer_tokens["g11_raw"].mean().item() == pytest.approx(11.0)
    assert features.layer_tokens["g17_raw"].mean().item() == pytest.approx(17.0)
    assert features.layer_tokens["g23_raw"].mean().item() == pytest.approx(23.0)


def test_geometry_bank_modules_shape_and_topk():
    raw = {
        "g11_raw": torch.randn(3, 5, 32),
        "g17_raw": torch.randn(3, 5, 32),
        "g23_raw": torch.randn(3, 5, 32),
    }
    projector = GeoProjector({"g11_raw": 32, "g17_raw": 32, "g23_raw": 32}, d_geom=16)
    projected = projector(raw)
    continuity = ContinuityBuilder(d_geom=16, radius_t=1)(projected["g11"], projected["g17"], projected["g23"])
    frame_layout = types.SimpleNamespace(token_counts=[5, 5, 5], frame_shapes=[(1, 5), (1, 5), (1, 5)])
    bank = GeometryBank(use_continuity=True)(
        projected["g11"],
        projected["g17"],
        projected["g23"],
        continuity,
        frame_layout,
    )

    router = BankRouter(hidden_size=20, d_geom=16, num_layers=20, topk=2)
    hidden = torch.randn(5, 20)
    selected, router_info = router(hidden, bank.get_frame_bank(0), layer_id=0)
    fusion = BankFusionBlock(hidden_size=20, d_geom=16, gate_mode="scalar")
    delta, gate = fusion(hidden, selected)

    assert continuity.shape == (3, 5, 16)
    assert bank.bank.shape == (3, 5, 4, 16)
    assert selected.shape == (5, 16)
    assert delta.shape == hidden.shape
    assert gate.shape == (5, 1)
    assert router_info["topk_idx"].min().item() >= 0
    assert router_info["topk_idx"].max().item() <= 3
    assert torch.allclose(
        router_info["topk_weight"].sum(dim=-1),
        torch.ones(router_info["topk_weight"].shape[0]),
        atol=1e-5,
    )
    assert not torch.isnan(delta).any()


def test_geometry_bank_modules_accept_bfloat16_inputs():
    raw = {
        "g11_raw": torch.randn(2, 4, 32, dtype=torch.bfloat16),
        "g17_raw": torch.randn(2, 4, 32, dtype=torch.bfloat16),
        "g23_raw": torch.randn(2, 4, 32, dtype=torch.bfloat16),
    }
    projector = GeoProjector({"g11_raw": 32, "g17_raw": 32, "g23_raw": 32}, d_geom=16)
    projected = projector(raw)
    continuity = ContinuityBuilder(d_geom=16, radius_t=1)(projected["g11"], projected["g17"], projected["g23"])
    frame_layout = types.SimpleNamespace(token_counts=[4, 4], frame_shapes=[(1, 4), (1, 4)])
    bank = GeometryBank(use_continuity=True)(
        projected["g11"],
        projected["g17"],
        projected["g23"],
        continuity,
        frame_layout,
    )

    router = BankRouter(hidden_size=20, d_geom=16, num_layers=20, topk=2)
    hidden = torch.randn(4, 20, dtype=torch.bfloat16)
    selected, _ = router(hidden, bank.get_frame_bank(0), layer_id=0)
    fusion = BankFusionBlock(hidden_size=20, d_geom=16, gate_mode="scalar")
    delta, gate = fusion(hidden, selected)

    assert projected["g11"].dtype == torch.float32
    assert continuity.dtype == torch.float32
    assert selected.dtype == torch.float32
    assert delta.dtype == hidden.dtype
    assert gate.dtype == hidden.dtype
    assert not torch.isnan(delta.float()).any()


def test_decoder_layer_only_enables_first_twenty_bank_layers():
    config = Qwen2_5_VLConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    setattr(config, "geo_cross_attn", True)
    setattr(config, "geo_inject_version", "zenview_vggt_bank")
    setattr(config, "vggt_bank_num_layers", 20)
    setattr(config, "vggt_bank_d_geom", 16)
    setattr(config, "vggt_bank_topk", 2)
    setattr(config, "bank_gate_mode", "scalar")
    setattr(config, "geo_importance_gate", False)
    setattr(config, "depart_smi_token", False)
    setattr(config, "_attn_implementation", "sdpa")

    layer_19 = Qwen2_5_VLDecoderLayer(config, 19)
    layer_20 = Qwen2_5_VLDecoderLayer(config, 20)

    assert layer_19.cross_flag is True
    assert layer_20.cross_flag is False


def test_depart_smi_alignment_resizes_image_embeds_to_bank_grid():
    model = object.__new__(Qwen2_5_VLForConditionalGenerationWithVGGT)
    model.config = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))

    dim = 6
    num_images = 2
    source_h = 4
    source_w = 4
    target_grid = (2, 2)
    image_embeds = torch.arange(num_images * source_h * source_w * dim, dtype=torch.float32).reshape(-1, dim)
    image_grid_thw = torch.tensor([[1, 8, 8], [1, 8, 8]], dtype=torch.long)

    aligned = model._align_depart_smi_image_embeds(
        image_embeds=image_embeds,
        num_images=num_images,
        target_grid=target_grid,
        image_grid_thw=image_grid_thw,
    )

    expected = image_embeds.view(num_images, source_h, source_w, dim).permute(0, 3, 1, 2)
    expected = torch.nn.functional.interpolate(expected, size=target_grid, mode="bilinear", align_corners=False)
    expected = expected.permute(0, 2, 3, 1).reshape(-1, dim)

    assert aligned.shape == (num_images * target_grid[0] * target_grid[1], dim)
    assert torch.allclose(aligned, expected)

from pathlib import Path
import sys
import warnings
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_vl.model.geometry_bank.geometry_bank import GeometryBankOutput
from qwen_vl.model.msgf_utils import FrameLayout
from qwen_vl.model.qwen_interaction import QwenGeoBridgeHGB
from qwen_vl.model.stage2 import CompetitiveBridgeGate, LocalLevelRouter
from qwen_vl.train.argument import DataArguments, ModelArguments, build_hgb_effective_config


def _make_config(**overrides):
    config = {
        "hidden_size": 16,
        "vggt_bank_d_geom": 8,
        "vggt_bank_topk": 2,
        "vggt_bank_num_layers": 8,
        "vggt_bank_fusion_layer_indices": "0,1,3,6",
        "vggt_bank_use_layer_embedding": False,
        "geo_inject_version": "geobridge_hgb",
        "bank_debug": False,
        "normalize_query": True,
        "normalize_bank": True,
        "bank_temperature": 0.07,
        "candidate_dropout_enabled": False,
        "g11_drop_prob": 0.0,
        "g17_drop_prob": 0.0,
        "g23_drop_prob": 0.0,
        "bank_gate_mode": "scalar",
        "hgb_use_saliency_prior": True,
        "hgb_local_topk": 2,
        "hgb_corr_topk_neighbors": 8,
        "hgb_temporal_radius": 2,
        "hgb_layer_scale_init": 0.05,
        "hgb_gate_none_bias": 0.0,
        "hgb_gate_local_bias": 0.4,
        "hgb_gate_cont_bias": 0.6,
        "hgb_use_gate_bias_init": True,
        "hgb_layer0_g11_logit_bias": 2.0,
        "hgb_strict_alignment": True,
        "hgb_allow_layout_fallback": False,
        "hgb_alignment_audit_only": False,
        "hgb_min_overlap_ratio": 1.0,
        "vision_config": SimpleNamespace(spatial_merge_size=1),
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _make_geometry_bank(token_count=5, bank_token_count=None, d_geom=8, identical_local=False):
    torch.manual_seed(0)
    bank_token_count = token_count if bank_token_count is None else bank_token_count
    if identical_local:
        g11 = torch.ones(1, bank_token_count, d_geom)
        g17 = torch.ones(1, bank_token_count, d_geom)
        g23 = torch.ones(1, bank_token_count, d_geom)
    else:
        g11 = torch.randn(1, bank_token_count, d_geom)
        g17 = torch.randn(1, bank_token_count, d_geom)
        g23 = torch.randn(1, bank_token_count, d_geom)
    cont = torch.randn(1, bank_token_count, d_geom)
    bank = torch.stack([g11, g17, g23, cont], dim=2)
    stats = {
        "g11_norm": float(g11.norm(dim=-1).mean().item()),
        "g17_norm": float(g17.norm(dim=-1).mean().item()),
        "g23_norm": float(g23.norm(dim=-1).mean().item()),
        "cont_norm": float(cont.norm(dim=-1).mean().item()),
    }
    return GeometryBankOutput(
        bank=bank,
        frame_layout=FrameLayout(token_counts=[token_count], frame_shapes=[(token_count, 1)]),
        valid_mask=torch.ones(1, bank_token_count, dtype=torch.bool),
        stats=stats,
        use_continuity=True,
        saliency=torch.linspace(0.1, 0.9, steps=bank_token_count).unsqueeze(0),
    )


def _make_hidden(token_count=5, hidden_size=16, requires_grad=False):
    torch.manual_seed(1)
    return torch.randn(1, token_count, hidden_size, requires_grad=requires_grad)


def _make_hgb(layer_idx, **config_overrides):
    config = _make_config(**config_overrides)
    return QwenGeoBridgeHGB(config, layer_idx=layer_idx)


def test_local_level_router_shapes():
    router = LocalLevelRouter(hidden_size=16, d_geom=8, topk=2)
    hidden = torch.randn(5, 16)
    bank = torch.randn(5, 3, 8)
    selected, info = router(hidden, bank)
    assert selected.shape == (5, 8)
    assert info["selection_histogram"].shape[0] == 3
    assert info["router_probs"].shape == (5, 3)


def test_competitive_gate_nonzero_warm_start():
    torch.manual_seed(2)
    gate = CompetitiveBridgeGate(hidden_size=16, d_geom=8, use_saliency_prior=True)
    hidden = torch.randn(5, 16)
    local_mem = torch.randn(5, 8)
    cont_mem = torch.randn(5, 8)
    local_delta = torch.randn(5, 16)
    cont_delta = torch.randn(5, 16)
    saliency = torch.rand(5)
    output, info = gate(hidden, local_mem, cont_mem, local_delta, cont_delta, saliency_prior=saliency)
    assert output.shape == (5, 16)
    assert not torch.allclose(output, torch.zeros_like(output))
    probs = info["gate_probs"]
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    final_bias = gate.mlp[-1].bias.detach()
    assert torch.allclose(final_bias, torch.tensor([0.0, 0.4, 0.6], dtype=final_bias.dtype))


def test_competitive_gate_backward_gradients():
    hidden = _make_hidden(requires_grad=True)
    geometry_bank = _make_geometry_bank()
    module = _make_hgb(layer_idx=0)
    output = module(hidden, geometry_bank)
    loss = output.sum()
    loss.backward()

    assert module.bridge_gate.layer_scale.grad is not None
    assert module.bridge_gate.mlp[0].weight.grad is not None
    assert module.local_router.query_proj.weight.grad is not None
    assert module.local_fusion.output_proj.weight.grad is not None
    assert module.cont_fusion.output_proj.weight.grad is not None


def test_hgb_strict_alignment_success():
    hidden = _make_hidden()
    geometry_bank = _make_geometry_bank(token_count=5, bank_token_count=5)
    module = _make_hgb(layer_idx=1)
    output = module(hidden, geometry_bank)

    assert output.shape == hidden.shape
    assert module.last_hgb_stats["active"] is True
    assert module.last_hgb_stats["token_mismatch_count"] == 0
    assert module.last_hgb_stats["layout_fallback_count"] == 0
    assert module.last_hgb_stats["min_overlap_ratio"] == 1.0


def test_hgb_strict_alignment_mismatch_raises():
    hidden = _make_hidden(token_count=5)
    geometry_bank = _make_geometry_bank(token_count=5, bank_token_count=4)
    module = _make_hgb(layer_idx=1, hgb_strict_alignment=True, hgb_alignment_audit_only=False)

    try:
        module(hidden, geometry_bank)
        assert False, "expected RuntimeError on strict alignment mismatch"
    except RuntimeError as exc:
        assert "token alignment mismatch" in str(exc)


def test_hgb_audit_mode_records_mismatch():
    hidden = _make_hidden(token_count=5)
    geometry_bank = _make_geometry_bank(token_count=5, bank_token_count=4)
    module = _make_hgb(layer_idx=3, hgb_strict_alignment=True, hgb_alignment_audit_only=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = module(hidden, geometry_bank)

    assert output.shape == hidden.shape
    assert module.last_hgb_stats["token_mismatch_count"] == 1
    assert module.last_hgb_stats["min_overlap_ratio"] < 1.0
    assert torch.allclose(output[:, -1, :], torch.zeros_like(output[:, -1, :]))


def test_hgb_sparse4_only_active_layers():
    hidden = _make_hidden()
    geometry_bank = _make_geometry_bank()
    active_layers = {0, 1, 3, 6}

    for layer_idx in range(8):
        module = _make_hgb(layer_idx=layer_idx)
        output = module(hidden, geometry_bank)
        if layer_idx in active_layers:
            assert module.last_hgb_stats["active"] is True
            assert output.shape == hidden.shape
        else:
            assert module.last_hgb_stats["active"] is False
            assert torch.allclose(output, torch.zeros_like(output))


def test_hgb_layer0_uses_gate_with_g11_bias():
    hidden = _make_hidden()
    geometry_bank = _make_geometry_bank(identical_local=True)
    module = _make_hgb(layer_idx=0, hgb_layer0_g11_logit_bias=3.0, hgb_local_topk=1)
    output = module(hidden, geometry_bank)

    assert output.shape == hidden.shape
    assert module.last_hgb_stats["active"] is True
    assert module.last_hgb_stats["used_layer0_g11_bias"] is True
    assert sum(module.last_hgb_stats["gate_histogram"]) == hidden.shape[1]
    assert module.last_hgb_stats["local_selection_ratio"][0] > module.last_hgb_stats["local_selection_ratio"][1]


def test_hgb_sparse4_ablation_schedule_compatibility():
    hidden = _make_hidden()
    geometry_bank = _make_geometry_bank()
    module = _make_hgb(layer_idx=2, vggt_bank_fusion_layer_indices="0,2,4,6")
    output = module(hidden, geometry_bank)

    assert module.last_hgb_stats["active"] is True
    assert output.shape == hidden.shape


def test_hgb_effective_config_has_no_dead_knobs():
    model_args = ModelArguments(
        geo_inject_version="geobridge_hgb",
        stage1_checkpoint_path="/tmp/stage1.pt",
        hgb_local_topk=3,
    )
    data_args = DataArguments(
        geometry_cache_use=True,
        geometry_cache_required=True,
        geometry_cache_dir="/tmp/cache",
        geometry_cache_manifest="/tmp/cache/manifest.jsonl",
    )
    config = build_hgb_effective_config(model_args, data_args)

    assert config["hgb_local_topk"] == 3
    assert config["geometry_cache_required"] is True
    assert "vggt_bank_topk" not in config
    assert "candidate_dropout_enabled" not in config
    assert "g11_drop_prob" not in config
    assert "g17_drop_prob" not in config
    assert "g23_drop_prob" not in config
    assert "continuity_drop_prob" not in config

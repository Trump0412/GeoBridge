import torch

from qwen_vl.model.geometry_bank.correspondence_losses import reconstruction_metrics
from qwen_vl.model.geometry_bank.geometry_decoder import GeometryDecoder
from qwen_vl.model.geometry_bank import ActivatedCorrespondenceGraph, ContinuityUtilitySelector, continuity_utility_loss
from qwen_vl.train.train_stage1_continuity_v2 import build_corr_tube_mask


def _toy_neighbors():
    neighbor_indices = torch.full((1, 2, 3, 2, 2), -1, dtype=torch.long)
    neighbor_scores = torch.zeros((1, 2, 3, 2), dtype=torch.float32)
    # frame 0, patch 0 <-> frame 1, patch 1
    neighbor_indices[0, 0, 0, 0] = torch.tensor([1, 1])
    neighbor_indices[0, 1, 1, 0] = torch.tensor([0, 0])
    neighbor_scores[0, 0, 0, 0] = 0.9
    neighbor_scores[0, 1, 1, 0] = 0.85
    # frame 0, patch 2 -> frame 1, patch 2
    neighbor_indices[0, 0, 2, 0] = torch.tensor([1, 2])
    neighbor_indices[0, 1, 2, 0] = torch.tensor([0, 2])
    neighbor_scores[0, 0, 2, 0] = 0.4
    neighbor_scores[0, 1, 2, 0] = 0.35
    return neighbor_indices, neighbor_scores


def test_continuity_selector_outputs_valid_probs_and_loss():
    selector = ContinuityUtilitySelector(d_geom=8)
    z = torch.randn(1, 2, 3, 8)
    valid = torch.tensor([[[True, True, True], [True, True, True]]])
    neighbor_indices, neighbor_scores = _toy_neighbors()

    output = selector(z, neighbor_indices, neighbor_scores, valid)
    assert output["probs"].shape == (1, 2, 3)
    assert output["stats"].shape == (1, 2, 3, 4)
    assert torch.isfinite(output["probs"]).all()

    utility_target = torch.full((1, 2, 3), 0.6)
    loss, aux = continuity_utility_loss(output["logits"], utility_target, valid, budget_ratio=0.2, budget_weight=0.1)
    assert loss.item() >= 0.0
    assert aux["mean_budget"].item() >= 0.0


def test_activated_corr_graph_masks_invalid_edges():
    graph = ActivatedCorrespondenceGraph(hidden_dim=16)
    neighbor_indices, neighbor_scores = _toy_neighbors()
    saliency = torch.tensor([[[0.9, 0.1, 0.7], [0.2, 0.8, 0.4]]], dtype=torch.float32)
    output = graph(neighbor_indices, neighbor_scores, saliency)

    assert output["activation"].shape == neighbor_scores.shape
    assert output["logits"].shape == neighbor_scores.shape
    invalid_mask = neighbor_indices[..., 0] < 0
    assert torch.allclose(output["activation"][invalid_mask], torch.zeros_like(output["activation"][invalid_mask]))


def test_corr_tube_mask_prefers_high_saliency_anchor():
    valid = torch.tensor([[[True, False, True], [False, True, True]]])
    neighbor_indices, _ = _toy_neighbors()
    saliency = torch.tensor([[[0.95, 0.0, 0.1], [0.0, 0.2, 0.1]]], dtype=torch.float32)

    mask = build_corr_tube_mask(
        valid,
        neighbor_indices,
        masked_ratio=0.3,
        positive_topk=1,
        saliency_probs=saliency,
    )

    # highest-saliency anchor is frame0/patch0, so its corr neighbor frame1/patch1 should also be masked
    assert mask[0, 0, 0]
    assert mask[0, 1, 1]


def test_g11_only_decoder_and_metrics():
    decoder = GeometryDecoder(d_geom=8, layer_names=("g11",))
    continuity = torch.randn(1, 2, 3, 8)
    predictions = decoder(continuity)

    assert tuple(predictions.keys()) == ("g11",)
    valid_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    targets = {"g11": torch.randn(1, 2, 3, 8)}
    metrics = reconstruction_metrics(predictions, targets, valid_mask, layer_names=("g11",))

    assert "g11_l1" in metrics
    assert "g11_cos" in metrics
    assert torch.isfinite(metrics["total"])

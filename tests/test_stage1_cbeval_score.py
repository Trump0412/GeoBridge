from qwen_vl.eval.stage1_cbeval.score import score_metrics_payload


def test_score_metrics_payload_ranks_checkpoint_and_marks_guards():
    metrics_payload = {
        "methods": {
            "current_frame_only": {
                "kind": "baseline",
                "checkpoint_path": "",
                "num_windows": 10,
                "metrics": {
                    "mccr_cos_primary": 0.30,
                    "tcs": 0.35,
                    "shuffle_gap": 0.00,
                    "cfo_gap": 0.00,
                    "cos_rep_g11": 0.75,
                },
            },
            "g11_knn": {
                "kind": "baseline",
                "checkpoint_path": "",
                "num_windows": 10,
                "metrics": {
                    "mccr_cos_primary": 0.40,
                    "tcs": 0.45,
                    "shuffle_gap": 0.02,
                    "cfo_gap": 0.10,
                    "cos_rep_g11": 0.82,
                },
            },
            "FCP-c6000": {
                "kind": "checkpoint",
                "checkpoint_path": "/tmp/checkpoint-6000.pt",
                "num_windows": 10,
                "metrics": {
                    "mccr_cos_primary": 0.48,
                    "tcs": 0.55,
                    "shuffle_gap": 0.10,
                    "cfo_gap": 0.20,
                    "cos_rep_g11": 0.80,
                },
            },
            "FCP-c7000": {
                "kind": "checkpoint",
                "checkpoint_path": "/tmp/checkpoint-7000.pt",
                "num_windows": 10,
                "metrics": {
                    "mccr_cos_primary": 0.35,
                    "tcs": 0.44,
                    "shuffle_gap": 0.01,
                    "cfo_gap": 0.05,
                    "cos_rep_g11": 0.99,
                },
            },
        }
    }

    score_payload = score_metrics_payload(metrics_payload)

    assert score_payload["ranked_checkpoints"][0]["method"] == "FCP-c6000"
    assert score_payload["guard_report"]["FCP-c6000"] == "pass"
    assert score_payload["guard_report"]["FCP-c7000"].startswith("fail:")

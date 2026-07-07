from qwen_vl.eval.stage1_cbeval.build_revsi_scene_splits import allocate_counts, sample_scenes_by_dataset


def test_allocate_counts_matches_total():
    allocation = allocate_counts(
        {
            "ARKitScenes": 170,
            "ScanNetv2": 92,
            "ScanNetPPv2": 48,
            "3RScan": 46,
            "MultiScan": 24,
        },
        128,
    )
    assert sum(allocation.values()) == 128
    assert allocation == {
        "3RScan": 16,
        "ARKitScenes": 57,
        "MultiScan": 8,
        "ScanNetPPv2": 16,
        "ScanNetv2": 31,
    }


def test_sample_scenes_by_dataset_respects_split_sizes():
    inventory = []
    dataset_counts = {
        "ARKitScenes": 10,
        "ScanNetv2": 6,
        "ScanNetPPv2": 4,
    }
    for dataset, count in dataset_counts.items():
        for index in range(count):
            inventory.append(
                {
                    "scene_id": f"{dataset}_{index:02d}",
                    "dataset": dataset,
                    "question_count": index + 1,
                    "question_type_histogram": {"dummy": index + 1},
                    "num_frames_values": ["32"],
                    "has_video": True,
                    "has_sampled_frame_indices": True,
                }
            )

    selection, locked_test, reserve, counts_by_split = sample_scenes_by_dataset(
        inventory,
        selection_scenes=8,
        locked_test_scenes=8,
        seed=20260524,
    )

    assert len(selection) == 8
    assert len(locked_test) == 8
    assert len(reserve) == 4
    assert counts_by_split["selection_dev"] == {
        "ARKitScenes": 4,
        "ScanNetPPv2": 2,
        "ScanNetv2": 2,
    }
    assert counts_by_split["locked_test"] == {
        "ARKitScenes": 4,
        "ScanNetPPv2": 1,
        "ScanNetv2": 3,
    }
    selection_ids = {row["scene_id"] for row in selection}
    locked_ids = {row["scene_id"] for row in locked_test}
    reserve_ids = {row["scene_id"] for row in reserve}
    assert not (selection_ids & locked_ids)
    assert not (selection_ids & reserve_ids)
    assert not (locked_ids & reserve_ids)

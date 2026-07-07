import json
import sys
from pathlib import Path

import qwen_vl.eval.stage1_cbeval.build_splits as build_splits


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_splits_is_source_level_and_leak_free(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    train_manifest_path = tmp_path / "train_manifest.jsonl"
    output_dir = tmp_path / "outputs"

    manifest_rows = [
        {"group_id": "s0_w0", "source_dataset": "revsi", "source_sample_id": "s0", "window_id": "window_0", "frame_paths": ["a/0.png"]},
        {"group_id": "s0_w1", "source_dataset": "revsi", "source_sample_id": "s0", "window_id": "window_1", "frame_paths": ["a/1.png"]},
        {"group_id": "s1_w0", "source_dataset": "revsi", "source_sample_id": "s1", "window_id": "window_0", "frame_paths": ["b/0.png"]},
        {"group_id": "s1_w1", "source_dataset": "revsi", "source_sample_id": "s1", "window_id": "window_1", "frame_paths": ["b/1.png"]},
        {"group_id": "s2_w0", "source_dataset": "revsi", "source_sample_id": "s2", "window_id": "window_0", "frame_paths": ["c/0.png"]},
        {"group_id": "s2_w1", "source_dataset": "revsi", "source_sample_id": "s2", "window_id": "window_1", "frame_paths": ["c/1.png"]},
        {"group_id": "s3_w0", "source_dataset": "revsi", "source_sample_id": "s3", "window_id": "window_0", "frame_paths": ["d/0.png"]},
        {"group_id": "s3_w1", "source_dataset": "revsi", "source_sample_id": "s3", "window_id": "window_1", "frame_paths": ["d/1.png"]},
    ]
    train_rows = [
        {"group_id": "train_w0", "source_dataset": "llava_hound_64k", "source_sample_id": "s0", "frame_paths": ["train/0.png"]},
    ]
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(train_manifest_path, train_rows)

    argv = sys.argv[:]
    try:
        sys.argv = [
            "build_splits.py",
            "--manifest",
            str(manifest_path),
            "--train_manifest",
            str(train_manifest_path),
            "--output_dir",
            str(output_dir),
            "--seed",
            "7",
            "--selection_sources",
            "2",
            "--test_sources",
            "1",
            "--max_windows_per_source",
            "1",
        ]
        build_splits.main()
    finally:
        sys.argv = argv

    selection_rows = [json.loads(line) for line in (output_dir / "selection_dev_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    locked_rows = [json.loads(line) for line in (output_dir / "locked_test_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    leakage_report = json.loads((output_dir / "leakage_report.json").read_text(encoding="utf-8"))

    assert leakage_report["ok"] is True
    assert {row["source_group_key"] for row in selection_rows}.isdisjoint({row["source_group_key"] for row in locked_rows})
    assert "s0" not in {row["source_group_key"] for row in selection_rows}
    assert "s0" not in {row["source_group_key"] for row in locked_rows}
    assert len(selection_rows) == 2
    assert len(locked_rows) == 1

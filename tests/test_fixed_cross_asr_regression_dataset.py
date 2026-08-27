from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.cross_asr_boundary_evaluation import (
    evaluate_cross_asr_boundaries,
)


DATASET_PATH = Path(
    "data/cross_asr_boundaries/20260826/fixed_regressions.json"
)


def test_fixed_cross_asr_regressions_are_reviewed_merge_examples() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    samples = dataset["samples"]

    assert dataset["annotation_status"] == "reviewed"
    assert len(samples) == 2
    assert len({sample["id"] for sample in samples}) == len(samples)
    assert {sample["gold_label"] for sample in samples} == {"merge"}
    assert {sample["review_status"] for sample in samples} == {"reviewed"}
    assert {sample["expected_text"] for sample in samples} == {
        "中はこんな感じです",
        "どうしてこのポッドキャストを使ったスピーキングの練習っていうのがいいのかっていうのをゆうゆくんはどうして知っているのかっていう話なんですが、僕自身プライベートレッスンの学生数名に",
    }


def test_fixed_cross_asr_regression_evidence_exists() -> None:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["samples"]

    for sample in samples:
        assert Path(sample["artifact_path"]).is_file()


def test_latest_artifacts_report_both_fixed_boundaries_as_missed_merges() -> None:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["samples"]

    for sample in samples:
        report = evaluate_cross_asr_boundaries(
            DATASET_PATH,
            Path(sample["artifact_path"]),
        )
        missed_ids = {
            item["id"] for item in report["errors"]["missed_merge"]
        }
        assert sample["id"] in missed_ids

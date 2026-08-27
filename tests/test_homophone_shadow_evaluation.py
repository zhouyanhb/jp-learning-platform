from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.homophone_shadow_evaluation import (
    evaluate_homophone_shadow_candidates,
    write_homophone_shadow_evaluation,
)


DATASET_PATH = Path("data/homophone_shadow_candidates/20260816/baseline.json")


def test_shadow_dataset_covers_three_candidate_strategies() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    assert dataset["annotation_status"] == "reviewed"
    assert {sample["strategy"] for sample in dataset["samples"]} == {
        "single_character",
        "cross_morpheme",
        "inflected",
    }
    assert len(dataset["samples"]) == 38
    assert sum(sample["label"] == "positive" for sample in dataset["samples"]) == 9
    assert sum(sample["label"] == "negative" for sample in dataset["samples"]) == 29
    draw_samples = [
        sample
        for sample in dataset["samples"]
        if sample["id"].startswith("inflected-positive-draw-")
    ]
    assert len(draw_samples) == 4
    assert {sample["surface"] for sample in draw_samples} == {"書い"}
    assert {sample["expected_candidate"] for sample in draw_samples} == {"描い"}
    new_inflected_negatives = [
        sample
        for sample in dataset["samples"]
        if sample["id"].startswith("inflected-negative-vlog-")
        or sample["id"].startswith("inflected-negative-podcast-")
    ]
    assert len(new_inflected_negatives) == 13
    assert {sample["surface"] for sample in new_inflected_negatives} >= {
        "飲み",
        "買っ",
        "話す",
        "撮っ",
        "取る",
        "経っ",
    }


def test_shadow_evaluator_reports_recall_and_false_negatives(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "single",
                        "source_path": "lesson.mp4",
                        "strategy": "single_character",
                        "time_range": {"start_seconds": 1.0, "end_seconds": 2.0},
                        "surface": "得",
                        "expected_candidate": "徳",
                    },
                    {
                        "id": "inflected",
                        "source_path": "lesson.mp4",
                        "strategy": "inflected",
                        "time_range": {"start_seconds": 1.0, "end_seconds": 2.0},
                        "surface": "詰め",
                        "expected_candidate": "積め",
                    },
                    {
                        "id": "negative",
                        "label": "negative",
                        "source_path": "lesson.mp4",
                        "strategy": "single_character",
                        "time_range": {"start_seconds": 1.0, "end_seconds": 2.0},
                        "surface": "次"
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    artifact_directory = tmp_path / "run" / "lesson"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "manifest.json").write_text(
        json.dumps({"source_path": "lesson.mp4"}),
        encoding="utf-8",
    )
    (artifact_directory / "04_homophone_resolution.json").write_text(
        json.dumps(
            {
                "context": {
                    "document": {
                        "segments": [
                            {
                                "position": 3,
                                "time_range": {
                                    "start_seconds": 1.0,
                                    "end_seconds": 2.0,
                                },
                            }
                        ]
                    }
                },
                "data": {
                    "shadow_candidates": [
                        {
                            "segment_position": 3,
                            "strategy": "single_character",
                            "surface": "得",
                            "candidates": ["徳"],
                            "original_score": 0.1,
                            "relative_acceptance_status": "accepted",
                            "relative_acceptance_reason": "candidate_score_higher",
                            "accepted_candidate": "徳",
                            "candidate_scores": [
                                {"text": "徳", "score": 0.8}
                            ],
                        },
                        {
                            "segment_position": 3,
                            "strategy": "single_character",
                            "surface": "次",
                            "candidates": ["継"],
                            "original_score": 0.1,
                            "relative_acceptance_status": "accepted",
                            "relative_acceptance_reason": "candidate_score_higher",
                            "accepted_candidate": "継",
                            "candidate_scores": [
                                {"text": "継", "score": 0.8}
                            ]
                        },
                        {
                            "segment_position": 3,
                            "strategy": "inflected",
                            "surface": "詰め",
                            "candidates": ["込め", "積め"],
                            "original_score": 0.1,
                            "relative_acceptance_status": "accepted",
                            "relative_acceptance_reason": "candidate_score_higher",
                            "accepted_candidate": "込め",
                            "candidate_scores": [
                                {"text": "込め", "score": 0.8},
                                {"text": "積め", "score": 0.2}
                            ],
                        },
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_homophone_shadow_candidates(dataset_path, tmp_path / "run")

    assert report["metrics"] == {
        "total": 3,
        "evaluated": 3,
        "positive_evaluated": 2,
        "negative_evaluated": 1,
        "generated": 2,
        "candidate_missing": 0,
        "target_missing": 0,
        "missing_artifact": 0,
        "target_recall": 1.0,
        "top_1_correct": 1,
        "top_1_accuracy": 0.5,
        "top_1_decision_correct": 1,
        "top_1_decision_accuracy": 1 / 3,
        "acceptance_true_positive": 1,
        "acceptance_false_positive": 2,
        "acceptance_evaluated": 3,
        "acceptance_positive_evaluated": 2,
        "acceptance_negative_evaluated": 1,
        "acceptance_not_evaluated": 0,
        "acceptance_precision": 1 / 3,
        "acceptance_recall": 0.5,
        "negative_passed": 0,
        "false_positive": 1,
        "false_positive_rate": 1.0,
        "score_missing": 0,
        "mean_top_1_margin": 0.6000000000000001,
        "mean_top_score_ratio_vs_original": 8.0,
        "candidate_count": 4,
    }
    assert report["false_negatives"] == []
    assert [item["id"] for item in report["misranked"]] == ["inflected"]
    assert [item["id"] for item in report["false_positives"]] == ["negative"]
    assert report["results"][1]["expected_rank"] == 2


def test_shadow_evaluator_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    report = write_homophone_shadow_evaluation(
        DATASET_PATH,
        tmp_path / "missing",
        output,
    )

    assert output.is_file()
    assert report["metrics"]["missing_artifact"] == 38

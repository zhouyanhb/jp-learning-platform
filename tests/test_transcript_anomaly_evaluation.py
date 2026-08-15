from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.transcript_anomaly_dataset import (
    build_transcript_anomaly_dataset,
)
from jp_learning_platform.transcript_anomaly_evaluation import (
    evaluate_transcript_anomalies,
)


def test_builds_sentence_scoped_anomaly_samples(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "context": {
                    "document": {
                        "source_path": "audio.mp3",
                        "segments": [
                            {
                                "position": 7,
                                "sentences": [
                                    {
                                        "text": "ふううううううう",
                                        "time_range": {
                                            "start_seconds": 1.0,
                                            "end_seconds": 2.0,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                },
                "data": {
                    "candidates": [
                        {
                            "kind": "possible_alignment_failure",
                            "segment_positions": [7],
                            "sentence_indexes": [0],
                        },
                        {
                            "kind": "possible_repeated_vocalization",
                            "segment_positions": [7],
                            "sentence_indexes": [],
                        },
                        {
                            "kind": "possible_asr_omission",
                            "segment_positions": [7],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = build_transcript_anomaly_dataset((artifact,))

    assert len(dataset["samples"]) == 1
    assert dataset["samples"][0]["predicted_anomaly_kinds"] == [
        "possible_alignment_failure",
        "possible_repeated_vocalization",
    ]


def test_dataset_can_include_neighboring_negative_candidates(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    sentences = [
        {
            "text": text,
            "time_range": {"start_seconds": index, "end_seconds": index + 0.5},
        }
        for index, text in enumerate(("before", "へへへ", "after"))
    ]
    artifact.write_text(
        json.dumps(
            {
                "context": {
                    "document": {
                        "segments": [
                            {"position": index, "sentences": [sentence]}
                            for index, sentence in enumerate(sentences)
                        ]
                    }
                },
                "data": {
                    "candidates": [
                        {
                            "kind": "possible_repeated_laughter",
                            "segment_positions": [1],
                            "sentence_indexes": [0],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = build_transcript_anomaly_dataset((artifact,), context_radius=1)

    assert len(dataset["samples"]) == 3
    assert [item["sample_kind"] for item in dataset["samples"]] == [
        "context_negative_candidate",
        "prediction",
        "context_negative_candidate",
    ]


def test_reports_overall_per_kind_and_error_details(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "anomaly_kinds": ["alignment", "repetition"],
                "documents": [{"id": "document-001"}],
                "samples": [
                    {
                        "id": "one",
                        "document_id": "document-001",
                        "segment_position": 1,
                        "sentence_index": 0,
                        "text": "a",
                        "predicted_anomaly_kinds": ["alignment", "repetition"],
                        "gold_anomaly_kinds": ["alignment"],
                        "review_status": "reviewed",
                    },
                    {
                        "id": "two",
                        "document_id": "document-001",
                        "segment_position": 2,
                        "sentence_index": 0,
                        "text": "b",
                        "predicted_anomaly_kinds": [],
                        "gold_anomaly_kinds": ["repetition"],
                        "review_status": "reviewed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_transcript_anomalies(annotation)

    assert report["metrics"] == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "evaluation_status": "evaluated",
        "predicted_support": 2,
        "gold_support": 2,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert report["metrics_by_anomaly_kind"]["alignment"]["precision"] == 1.0
    assert report["metrics_by_anomaly_kind"]["repetition"]["recall"] == 0.0
    assert report["errors"]["false_positive"][0]["anomaly_kind"] == "repetition"
    assert report["errors"]["false_negative"][0]["text"] == "b"


def test_reports_gold_only_kind_as_evaluated_with_zero_recall(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "anomaly_kinds": ["alignment"],
                "samples": [
                    {
                        "id": "one",
                        "document_id": "document-001",
                        "segment_position": 1,
                        "sentence_index": 0,
                        "predicted_anomaly_kinds": [],
                        "gold_anomaly_kinds": ["alignment"],
                        "review_status": "reviewed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluate_transcript_anomalies(annotation)[
        "metrics_by_anomaly_kind"
    ]["alignment"]

    assert metrics["precision"] is None
    assert metrics["recall"] == 0.0
    assert metrics["evaluation_status"] == "evaluated_no_predictions"

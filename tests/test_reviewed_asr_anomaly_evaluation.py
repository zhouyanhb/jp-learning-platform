from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.reviewed_asr_anomaly_evaluation import (
    evaluate_reviewed_asr_anomalies,
)


def test_evaluates_reviewed_ranges_by_kind_and_content_category(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "artifact"
    artifact_directory.mkdir()
    (artifact_directory / "04c_transcript_anomaly_analysis.json").write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "kind": "possible_asr_omission",
                            "time_range": {
                                "start_seconds": 1.0,
                                "end_seconds": 2.0,
                            },
                        },
                        {
                            "kind": "possible_asr_omission",
                            "time_range": {
                                "start_seconds": 5.0,
                                "end_seconds": 6.0,
                            },
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    annotation = tmp_path / "reviewed.json"
    annotation.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "one",
                        "source_path": "input/example.m4a",
                        "artifact_directory": str(artifact_directory),
                    }
                ],
                "samples": [
                    _sample("tp", 1.0, 2.0, ["possible_asr_omission"]),
                    _sample("fn", 3.0, 4.0, ["possible_asr_omission"]),
                    _sample("fp", 5.0, 6.0, []),
                    _sample("tn", 7.0, 8.0, []),
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_reviewed_asr_anomalies(annotation)

    assert report["metrics"] == {
        "evaluated": 4,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "missing_artifact": 0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert report["metrics_by_kind"]["possible_asr_omission"]["evaluated"] == 4
    assert report["metrics_by_content_category"]["interview_quote"][
        "evaluated"
    ] == 4


def _sample(
    sample_id: str,
    start_seconds: float,
    end_seconds: float,
    gold: list[str],
) -> dict[str, object]:
    return {
        "id": sample_id,
        "document_id": "one",
        "time_range": {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
        },
        "content_category": "interview_quote",
        "evaluated_anomaly_kinds": ["possible_asr_omission"],
        "gold_anomaly_kinds": gold,
        "review_status": "reviewed",
    }

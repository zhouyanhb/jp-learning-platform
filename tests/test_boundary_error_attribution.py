from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.boundary_error_attribution import (
    attribute_boundary_errors,
)


def test_attributes_missing_and_extra_boundaries(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    artifact_path = tmp_path / "artifact.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "errors": {
                    "false_negative": [
                        {
                            "sample_id": "sample-1",
                            "role": "dialogue",
                            "boundary_types": ["speaker_turn"],
                            "projected_hypothesis_range": [3, 3],
                        }
                    ],
                    "false_positive": [
                        {"hypothesis_position": 6, "hypothesis_context": "abc|def"}
                    ],
                    "unaligned_reference": [],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "context": {
                    "document": {
                        "segments": [
                            {
                                "sentences": [
                                    {
                                        "text": "abcdefghi",
                                        "words": [
                                            {
                                                "text": "abc",
                                                "time_range": {
                                                    "start_seconds": 0.0,
                                                    "end_seconds": 0.4,
                                                },
                                            },
                                            {
                                                "text": "def",
                                                "time_range": {
                                                    "start_seconds": 0.6,
                                                    "end_seconds": 1.0,
                                                },
                                            },
                                            {
                                                "text": "ghi",
                                                "time_range": {
                                                    "start_seconds": 2.6,
                                                    "end_seconds": 3.0,
                                                },
                                            },
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                },
                "data": {
                    "decisions": [
                        {
                            "left_text": "abcdef",
                            "right_text": "ghi",
                            "reason": "strong_pause",
                            "gap_seconds": 1.6,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    report = attribute_boundary_errors(evaluation_path, artifact_path)

    assert report["summary"]["false_negative_categories"] == {
        "speaker_turn_missing": 1
    }
    assert report["summary"]["false_positive_categories"] == {
        "strong_pause_over_split": 1
    }

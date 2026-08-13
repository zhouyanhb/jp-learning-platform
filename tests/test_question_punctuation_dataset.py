from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.question_punctuation_dataset import (
    build_question_punctuation_dataset,
)


def test_builds_reviewable_typed_question_dataset(tmp_path: Path) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:03,000
「トイレ借りていい？」って聞きました。

2
00:00:04,000 --> 00:00:05,000
何をしますか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "トイレ借りていい",
                            "time_range": {
                                "start_seconds": 1.5,
                                "end_seconds": 2.5,
                            },
                            "evidence": ["embedded_quoted_question"],
                        },
                        {
                            "segment_position": 1,
                            "sentence_index": 0,
                            "text": "今日は晴れですか",
                            "time_range": {
                                "start_seconds": 6.0,
                                "end_seconds": 7.0,
                            },
                            "evidence": ["semantic_question_boundary"],
                        },
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = build_question_punctuation_dataset(reference, artifact)

    assert dataset["annotation_status"] == "silver_needs_review"
    predictions = [
        item for item in dataset["samples"] if item["sample_kind"] == "prediction"
    ]
    assert predictions[0]["candidate_type"] == "embedded_quoted_question"
    assert predictions[0]["gold_label"] == "question"
    assert predictions[1]["candidate_type"] == "sentence_terminal_question"
    assert predictions[1]["gold_label"] == "non_question"
    missed = [
        item
        for item in dataset["samples"]
        if item["sample_kind"] == "missed_reference"
    ]
    assert len(missed) == 1
    assert missed[0]["gold_candidate_type"] is None
    assert all(item["review_status"] == "needs_review" for item in dataset["samples"])

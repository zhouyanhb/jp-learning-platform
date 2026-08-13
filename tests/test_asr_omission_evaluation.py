from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.asr_omission_evaluation import evaluate_asr_omissions


def test_evaluates_omission_precision_and_recall_from_reference_timeline(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:00,000 --> 00:00:01,000
こんにちは

2
00:00:02,000 --> 00:00:03,000
聞こえない部分

3
00:00:04,000 --> 00:00:05,000
続きます

4
00:00:08,000 --> 00:00:09,000
もう一つの欠落
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "context": {
                    "document": {
                        "segments": [
                            {
                                "text": "こんにちは",
                                "time_range": {
                                    "start_seconds": 0.0,
                                    "end_seconds": 1.0,
                                },
                            },
                            {
                                "text": "続きます",
                                "time_range": {
                                    "start_seconds": 4.0,
                                    "end_seconds": 5.0,
                                },
                            },
                        ]
                    }
                },
                "data": {
                    "candidates": [
                        {
                            "kind": "possible_asr_omission",
                            "time_range": {
                                "start_seconds": 1.0,
                                "end_seconds": 4.0,
                            },
                            "confidence": 0.8,
                            "evidence": ["gap"],
                        },
                        {
                            "kind": "possible_internal_asr_omission",
                            "time_range": {
                                "start_seconds": 6.0,
                                "end_seconds": 7.0,
                            },
                            "confidence": 0.7,
                            "evidence": ["internal_gap"],
                        },
                        {
                            "kind": "possible_background_speech",
                            "time_range": {
                                "start_seconds": 8.0,
                                "end_seconds": 9.0,
                            },
                        },
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_asr_omissions(reference, artifact)

    assert report["label_source"] == "reference-derived"
    assert report["coverage"]["predicted_regions"] == 2
    assert report["coverage"]["reference_omission_regions"] == 2
    assert report["metrics"] == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert len(report["errors"]["false_positive"]) == 1
    assert len(report["errors"]["false_negative"]) == 1

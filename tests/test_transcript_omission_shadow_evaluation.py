from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.transcript_omission_shadow_evaluation import (
    evaluate_transcript_omission_shadow,
)


def test_reports_detection_recovery_and_unsafe_validation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "04d_transcript_omission_shadow.json").write_text(
        json.dumps(
            {
                "data": {
                    "audits": [
                        _audit(
                            1.0,
                            2.0,
                            [],
                            False,
                            foreground_candidates=["あっ、ありがとう！"],
                        ),
                        _audit(3.0, 4.0, ["幻覚です"], True),
                    ]
                }
            },
            ensure_ascii=False,
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
                        "source_path": "input/one.mp4",
                        "artifact_directory": str(artifact),
                    }
                ],
                "samples": [
                    _sample("positive", 1.0, 2.0, True, "あっありがとう"),
                    _sample("negative", 3.0, 4.0, False, ""),
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_transcript_omission_shadow(annotation)

    assert report["metrics"]["detector"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_negative": 0,
        "precision": 0.5,
        "recall": 1.0,
    }
    assert report["metrics"]["recovery"]["recall"] == 1.0
    assert report["metrics"]["recovery"]["recovered_by_full_gap"] == 0
    assert report["metrics"]["recovery"]["recovered_by_foreground_probe"] == 1
    assert report["metrics"]["validation"] == {"passed": 1, "unsafe": 1}


def _sample(
    sample_id: str,
    start: float,
    end: float,
    expected_omission: bool,
    expected_text: str,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "document_id": "one",
        "time_range": {"start_seconds": start, "end_seconds": end},
        "content_category": "foreground_speech_in_music",
        "expected_omission": expected_omission,
        "expected_recovery_text": expected_text,
        "review_status": "reviewed",
    }


def _audit(
    start: float,
    end: float,
    candidates: list[str],
    validation_passed: bool,
    foreground_candidates: list[str] | None = None,
) -> dict[str, object]:
    return {
        "time_range": {"start_seconds": start, "end_seconds": end},
        "extracted_candidate_texts": candidates,
        "foreground_probe_audits": [
            {"extracted_text": text}
            for text in (foreground_candidates or [])
        ],
        "validation_passed": validation_passed,
    }

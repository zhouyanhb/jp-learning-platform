from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.cross_asr_boundary_dataset import (
    build_cross_asr_boundary_dataset,
)
from jp_learning_platform.cross_asr_boundary_evaluation import (
    evaluate_cross_asr_boundaries,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _word(text: str, start: float, end: float) -> dict[str, object]:
    return {
        "text": text,
        "time_range": {"start_seconds": start, "end_seconds": end},
    }


def _artifact(*, decisions: list[dict[str, object]] | None = None) -> dict[str, object]:
    first = {
        "text": "電車が",
        "time_range": {"start_seconds": 0.0, "end_seconds": 1.0},
        "words": [_word("電車が", 0.0, 1.0)],
        "asr_boundary_word_indexes": [],
    }
    second = {
        "text": "動き始めました",
        "time_range": {"start_seconds": 1.0, "end_seconds": 2.0},
        "words": [_word("動き始めました", 1.0, 2.0)],
        "asr_boundary_word_indexes": [],
    }
    return {
        "source_path": "input/train.mp4",
        "context": {
            "document": {
                "source_path": "input/train.mp4",
                "segments": [
                    {
                        "position": 0,
                        "text": "電車が動き始めました",
                        "time_range": {"start_seconds": 0.0, "end_seconds": 2.0},
                        "sentences": [first, second],
                    }
                ],
            }
        },
        "data": {"cross_segment_merges": decisions or []},
    }


def _decision() -> dict[str, object]:
    return {
        "left_end_seconds": 1.0,
        "right_start_seconds": 1.0,
        "score": 5,
        "reason": "cross_asr_syntactic_continuation",
        "evidence": [{"name": "tight_subject_predicate", "score": 4}],
    }


def test_cross_asr_dataset_keeps_prediction_separate_from_gold(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "source.json", _artifact())
    prediction = _write_json(tmp_path / "prediction.json", _artifact(decisions=[_decision()]))

    dataset = build_cross_asr_boundary_dataset(
        source,
        prediction_artifact_path=prediction,
    )

    assert len(dataset["samples"]) == 1
    sample = dataset["samples"][0]
    assert sample["predicted_label"] == "merge"
    assert sample["gold_label"] is None
    assert sample["review_status"] == "needs_review"
    assert sample["prediction"]["score"] == 5


def test_cross_asr_evaluator_reports_false_and_missed_merges(tmp_path: Path) -> None:
    artifact = _write_json(tmp_path / "artifact.json", _artifact(decisions=[_decision()]))
    dataset = build_cross_asr_boundary_dataset(artifact)
    false_merge = dataset["samples"][0]
    false_merge["gold_label"] = "keep"
    false_merge["review_status"] = "reviewed"
    missed_merge = {
        **false_merge,
        "id": "cross-asr-missed",
        "left_end_seconds": 3.0,
        "right_start_seconds": 3.1,
        "gold_label": "merge",
    }
    dataset["samples"].append(missed_merge)
    dataset_path = _write_json(tmp_path / "dataset.json", dataset)

    report = evaluate_cross_asr_boundaries(dataset_path, artifact)

    assert report["metrics"]["false_merge"] == 1
    assert report["metrics"]["missed_merge"] == 1
    assert report["metrics"]["merge_precision"] == 0.0
    assert not report["metrics"]["precision_target_met"]
    assert len(report["errors"]["false_merge"]) == 1
    assert len(report["errors"]["missed_merge"]) == 1

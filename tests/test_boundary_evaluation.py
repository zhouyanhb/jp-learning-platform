from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.boundary_evaluation import evaluate_boundary_artifact


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _dataset(sentences: tuple[str, ...]) -> dict[str, object]:
    offset = 0
    items = []
    for text in sentences:
        items.append(
            {
                "text": text,
                "role": "dialogue",
                "start_char": offset,
                "end_char": offset + len(text),
            }
        )
        offset += len(text)
    return {"samples": [{"id": "sample-1", "sentences": items}]}


def _artifact(sentences: tuple[str, ...]) -> dict[str, object]:
    return {
        "context": {
            "document": {
                "segments": [
                    {"sentences": [{"text": text} for text in sentences]}
                ]
            }
        }
    }


def test_scores_exact_sentence_boundaries(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    artifact_path = tmp_path / "artifact.json"
    _write_json(dataset_path, _dataset(("今日は晴れです", "散歩します")))
    _write_json(artifact_path, _artifact(("今日は晴れです。", "散歩します。")))

    report = evaluate_boundary_artifact(dataset_path, artifact_path)

    assert report["metrics"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "unaligned_reference": 0,
        "ignored_hypothesis": 0,
    }


def test_reports_extra_and_missing_boundaries(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    artifact_path = tmp_path / "artifact.json"
    _write_json(dataset_path, _dataset(("今日は晴れです", "散歩します")))
    _write_json(artifact_path, _artifact(("今日は", "晴れです散歩します")))

    report = evaluate_boundary_artifact(dataset_path, artifact_path)

    assert report["metrics"]["true_positive"] == 0
    assert report["metrics"]["false_positive"] == 1
    assert report["metrics"]["false_negative"] == 1


def test_ignores_unaligned_audio_outside_reference(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    artifact_path = tmp_path / "artifact.json"
    _write_json(dataset_path, _dataset(("今日は晴れです", "散歩します")))
    _write_json(
        artifact_path,
        _artifact(("説明を始めます", "今日は晴れです", "散歩します")),
    )

    report = evaluate_boundary_artifact(dataset_path, artifact_path)

    assert report["metrics"]["true_positive"] == 1
    assert report["metrics"]["false_positive"] == 0
    assert report["metrics"]["ignored_hypothesis"] == 1


def test_accepts_boundary_inside_an_asr_insertion(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    artifact_path = tmp_path / "artifact.json"
    _write_json(dataset_path, _dataset(("今日は晴れです", "散歩します")))
    _write_json(artifact_path, _artifact(("今日は晴れですね", "散歩します")))

    report = evaluate_boundary_artifact(dataset_path, artifact_path)

    assert report["metrics"]["true_positive"] == 1
    assert report["metrics"]["false_negative"] == 0


def test_structure_only_boundary_is_excluded_from_language_metrics(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    artifact_path = tmp_path / "artifact.json"
    dataset = _dataset(("問題一", "選択肢一"))
    sample = dataset["samples"][0]
    sample["boundaries"] = [
        {
            "after_char": 3,
            "types": ["source_line_break"],
            "dimensions": ["content_structure"],
        }
    ]
    _write_json(dataset_path, dataset)
    _write_json(artifact_path, _artifact(("問題一", "選択肢一")))

    report = evaluate_boundary_artifact(dataset_path, artifact_path)

    assert report["metrics"]["false_positive"] == 0
    assert report["metrics"]["ignored_hypothesis"] == 1
    assert report["dimensions"]["content_structure"]["reference_boundaries"] == 1

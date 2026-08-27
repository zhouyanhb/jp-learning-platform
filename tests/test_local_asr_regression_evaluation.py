from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.local_asr_regression_evaluation import (
    evaluate_local_asr_regressions,
)


def _stage(path: Path, text: str) -> None:
    path.write_text(
        json.dumps(
            {
                "data": {
                    "segments": [
                        {
                            "position": 0,
                            "sentences": [
                                {
                                    "text": text,
                                    "time_range": {
                                        "start_seconds": 1.0,
                                        "end_seconds": 2.0,
                                    },
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def _artifact_directory(root: Path, name: str, text: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for filename in (
        "01_whisper.json",
        "04_homophone_resolution.json",
        "06_word_normalization.json",
    ):
        _stage(directory / filename, text)
    return directory


def test_evaluates_unresolved_resolved_and_regressed_samples(tmp_path: Path) -> None:
    wrong = _artifact_directory(tmp_path, "wrong", "必要な得が不足")
    fixed = _artifact_directory(tmp_path, "fixed", "何を")
    regressed = _artifact_directory(tmp_path, "regressed", "なにょ")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "samples": [
                    _sample("one", "unresolved", wrong, "得", "徳"),
                    _sample("two", "unresolved", fixed, "なにょ", "何を"),
                    _sample(
                        "three",
                        "resolved_regression_guard",
                        regressed,
                        "何を",
                        "何を",
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_local_asr_regressions(dataset)

    assert report["metrics"] == {
        "total": 3,
        "evaluated": 3,
        "resolved": 1,
        "unresolved": 1,
        "regressed": 1,
        "changed_unverified": 0,
        "missing_artifact": 0,
        "resolution_rate": 1 / 3,
    }
    assert [item["status"] for item in report["results"]] == [
        "unresolved",
        "resolved",
        "regressed",
    ]


def test_can_match_artifacts_from_a_new_run_by_source_name(tmp_path: Path) -> None:
    run = tmp_path / "run"
    artifact = _artifact_directory(run, "safe-name", "必要な徳が不足")
    (artifact / "manifest.json").write_text(
        json.dumps({"source_path": "input/temp/lesson.mp4"}),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"
    sample = _sample("one", "unresolved", tmp_path / "old", "得", "徳")
    sample["source_path"] = "input/source/lesson.mp4"
    dataset.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")

    report = evaluate_local_asr_regressions(dataset, artifact_root=run)

    assert report["results"][0]["status"] == "resolved"
    assert report["results"][0]["artifact_directory"] == str(artifact)


def _sample(
    sample_id: str,
    status: str,
    artifact_directory: Path,
    observed: str,
    expected: str,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "status": status,
        "failure_layer": "candidate_generation",
        "source_path": "input/source/example.mp4",
        "artifact_directory": str(artifact_directory),
        "time_range": {"start_seconds": 1.0, "end_seconds": 2.0},
        "target_observed": observed,
        "target_expected": expected,
    }

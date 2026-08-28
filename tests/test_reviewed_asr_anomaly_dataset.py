from __future__ import annotations

import json
from pathlib import Path


DATASET_PATH = Path(
    "data/reviewed_asr_anomalies/20260827/reviewed.json"
)


def test_reviewed_asr_anomaly_dataset_has_fixed_unique_samples() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    samples = dataset["samples"]

    assert dataset["annotation_status"] == "reviewed"
    assert len(samples) == 7
    assert len({sample["id"] for sample in samples}) == len(samples)
    assert all(sample["review_status"] == "reviewed" for sample in samples)
    assert all(sample["evaluated_anomaly_kinds"] for sample in samples)


def test_reviewed_asr_anomaly_dataset_keeps_content_layers_separate() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    samples = dataset["samples"]

    assert {sample["content_category"] for sample in samples} == {
        "interview_quote",
        "mixed_language_promotion",
        "overlapping_background_speech",
        "foreground_reaction",
    }
    assert all("sentence" not in sample for sample in samples)
    assert {
        kind
        for sample in samples
        for kind in sample["gold_anomaly_kinds"]
    } == {
        "possible_asr_omission",
        "possible_background_speech",
        "possible_mixed_language_asr_error",
    }


def test_reviewed_asr_anomaly_dataset_evidence_exists() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    for document in dataset["documents"]:
        artifact_directory = Path(document["artifact_directory"])
        assert (artifact_directory / "04c_transcript_anomaly_analysis.json").is_file()

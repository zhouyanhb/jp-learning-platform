from __future__ import annotations

import json
from pathlib import Path


DATASET_PATH = Path(
    "data/local_asr_regressions/20260816/baseline.json"
)


def test_fixed_local_asr_dataset_has_reviewed_unique_samples() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    samples = dataset["samples"]

    assert dataset["annotation_status"] == "reviewed"
    assert len(samples) == 11
    assert len({sample["id"] for sample in samples}) == len(samples)
    assert {sample["status"] for sample in samples} == {
        "unresolved",
        "resolved_regression_guard",
    }
    assert all(sample["observed_text"] for sample in samples)
    assert all(sample["expected_text"] for sample in samples)
    assert all(sample["audit"]["decision_reason"] for sample in samples)


def test_fixed_local_asr_dataset_covers_expected_targets_and_layers() -> None:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["samples"]

    assert {sample["target_expected"] for sample in samples} == {
        "オオアリクイ",
        "徳",
        "中学時代",
        "年商10億",
        "徳が積めそう",
        "札勘する舞の手元の寄り",
        "なって、止まりました",
        "何を",
        "特急券",
        "都留市駅",
        "作文",
    }
    assert {
        sample["failure_layer"] for sample in samples
    } == {
        "candidate_generation",
        "local_asr_candidate_generation",
        "local_asr_candidate_acceptance",
        "candidate_rejection",
    }


def test_dataset_reference_and_artifact_evidence_exists() -> None:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["samples"]

    for sample in samples:
        assert Path(sample["reference_path"]).is_file()
        artifact_directory = Path(sample["artifact_directory"])
        assert (artifact_directory / "01_whisper.json").is_file()
        assert (artifact_directory / "04_homophone_resolution.json").is_file()
        assert (artifact_directory / "06_word_normalization.json").is_file()


def test_candidate_generation_failures_have_reviewed_root_causes() -> None:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["samples"]
    failures = [
        sample
        for sample in samples
        if sample["failure_layer"] == "candidate_generation"
    ]

    assert {sample["audit"]["root_cause"] for sample in failures} == {
        "strict_homophone_not_applicable",
        "single_character_target_filtered",
        "cross_morpheme_compound_not_targeted",
        "inflected_candidate_missing_from_vocabulary_composition",
    }

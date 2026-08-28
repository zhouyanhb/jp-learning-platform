from __future__ import annotations

import json
from pathlib import Path


DATASET_PATH = Path("data/transcript_omission_shadow/20260828/reviewed.json")


def test_shadow_dataset_contains_reviewed_positive_and_negative_contexts() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    samples = dataset["samples"]

    assert dataset["annotation_status"] == "reviewed"
    assert len(samples) == 4
    assert len({sample["id"] for sample in samples}) == len(samples)
    assert sum(bool(sample["expected_omission"]) for sample in samples) == 1
    assert all(sample["review_status"] == "reviewed" for sample in samples)
    assert all("sentence" not in sample for sample in samples)


def test_shadow_dataset_artifacts_and_references_exist() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    for document in dataset["documents"]:
        assert Path(document["reference_path"]).is_file()
        assert (
            Path(document["artifact_directory"])
            / "04d_transcript_omission_shadow.json"
        ).is_file()

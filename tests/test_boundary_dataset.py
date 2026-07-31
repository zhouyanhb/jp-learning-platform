from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.boundary_dataset import build_boundary_dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "input" / "2021年12月N2听力.md"
DATASET_PATH = ROOT / "data" / "sentence_boundaries" / "2021_12_n2.json"


def test_generated_n2_boundary_dataset_is_current_and_self_consistent() -> None:
    expected = build_boundary_dataset(SOURCE_PATH)
    stored = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    assert stored == expected
    assert stored["schema_version"] == 2
    assert stored["annotation_status"] == "silver"
    assert len(stored["samples"]) >= 20

    for sample in stored["samples"]:
        sentences = sample["sentences"]
        boundaries = sample["boundaries"]
        assert "".join(sentence["text"] for sentence in sentences) == sample["input_text"]
        assert [sentence["end_char"] for sentence in sentences[:-1]] == [
            boundary["after_char"] for boundary in boundaries
        ]
        assert all(
            sentence["start_char"] < sentence["end_char"]
            for sentence in sentences
        )
        assert all(
            sentence["role"] in {"instruction", "dialogue", "question", "option"}
            for sentence in sentences
        )
        assert all(
            0 < boundary["after_char"] < len(sample["input_text"])
            for boundary in boundaries
        )


def test_n2_boundary_dataset_contains_semantic_boundaries_without_punctuation() -> None:
    dataset = build_boundary_dataset(SOURCE_PATH)
    first = dataset["samples"][0]

    assert "。" not in first["input_text"]
    assert "、" not in first["input_text"]
    assert "女：" not in first["input_text"]
    assert len(first["sentences"]) > 5
    assert "speaker_turn" in {
        boundary_type
        for boundary in first["boundaries"]
        for boundary_type in boundary["types"]
    }


def test_n2_boundary_dataset_assigns_exam_roles_from_structure() -> None:
    dataset = build_boundary_dataset(SOURCE_PATH)
    samples = {sample["id"]: sample for sample in dataset["samples"]}

    four_options = samples["2021-12-n2-012"]["sentences"]
    assert sum(item["role"] == "question" for item in four_options) == 1
    assert sum(item["role"] == "option" for item in four_options) == 4

    response_item = samples["2021-12-n2-017"]["sentences"]
    assert [item["role"] for item in response_item] == [
        "instruction",
        "dialogue",
        "option",
        "option",
        "option",
    ]

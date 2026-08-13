"""Build reviewable, typed Japanese question-punctuation datasets."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from jp_learning_platform.question_punctuation_evaluation import (
    evaluate_question_punctuation,
)


_CANDIDATE_TYPES = (
    "sentence_terminal_question",
    "embedded_quoted_question",
    "elliptical_question",
)


def build_question_punctuation_dataset(
    reference_path: Path,
    artifact_path: Path,
) -> dict[str, object]:
    report = evaluate_question_punctuation(reference_path, artifact_path)
    matched = {
        _prediction_key(item["prediction"]): item
        for item in report["matches"]
    }
    false_positive = {
        _prediction_key(item): item
        for item in report["errors"]["false_positive"]
    }
    predictions = tuple(
        (*matched.keys(), *false_positive.keys())
    )
    samples = []
    for index, key in enumerate(
        sorted(predictions, key=lambda item: (item[0], item[1], item[2])),
        start=1,
    ):
        match = matched.get(key)
        prediction = (
            match["prediction"] if match is not None else false_positive[key]
        )
        samples.append(
            {
                "id": f"question-candidate-{index:05d}",
                "sample_kind": "prediction",
                "candidate_type": _candidate_type(prediction),
                "predicted_text": prediction["text"],
                "predicted_time_range": _time_range(prediction),
                "evidence": list(prediction.get("evidence") or ()),
                "automatic_match": "true_positive" if match else "false_positive",
                "reference": match["reference"] if match else None,
                "gold_label": "question" if match else "non_question",
                "gold_candidate_type": (
                    _candidate_type(prediction) if match else None
                ),
                "review_status": "needs_review",
                "review_note": "",
            }
        )
    for missed_index, item in enumerate(
        report["errors"]["false_negative"],
        start=1,
    ):
        samples.append(
            {
                "id": f"missed-reference-{missed_index:05d}",
                "sample_kind": "missed_reference",
                "candidate_type": None,
                "predicted_text": None,
                "predicted_time_range": None,
                "evidence": [],
                "automatic_match": "false_negative",
                "reference": {
                    key: value
                    for key, value in item.items()
                    if key != "overlapping_language_sentences"
                },
                "language_sentence_context": item.get(
                    "overlapping_language_sentences", []
                ),
                "gold_label": "question",
                "gold_candidate_type": None,
                "review_status": "needs_review",
                "review_note": "",
            }
        )
    return {
        "schema_version": 1,
        "language": "ja",
        "annotation_status": "silver_needs_review",
        "reference": str(reference_path),
        "candidate_artifact": str(artifact_path),
        "candidate_types": {
            "sentence_terminal_question": "Question at a language-sentence endpoint.",
            "embedded_quoted_question": "Question embedded inside quoted speech.",
            "elliptical_question": "Short question without an explicit predicate.",
        },
        "gold_labels": {
            "question": "Reference evidence supports a question mark at this span.",
            "non_question": "Reference evidence does not support the predicted question.",
            "uncertain": "Human review cannot decide reliably.",
        },
        "review_instructions": [
            "Confirm gold_label for every prediction sample.",
            "Assign gold_candidate_type for every confirmed question.",
            "Review missed references before using recall by candidate type.",
        ],
        "evaluation_snapshot": {
            "metrics": report["metrics"],
            "metrics_by_candidate_type": report["metrics_by_candidate_type"],
        },
        "samples": samples,
    }


def write_question_punctuation_dataset(
    reference_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, object]:
    dataset = build_question_punctuation_dataset(reference_path, artifact_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(dataset, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return dataset


def _prediction_key(item: dict[str, object]) -> tuple[float, float, str]:
    time_range = _time_range(item)
    return (
        float(time_range["start_seconds"]),
        float(time_range["end_seconds"]),
        str(item["text"]),
    )


def _time_range(item: dict[str, object]) -> dict[str, float]:
    return {
        "start_seconds": float(item["start_seconds"]),
        "end_seconds": float(item["end_seconds"]),
    }


def _candidate_type(item: dict[str, object]) -> str:
    evidence = set(item.get("evidence") or ())
    if "embedded_quoted_question" in evidence:
        return "embedded_quoted_question"
    if "short_pronominal_case_phrase" in evidence:
        return "elliptical_question"
    return "sentence_terminal_question"


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dataset = write_question_punctuation_dataset(
        args.reference,
        args.artifact,
        args.output,
    )
    print(f"samples={len(dataset['samples'])}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_question_punctuation_dataset",
    "write_question_punctuation_dataset",
]

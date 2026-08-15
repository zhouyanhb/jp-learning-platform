"""Evaluate transcript-content anomaly isolation against reviewed labels."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path


def evaluate_transcript_anomalies(annotation_path: Path) -> dict[str, object]:
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    artifact_predictions = _artifact_predictions(annotation)
    reviewed = tuple(
        sample
        for sample in annotation.get("samples") or ()
        if sample.get("review_status") == "reviewed"
    )
    kinds = tuple(str(kind) for kind in annotation.get("anomaly_kinds") or ())
    true_positive: list[dict[str, object]] = []
    false_positive: list[dict[str, object]] = []
    false_negative: list[dict[str, object]] = []
    per_kind: dict[str, dict[str, int]] = {
        kind: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for kind in kinds
    }
    for sample in reviewed:
        key = _sample_key(sample)
        predicted = artifact_predictions.get(
            key,
            set(sample.get("predicted_anomaly_kinds") or ()),
        )
        gold = set(sample.get("gold_anomaly_kinds") or ())
        for kind in sorted(predicted | gold):
            detail = _detail(sample, kind)
            if kind in predicted and kind in gold:
                true_positive.append(detail)
                per_kind.setdefault(kind, _empty_counts())["true_positive"] += 1
            elif kind in predicted:
                false_positive.append(detail)
                per_kind.setdefault(kind, _empty_counts())["false_positive"] += 1
            else:
                false_negative.append(detail)
                per_kind.setdefault(kind, _empty_counts())["false_negative"] += 1
    counts = {
        "true_positive": len(true_positive),
        "false_positive": len(false_positive),
        "false_negative": len(false_negative),
    }
    unreviewed = tuple(
        sample
        for sample in annotation.get("samples") or ()
        if sample.get("review_status") != "reviewed"
    )
    annotated_keys = {_sample_key(sample) for sample in annotation.get("samples") or ()}
    unreviewed_predictions = tuple(
        {
            "document_id": key[0],
            "segment_position": key[1],
            "sentence_index": key[2],
            "predicted_anomaly_kinds": sorted(predicted),
        }
        for key, predicted in sorted(artifact_predictions.items())
        if key not in annotated_keys
    )
    return {
        "schema_version": 1,
        "annotation": str(annotation_path),
        "evaluation_scope": "reviewed_samples_only",
        "coverage": {
            "documents": len(annotation.get("documents") or ()),
            "samples": len(annotation.get("samples") or ()),
            "reviewed_samples": len(reviewed),
            "unreviewed_samples": len(unreviewed),
            "unreviewed_predictions": len(unreviewed_predictions),
            "coverage": _ratio(len(reviewed), len(annotation.get("samples") or ())),
        },
        "metrics": _metrics(counts),
        "metrics_by_anomaly_kind": {
            kind: _metrics(values) for kind, values in sorted(per_kind.items())
        },
        "matches": true_positive,
        "errors": {
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "unreviewed": [_sample_identity(sample) for sample in unreviewed],
        "unreviewed_predictions": list(unreviewed_predictions),
    }


def write_transcript_anomaly_evaluation(
    annotation_path: Path,
    output_path: Path,
) -> dict[str, object]:
    report = evaluate_transcript_anomalies(annotation_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def _empty_counts() -> dict[str, int]:
    return {"true_positive": 0, "false_positive": 0, "false_negative": 0}


def _metrics(counts: dict[str, int]) -> dict[str, object]:
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    predicted_support = tp + fp
    gold_support = tp + fn
    precision = _optional_ratio(tp, predicted_support)
    recall = _optional_ratio(tp, gold_support)
    f1 = (
        _ratio(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluation_status": _evaluation_status(
            predicted_support,
            gold_support,
        ),
        "predicted_support": predicted_support,
        "gold_support": gold_support,
        **counts,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _evaluation_status(predicted_support: int, gold_support: int) -> str:
    if predicted_support and gold_support:
        return "evaluated"
    if gold_support:
        return "evaluated_no_predictions"
    if predicted_support:
        return "evaluated_no_gold_positives"
    return "not_evaluated_no_support"


def _sample_identity(sample: dict[str, object]) -> dict[str, object]:
    return {
        key: sample.get(key)
        for key in (
            "id",
            "document_id",
            "segment_position",
            "sentence_index",
            "time_range",
            "text",
        )
    }


def _detail(sample: dict[str, object], kind: str) -> dict[str, object]:
    return {**_sample_identity(sample), "anomaly_kind": kind}


def _sample_key(sample: dict[str, object]) -> tuple[str, int, int]:
    return (
        str(sample["document_id"]),
        int(sample["segment_position"]),
        int(sample["sentence_index"]),
    )


def _artifact_predictions(
    annotation: dict[str, object],
) -> dict[tuple[str, int, int], set[str]]:
    predictions: dict[tuple[str, int, int], set[str]] = {}
    allowed_kinds = set(annotation.get("anomaly_kinds") or ())
    for document_entry in annotation.get("documents") or ():
        artifact_path = document_entry.get("artifact")
        if not artifact_path:
            continue
        artifact = json.loads(Path(str(artifact_path)).read_text(encoding="utf-8"))
        document = ((artifact.get("context") or {}).get("document") or {})
        sentence_counts = {
            int(segment["position"]): len(segment.get("sentences") or ())
            for segment in document.get("segments") or ()
        }
        for candidate in (artifact.get("data") or {}).get("candidates") or ():
            kind = str(candidate.get("kind") or "")
            if kind not in allowed_kinds:
                continue
            for position_value in candidate.get("segment_positions") or ():
                position = int(position_value)
                indexes = candidate.get("sentence_indexes") or range(
                    sentence_counts.get(position, 0)
                )
                for sentence_index in indexes:
                    key = (str(document_entry["id"]), position, int(sentence_index))
                    predictions.setdefault(key, set()).add(kind)
    return predictions


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = write_transcript_anomaly_evaluation(args.annotation, args.output)
    metrics = report["metrics"]
    print(
        f"precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

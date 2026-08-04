"""Evaluate reviewed cross-ASR merge decisions."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from jp_learning_platform.cross_asr_boundary_dataset import (
    DEFAULT_MATCH_TOLERANCE_SECONDS,
)


def evaluate_cross_asr_boundaries(
    dataset_path: Path,
    artifact_path: Path,
    *,
    tolerance_seconds: float = DEFAULT_MATCH_TOLERANCE_SECONDS,
) -> dict[str, object]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    predictions = list((artifact.get("data") or {}).get("cross_segment_merges") or [])
    reviewed = [
        sample
        for sample in dataset["samples"]
        if sample.get("review_status") == "reviewed"
        and sample.get("gold_label") in {"merge", "keep", "uncertain"}
    ]
    outcomes = []
    for sample in reviewed:
        predicted_merge = any(
            abs(float(item["left_end_seconds"]) - float(sample["left_end_seconds"]))
            <= tolerance_seconds
            and abs(float(item["right_start_seconds"]) - float(sample["right_start_seconds"]))
            <= tolerance_seconds
            for item in predictions
        )
        outcomes.append((sample, predicted_merge))

    true_positive = sum(s["gold_label"] == "merge" and p for s, p in outcomes)
    false_positive = sum(s["gold_label"] == "keep" and p for s, p in outcomes)
    false_negative = sum(s["gold_label"] == "merge" and not p for s, p in outcomes)
    true_negative = sum(s["gold_label"] == "keep" and not p for s, p in outcomes)
    uncertain_kept = sum(s["gold_label"] == "uncertain" and not p for s, p in outcomes)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "artifact": str(artifact_path),
        "settings": {"tolerance_seconds": tolerance_seconds},
        "coverage": {
            "samples": len(dataset["samples"]),
            "reviewed": len(reviewed),
            "unreviewed": len(dataset["samples"]) - len(reviewed),
        },
        "metrics": {
            "merge_precision": precision,
            "merge_recall": recall,
            "true_merge": true_positive,
            "false_merge": false_positive,
            "missed_merge": false_negative,
            "true_keep": true_negative,
            "uncertain_kept": uncertain_kept,
            "precision_target_met": precision >= 0.95,
        },
        "errors": {
            "false_merge": [s for s, p in outcomes if s["gold_label"] == "keep" and p],
            "missed_merge": [s for s, p in outcomes if s["gold_label"] == "merge" and not p],
        },
    }


def write_cross_asr_boundary_evaluation(
    dataset_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, object]:
    report = evaluate_cross_asr_boundaries(dataset_path, artifact_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{json.dumps(report, ensure_ascii=False, indent=2)}\n", encoding="utf-8")
    return report


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = ArgumentParser(description="Evaluate reviewed cross-ASR boundaries.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = write_cross_asr_boundary_evaluation(args.dataset, args.artifact, args.output)
    metrics = report["metrics"]
    print(
        f"merge_precision={metrics['merge_precision']:.4f} "
        f"merge_recall={metrics['merge_recall']:.4f} "
        f"reviewed={report['coverage']['reviewed']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

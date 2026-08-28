"""Evaluate ASR anomaly candidates against fixed reviewed time ranges."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import unicodedata


DEFAULT_MATCH_IOU_THRESHOLD = 0.1


def evaluate_reviewed_asr_anomalies(
    annotation_path: Path,
    *,
    artifact_root: Path | None = None,
    match_iou_threshold: float = DEFAULT_MATCH_IOU_THRESHOLD,
) -> dict[str, object]:
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    run_artifacts = (
        _artifacts_by_source(Path(artifact_root))
        if artifact_root is not None
        else {}
    )
    documents = {
        str(document["id"]): document
        for document in annotation.get("documents") or ()
    }
    results: list[dict[str, object]] = []
    for sample in annotation.get("samples") or ():
        if sample.get("review_status") != "reviewed":
            continue
        document = documents[str(sample["document_id"])]
        artifact_directory = _artifact_directory(
            document,
            run_artifacts,
            artifact_root,
        )
        results.extend(
            _evaluate_sample(
                sample,
                artifact_directory,
                match_iou_threshold,
            )
        )
    return {
        "schema_version": 1,
        "annotation": str(annotation_path),
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "settings": {"match_iou_threshold": match_iou_threshold},
        "metrics": _metrics(results),
        "metrics_by_kind": _grouped_metrics(results, "kind"),
        "metrics_by_content_category": _grouped_metrics(
            results,
            "content_category",
        ),
        "results": results,
        "errors": {
            "false_positive": [
                item for item in results if item["status"] == "false_positive"
            ],
            "false_negative": [
                item for item in results if item["status"] == "false_negative"
            ],
            "missing_artifact": [
                item for item in results if item["status"] == "missing_artifact"
            ],
        },
    }


def write_reviewed_asr_anomaly_evaluation(
    annotation_path: Path,
    output_path: Path,
    *,
    artifact_root: Path | None = None,
    match_iou_threshold: float = DEFAULT_MATCH_IOU_THRESHOLD,
) -> dict[str, object]:
    report = evaluate_reviewed_asr_anomalies(
        annotation_path,
        artifact_root=artifact_root,
        match_iou_threshold=match_iou_threshold,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def _evaluate_sample(
    sample: dict[str, object],
    artifact_directory: Path | None,
    threshold: float,
) -> list[dict[str, object]]:
    gold = set(str(kind) for kind in sample.get("gold_anomaly_kinds") or ())
    evaluated = set(str(kind) for kind in sample.get("evaluated_anomaly_kinds") or ())
    kinds = sorted(gold | evaluated)
    predictions = (
        _artifact_predictions(artifact_directory)
        if artifact_directory is not None
        else ()
    )
    results: list[dict[str, object]] = []
    for kind in kinds:
        matching = [
            candidate
            for candidate in predictions
            if candidate["kind"] == kind
            and _temporal_iou(candidate["time_range"], sample["time_range"])
            >= threshold
        ]
        if artifact_directory is None:
            status = "missing_artifact"
        elif kind in gold and matching:
            status = "true_positive"
        elif kind in gold:
            status = "false_negative"
        elif matching:
            status = "false_positive"
        else:
            status = "true_negative"
        results.append(
            {
                "sample_id": sample["id"],
                "kind": kind,
                "content_category": sample["content_category"],
                "status": status,
                "time_range": sample["time_range"],
                "reference_text": sample.get("reference_text") or "",
                "observed_text": sample.get("observed_text") or "",
                "artifact_directory": (
                    str(artifact_directory) if artifact_directory is not None else None
                ),
                "matching_predictions": matching,
            }
        )
    return results


def _artifact_predictions(
    artifact_directory: Path,
) -> tuple[dict[str, object], ...]:
    artifact_path = artifact_directory / "04c_transcript_anomaly_analysis.json"
    if not artifact_path.is_file():
        return ()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return tuple(
        {
            "kind": str(candidate.get("kind") or ""),
            "time_range": candidate.get("time_range") or {},
            "confidence": candidate.get("confidence"),
            "evidence": candidate.get("evidence") or [],
        }
        for candidate in (artifact.get("data") or {}).get("candidates") or ()
        if candidate.get("time_range")
    )


def _artifact_directory(
    document: dict[str, object],
    run_artifacts: dict[str, Path],
    artifact_root: Path | None,
) -> Path | None:
    if artifact_root is None:
        path = Path(str(document["artifact_directory"]))
        return path if path.is_dir() else None
    return run_artifacts.get(_source_key(str(document["source_path"])))


def _artifacts_by_source(root: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    if not root.is_dir():
        return artifacts
    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = str(manifest.get("source_path") or "")
        if source_path:
            artifacts[_source_key(source_path)] = manifest_path.parent
    return artifacts


def _source_key(value: str) -> str:
    stem = Path(unicodedata.normalize("NFKC", value)).stem
    return "".join(character for character in stem if character.isalnum()).lower()


def _temporal_iou(left: dict[str, object], right: dict[str, object]) -> float:
    left_start = float(left.get("start_seconds", 0.0))
    left_end = float(left.get("end_seconds", left_start))
    right_start = float(right.get("start_seconds", 0.0))
    right_end = float(right.get("end_seconds", right_start))
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return overlap / union if union else 0.0


def _metrics(results: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(str(item["status"]) for item in results)
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return {
        "evaluated": len(results) - counts["missing_artifact"],
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": counts["true_negative"],
        "missing_artifact": counts["missing_artifact"],
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
    }


def _grouped_metrics(
    results: list[dict[str, object]],
    field: str,
) -> dict[str, dict[str, object]]:
    values = sorted({str(item[field]) for item in results})
    return {
        value: _metrics([item for item in results if item[field] == value])
        for value in values
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    report = write_reviewed_asr_anomaly_evaluation(
        args.annotation,
        args.output,
        artifact_root=args.artifact_root,
    )
    metrics = report["metrics"]
    print(
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"tp={metrics['true_positive']} "
        f"fp={metrics['false_positive']} "
        f"fn={metrics['false_negative']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

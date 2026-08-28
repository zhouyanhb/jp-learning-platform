"""Evaluate non-destructive omission retries against reviewed contexts."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import unicodedata


DEFAULT_MATCH_IOU_THRESHOLD = 0.5


def evaluate_transcript_omission_shadow(
    annotation_path: Path,
    *,
    artifact_root: Path | None = None,
    match_iou_threshold: float = DEFAULT_MATCH_IOU_THRESHOLD,
) -> dict[str, object]:
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    documents = {
        str(document["id"]): document
        for document in annotation.get("documents") or ()
    }
    run_artifacts = _artifacts_by_source(artifact_root) if artifact_root else {}
    results = [
        _evaluate_sample(
            sample,
            _artifact_directory(
                documents[str(sample["document_id"])],
                run_artifacts,
                artifact_root,
            ),
            match_iou_threshold,
        )
        for sample in annotation.get("samples") or ()
        if sample.get("review_status") == "reviewed"
    ]
    return {
        "schema_version": 1,
        "annotation": str(annotation_path),
        "artifact_root": str(artifact_root) if artifact_root else None,
        "settings": {"match_iou_threshold": match_iou_threshold},
        "metrics": _metrics(results),
        "metrics_by_content_category": _grouped_metrics(results),
        "results": results,
        "errors": {
            "detector_false_positive": [
                item for item in results if item["detector_status"] == "false_positive"
            ],
            "detector_false_negative": [
                item for item in results if item["detector_status"] == "false_negative"
            ],
            "recovery_false_negative": [
                item for item in results if item["recovery_status"] == "false_negative"
            ],
            "unsafe_validation": [
                item for item in results if item["unsafe_validation"]
            ],
        },
    }


def write_transcript_omission_shadow_evaluation(
    annotation_path: Path,
    output_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, object]:
    report = evaluate_transcript_omission_shadow(
        annotation_path,
        artifact_root=artifact_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def _evaluate_sample(
    sample: dict[str, object],
    artifact_directory: Path | None,
    threshold: float,
) -> dict[str, object]:
    audits = _artifact_audits(artifact_directory) if artifact_directory else ()
    matching = [
        audit
        for audit in audits
        if _temporal_iou(audit.get("time_range") or {}, sample["time_range"])
        >= threshold
    ]
    expected_omission = bool(sample.get("expected_omission"))
    triggered = bool(matching)
    if artifact_directory is None:
        detector_status = "missing_artifact"
    elif expected_omission and triggered:
        detector_status = "true_positive"
    elif expected_omission:
        detector_status = "false_negative"
    elif triggered:
        detector_status = "false_positive"
    else:
        detector_status = "true_negative"

    target = _normalize_text(str(sample.get("expected_recovery_text") or ""))
    recovered = bool(target) and any(
        target in _normalize_text(str(candidate))
        for audit in matching
        for candidate in audit.get("extracted_candidate_texts") or ()
    )
    if not expected_omission:
        recovery_status = "not_applicable"
    elif recovered:
        recovery_status = "true_positive"
    else:
        recovery_status = "false_negative"
    validation_passed = any(
        bool(audit.get("validation_passed")) for audit in matching
    )
    unsafe_validation = validation_passed and not recovered
    return {
        "sample_id": sample["id"],
        "content_category": sample["content_category"],
        "time_range": sample["time_range"],
        "reference_text": sample.get("reference_text") or "",
        "expected_omission": expected_omission,
        "detector_status": detector_status,
        "recovery_status": recovery_status,
        "target_recovered": recovered,
        "validation_passed": validation_passed,
        "unsafe_validation": unsafe_validation,
        "artifact_directory": str(artifact_directory) if artifact_directory else None,
        "matching_audits": matching,
    }


def _artifact_audits(artifact_directory: Path) -> tuple[dict[str, object], ...]:
    path = artifact_directory / "04d_transcript_omission_shadow.json"
    if not path.is_file():
        return ()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return tuple((artifact.get("data") or {}).get("audits") or ())


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
    if not root.is_dir():
        return {}
    artifacts: dict[str, Path] = {}
    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = str(manifest.get("source_path") or "")
        if source_path:
            artifacts[_source_key(source_path)] = manifest_path.parent
    return artifacts


def _source_key(value: str) -> str:
    stem = Path(unicodedata.normalize("NFKC", value)).stem
    return "".join(character for character in stem if character.isalnum()).lower()


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized if character.isalnum())


def _temporal_iou(left: dict[str, object], right: dict[str, object]) -> float:
    left_start = float(left.get("start_seconds", 0.0))
    left_end = float(left.get("end_seconds", left_start))
    right_start = float(right.get("start_seconds", 0.0))
    right_end = float(right.get("end_seconds", right_start))
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return overlap / union if union else 0.0


def _metrics(results: list[dict[str, object]]) -> dict[str, object]:
    detector = Counter(str(item["detector_status"]) for item in results)
    recovery = Counter(str(item["recovery_status"]) for item in results)
    detector_precision = _ratio(
        detector["true_positive"],
        detector["true_positive"] + detector["false_positive"],
    )
    detector_recall = _ratio(
        detector["true_positive"],
        detector["true_positive"] + detector["false_negative"],
    )
    return {
        "evaluated": len(results) - detector["missing_artifact"],
        "detector": {
            "true_positive": detector["true_positive"],
            "false_positive": detector["false_positive"],
            "false_negative": detector["false_negative"],
            "true_negative": detector["true_negative"],
            "precision": detector_precision,
            "recall": detector_recall,
        },
        "recovery": {
            "expected": recovery["true_positive"] + recovery["false_negative"],
            "recovered": recovery["true_positive"],
            "missed": recovery["false_negative"],
            "recall": _ratio(
                recovery["true_positive"],
                recovery["true_positive"] + recovery["false_negative"],
            ),
        },
        "validation": {
            "passed": sum(bool(item["validation_passed"]) for item in results),
            "unsafe": sum(bool(item["unsafe_validation"]) for item in results),
        },
    }


def _grouped_metrics(
    results: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    categories = sorted({str(item["content_category"]) for item in results})
    return {
        category: _metrics(
            [item for item in results if item["content_category"] == category]
        )
        for category in categories
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    report = write_transcript_omission_shadow_evaluation(
        args.annotation,
        args.output,
        artifact_root=args.artifact_root,
    )
    metrics = report["metrics"]
    print(
        f"detector_precision={metrics['detector']['precision']:.4f} "
        f"detector_recall={metrics['detector']['recall']:.4f} "
        f"recovery_recall={metrics['recovery']['recall']:.4f} "
        f"unsafe_validation={metrics['validation']['unsafe']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

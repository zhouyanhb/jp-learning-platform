"""Evaluate fixed local-ASR regressions against pipeline stage artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import unicodedata


STAGE_FILES = {
    "whisper": "01_whisper.json",
    "homophone": "04_homophone_resolution.json",
    "final": "06_word_normalization.json",
}


def evaluate_local_asr_regressions(
    dataset_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, object]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    run_artifacts = (
        _artifacts_by_source(Path(artifact_root))
        if artifact_root is not None
        else {}
    )
    results: list[dict[str, object]] = []
    for sample in dataset.get("samples") or ():
        artifact_directory = _artifact_directory(sample, run_artifacts, artifact_root)
        if artifact_directory is None:
            results.append(_missing_result(sample))
            continue
        stage_observations = {
            stage: _stage_observation(
                artifact_directory / filename,
                sample,
            )
            for stage, filename in STAGE_FILES.items()
        }
        final = stage_observations["final"]
        status = _sample_status(sample, final)
        results.append(
            {
                "id": sample["id"],
                "baseline_status": sample["status"],
                "failure_layer": sample["failure_layer"],
                "status": status,
                "artifact_directory": str(artifact_directory),
                "target_observed": sample["target_observed"],
                "target_expected": sample["target_expected"],
                "stage_observations": stage_observations,
            }
        )
    status_counts = Counter(str(item["status"]) for item in results)
    evaluated = len(results) - status_counts["missing_artifact"]
    resolved = status_counts["resolved"] + status_counts["regression_guard_passed"]
    return {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "metrics": {
            "total": len(results),
            "evaluated": evaluated,
            "resolved": resolved,
            "unresolved": status_counts["unresolved"],
            "regressed": status_counts["regressed"],
            "changed_unverified": status_counts["changed_unverified"],
            "missing_artifact": status_counts["missing_artifact"],
            "resolution_rate": _ratio(resolved, evaluated),
        },
        "metrics_by_failure_layer": _metrics_by_failure_layer(results),
        "results": results,
        "errors": {
            "unresolved": [
                item for item in results if item["status"] == "unresolved"
            ],
            "regressed": [
                item for item in results if item["status"] == "regressed"
            ],
            "changed_unverified": [
                item
                for item in results
                if item["status"] == "changed_unverified"
            ],
            "missing_artifact": [
                item for item in results if item["status"] == "missing_artifact"
            ],
        },
    }


def write_local_asr_regression_evaluation(
    dataset_path: Path,
    output_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, object]:
    report = evaluate_local_asr_regressions(
        dataset_path,
        artifact_root=artifact_root,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def _artifact_directory(
    sample: dict[str, object],
    run_artifacts: dict[str, Path],
    artifact_root: Path | None,
) -> Path | None:
    if artifact_root is None:
        path = Path(str(sample["artifact_directory"]))
        return path if path.is_dir() else None
    return run_artifacts.get(_source_key(str(sample["source_path"])))


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


def _stage_observation(
    artifact_path: Path,
    sample: dict[str, object],
) -> dict[str, object]:
    if not artifact_path.is_file():
        return {
            "available": False,
            "contains_observed": False,
            "contains_expected": False,
            "context_text": "",
        }
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    document = _artifact_document(artifact)
    context_text = _time_scoped_text(
        document.get("segments") or (),
        float(sample["time_range"]["start_seconds"]),
        float(sample["time_range"]["end_seconds"]),
    )
    normalized_context = _normalize_text(context_text)
    observed = _normalize_text(str(sample["target_observed"]))
    expected = _normalize_text(str(sample["target_expected"]))
    return {
        "available": True,
        "contains_observed": bool(observed and observed in normalized_context),
        "contains_expected": bool(expected and expected in normalized_context),
        "context_text": context_text,
    }


def _artifact_document(artifact: dict[str, object]) -> dict[str, object]:
    data = artifact.get("data") or {}
    if data.get("segments"):
        return data
    return ((artifact.get("context") or {}).get("document") or {})


def _time_scoped_text(
    segments: list[dict[str, object]],
    start_seconds: float,
    end_seconds: float,
) -> str:
    sentences = []
    for segment in segments:
        for sentence in segment.get("sentences") or ():
            time_range = sentence.get("time_range") or {}
            start = float(time_range.get("start_seconds", 0.0))
            end = float(time_range.get("end_seconds", 0.0))
            if end >= start_seconds - 0.5 and start <= end_seconds + 0.5:
                sentences.append((start, str(sentence.get("text") or "")))
    return "".join(text for _start, text in sorted(sentences))


def _normalize_text(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text)
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def _sample_status(
    sample: dict[str, object],
    final: dict[str, object],
) -> str:
    if not final["available"]:
        return "missing_artifact"
    if sample["status"] == "resolved_regression_guard":
        return "regression_guard_passed" if final["contains_expected"] else "regressed"
    if final["contains_observed"]:
        return "unresolved"
    if final["contains_expected"]:
        return "resolved"
    return "changed_unverified"


def _missing_result(sample: dict[str, object]) -> dict[str, object]:
    return {
        "id": sample["id"],
        "baseline_status": sample["status"],
        "failure_layer": sample["failure_layer"],
        "status": "missing_artifact",
        "artifact_directory": None,
        "target_observed": sample["target_observed"],
        "target_expected": sample["target_expected"],
        "stage_observations": {},
    }


def _metrics_by_failure_layer(
    results: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    layers: dict[str, Counter[str]] = {}
    for item in results:
        layer = str(item["failure_layer"])
        layers.setdefault(layer, Counter())[str(item["status"])] += 1
    return {
        layer: {"total": sum(counts.values()), **dict(sorted(counts.items()))}
        for layer, counts in sorted(layers.items())
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    report = write_local_asr_regression_evaluation(
        args.dataset,
        args.output,
        artifact_root=args.artifact_root,
    )
    metrics = report["metrics"]
    print(
        f"resolved={metrics['resolved']} unresolved={metrics['unresolved']} "
        f"regressed={metrics['regressed']} "
        f"changed_unverified={metrics['changed_unverified']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

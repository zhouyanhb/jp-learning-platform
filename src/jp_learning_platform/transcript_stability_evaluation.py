"""Compare repeated Whisper artifacts without changing pipeline output."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
from statistics import fmean
import unicodedata


DEFAULT_ALIGNMENT_TOLERANCE_SECONDS = 0.75
DEFAULT_HALLUCINATION_SIMILARITY_THRESHOLD = 0.35


def evaluate_transcript_stability(
    inputs: tuple[Path, ...],
    *,
    alignment_tolerance_seconds: float = DEFAULT_ALIGNMENT_TOLERANCE_SECONDS,
) -> dict[str, object]:
    artifacts = tuple(_load_artifact(path) for path in _discover_artifacts(inputs))
    grouped: dict[str, list[dict[str, object]]] = {}
    for artifact in artifacts:
        grouped.setdefault(str(artifact["source_key"]), []).append(artifact)

    sources = [
        _evaluate_source(
            source_key,
            tuple(source_artifacts),
            alignment_tolerance_seconds,
        )
        for source_key, source_artifacts in sorted(grouped.items())
        if len(source_artifacts) >= 2
    ]
    categories = Counter(
        str(region["classification"])
        for source in sources
        for region in source["regions"]
    )
    return {
        "schema_version": 1,
        "settings": {
            "alignment_tolerance_seconds": alignment_tolerance_seconds,
            "hallucination_similarity_threshold": (
                DEFAULT_HALLUCINATION_SIMILARITY_THRESHOLD
            ),
        },
        "coverage": {
            "input_artifacts": len(artifacts),
            "comparable_sources": len(sources),
            "skipped_sources": len(grouped) - len(sources),
        },
        "summary": {
            "regions": sum(len(source["regions"]) for source in sources),
            "classifications": dict(sorted(categories.items())),
        },
        "sources": sources,
    }


def write_transcript_stability_evaluation(
    inputs: tuple[Path, ...],
    output_path: Path,
) -> dict[str, object]:
    report = evaluate_transcript_stability(inputs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def _discover_artifacts(inputs: tuple[Path, ...]) -> tuple[Path, ...]:
    discovered: list[Path] = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_file():
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(sorted(path.rglob("01_whisper.json")))
        else:
            raise FileNotFoundError(path)
    unique = tuple(dict.fromkeys(item.resolve() for item in discovered))
    if len(unique) < 2:
        raise ValueError("At least two Whisper artifacts are required.")
    return unique


def _load_artifact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    document = ((payload.get("context") or {}).get("document") or {})
    source_path = str(document.get("source_path") or payload.get("source_path") or "")
    if not source_path:
        raise ValueError(f"Whisper artifact has no source_path: {path}")
    segments = tuple(
        _segment_record(segment)
        for segment in document.get("segments") or ()
        if str(segment.get("text") or "").strip()
    )
    return {
        "artifact": str(path),
        "run": str(payload.get("run_name") or path.parent.parent.name),
        "source_path": source_path,
        # Video inputs are transcribed through an extracted WAV whose basename is
        # identical for every source. The artifact's media directory remains stable
        # across runs and therefore provides the correct comparison identity.
        "source_key": path.parent.name,
        "segments": segments,
    }


def _segment_record(segment: dict[str, object]) -> dict[str, object]:
    time_range = segment["time_range"]
    confidences = tuple(
        float(word["confidence"])
        for sentence in segment.get("sentences") or ()
        for word in sentence.get("words") or ()
        if word.get("confidence") is not None
    )
    return {
        "start_seconds": float(time_range["start_seconds"]),
        "end_seconds": float(time_range["end_seconds"]),
        "text": str(segment["text"]).strip(),
        "confidence": fmean(confidences) if confidences else None,
    }


def _evaluate_source(
    source_key: str,
    artifacts: tuple[dict[str, object], ...],
    tolerance_seconds: float,
) -> dict[str, object]:
    reference = min(artifacts, key=lambda item: len(item["segments"]))
    clusters = [
        {"segments": {str(reference["artifact"]): [segment]}}
        for segment in reference["segments"]
    ]
    for artifact in artifacts:
        if artifact is reference:
            continue
        artifact_key = str(artifact["artifact"])
        for segment in artifact["segments"]:
            cluster = _best_cluster(clusters, segment, tolerance_seconds)
            if cluster is None:
                clusters.append({"segments": {artifact_key: [segment]}})
            else:
                cluster["segments"].setdefault(artifact_key, []).append(segment)

    artifact_keys = tuple(str(item["artifact"]) for item in artifacts)
    regions = [
        _evaluate_cluster(cluster, artifacts, artifact_keys)
        for cluster in sorted(clusters, key=_cluster_start)
    ]
    classifications = Counter(str(region["classification"]) for region in regions)
    return {
        "source_key": source_key,
        "source_paths": sorted({str(item["source_path"]) for item in artifacts}),
        "runs": [
            {"run": item["run"], "artifact": item["artifact"]}
            for item in artifacts
        ],
        "metrics": {
            "run_count": len(artifacts),
            "regions": len(regions),
            "stable_ratio": _ratio(classifications["stable"], len(regions)),
            "mean_character_consistency": (
                fmean(float(region["character_consistency"]) for region in regions)
                if regions
                else 0.0
            ),
            "classifications": dict(sorted(classifications.items())),
        },
        "regions": regions,
    }


def _best_cluster(
    clusters: list[dict[str, object]],
    segment: dict[str, object],
    tolerance_seconds: float,
) -> dict[str, object] | None:
    candidates = []
    for cluster in clusters:
        start, end = _cluster_bounds(cluster)
        overlap = min(end, segment["end_seconds"]) - max(
            start,
            segment["start_seconds"],
        )
        midpoint_distance = abs(
            (start + end) / 2
            - (segment["start_seconds"] + segment["end_seconds"]) / 2
        )
        if overlap > 0 or midpoint_distance <= tolerance_seconds:
            candidates.append((overlap, -midpoint_distance, cluster))
    return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def _evaluate_cluster(
    cluster: dict[str, object],
    artifacts: tuple[dict[str, object], ...],
    artifact_keys: tuple[str, ...],
) -> dict[str, object]:
    candidates = []
    normalized_texts: list[str] = []
    for artifact in artifacts:
        key = str(artifact["artifact"])
        segments = sorted(
            cluster["segments"].get(key, ()),
            key=lambda item: item["start_seconds"],
        )
        text = "".join(str(item["text"]) for item in segments)
        normalized = _normalize_text(text)
        confidences = [
            float(item["confidence"])
            for item in segments
            if item["confidence"] is not None
        ]
        normalized_texts.append(normalized)
        candidates.append(
            {
                "run": artifact["run"],
                "artifact": key,
                "text": text,
                "normalized_text": normalized,
                "mean_confidence": fmean(confidences) if confidences else None,
            }
        )

    nonempty = [text for text in normalized_texts if text]
    counts = Counter(nonempty)
    consensus = _consensus_text(nonempty)
    for candidate in candidates:
        normalized = str(candidate["normalized_text"])
        similarities = [
            _similarity(normalized, other)
            for other in nonempty
            if other != normalized or counts[normalized] > 1
        ]
        mean_similarity = fmean(similarities) if similarities else (1.0 if normalized else 0.0)
        support_count = counts[normalized] if normalized else 0
        confidence = candidate["mean_confidence"]
        candidate["support_count"] = support_count
        candidate["mean_similarity"] = round(mean_similarity, 6)
        candidate["rank_score"] = round(
            0.55 * _ratio(support_count, len(artifact_keys))
            + 0.35 * mean_similarity
            + 0.10 * (float(confidence) if confidence is not None else 0.0),
            6,
        )
    candidates.sort(key=lambda item: item["rank_score"], reverse=True)

    consistency = _pairwise_consistency(normalized_texts)
    classification = _classify_region(normalized_texts, candidates, consensus)
    start, end = _cluster_bounds(cluster)
    return {
        "time_range": {"start_seconds": start, "end_seconds": end},
        "classification": classification,
        "character_consistency": round(consistency, 6),
        "consensus_text": next(
            (item["text"] for item in candidates if item["normalized_text"] == consensus),
            "",
        ),
        "candidates": candidates,
    }


def _classify_region(
    texts: list[str],
    candidates: list[dict[str, object]],
    consensus: str,
) -> str:
    nonempty = [text for text in texts if text]
    if len(nonempty) < len(texts):
        return "possible_asr_omission"
    if len(set(nonempty)) == 1:
        return "stable"
    consensus_support = sum(text == consensus for text in nonempty)
    if len(texts) >= 3 and consensus_support >= 2:
        outliers = [
            item
            for item in candidates
            if item["normalized_text"] != consensus
            and item["support_count"] == 1
            and len(str(item["normalized_text"])) >= 4
            and float(item["mean_similarity"])
            < DEFAULT_HALLUCINATION_SIMILARITY_THRESHOLD
        ]
        if outliers:
            return "possible_hallucination"
    return "unstable_text"


def _consensus_text(texts: list[str]) -> str:
    if not texts:
        return ""
    counts = Counter(texts)
    highest_support = max(counts.values())
    supported = [text for text, count in counts.items() if count == highest_support]
    if len(supported) == 1:
        return supported[0]
    return max(
        supported,
        key=lambda text: fmean(_similarity(text, other) for other in texts),
    )


def _pairwise_consistency(texts: list[str]) -> float:
    pairs = [
        _similarity(left, right)
        for index, left in enumerate(texts)
        for right in texts[index + 1 :]
    ]
    return fmean(pairs) if pairs else 1.0


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _cluster_bounds(cluster: dict[str, object]) -> tuple[float, float]:
    segments = [
        segment
        for values in cluster["segments"].values()
        for segment in values
    ]
    return (
        min(float(item["start_seconds"]) for item in segments),
        max(float(item["end_seconds"]) for item in segments),
    )


def _cluster_start(cluster: dict[str, object]) -> float:
    return _cluster_bounds(cluster)[0]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = ArgumentParser(
        description="Compare repeated 01_whisper.json artifacts by source and timeline."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = write_transcript_stability_evaluation(tuple(args.inputs), args.output)
    print(
        f"sources={report['coverage']['comparable_sources']} "
        f"regions={report['summary']['regions']} "
        f"classifications={report['summary']['classifications']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

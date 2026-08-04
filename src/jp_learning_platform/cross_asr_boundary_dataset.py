"""Build a reviewable dataset of candidate cross-ASR language boundaries."""

from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


DEFAULT_MAX_GAP_SECONDS = 4.0
DEFAULT_MATCH_TOLERANCE_SECONDS = 0.03


def build_cross_asr_boundary_dataset(
    artifact_path: Path,
    *,
    prediction_artifact_path: Path | None = None,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> dict[str, object]:
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be non-negative.")
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    segments = artifact["context"]["document"]["segments"]
    source_path = str(artifact.get("source_path") or artifact["context"]["document"]["source_path"])
    boundaries = _candidate_boundaries(segments, max_gap_seconds)
    predictions = _prediction_decisions(prediction_artifact_path)
    samples = []
    for index, boundary in enumerate(boundaries, start=1):
        prediction = _matching_prediction(boundary, predictions)
        samples.append(
            {
                "id": f"cross-asr-{index:04d}-{_boundary_digest(source_path, boundary)}",
                **boundary,
                "predicted_label": "merge" if prediction is not None else "keep",
                "prediction": prediction,
                "gold_label": None,
                "review_status": "needs_review",
                "review_note": "",
            }
        )
    return {
        "schema_version": 1,
        "source_artifact": str(artifact_path),
        "prediction_artifact": (
            str(prediction_artifact_path) if prediction_artifact_path else None
        ),
        "source_path": source_path,
        "settings": {"max_gap_seconds": max_gap_seconds},
        "labels": {
            "merge": "Both sides belong to one language sentence.",
            "keep": "The boundary must remain between language sentences.",
            "uncertain": "Human review cannot confidently choose merge or keep.",
        },
        "samples": samples,
    }


def write_cross_asr_boundary_dataset(
    artifact_path: Path,
    output_path: Path,
    *,
    prediction_artifact_path: Path | None = None,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> dict[str, object]:
    dataset = build_cross_asr_boundary_dataset(
        artifact_path,
        prediction_artifact_path=prediction_artifact_path,
        max_gap_seconds=max_gap_seconds,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(dataset, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return dataset


def _candidate_boundaries(
    segments: list[dict[str, Any]],
    max_gap_seconds: float,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for segment_index, segment in enumerate(segments):
        sentences = segment.get("sentences") or []
        for sentence_index, sentence in enumerate(sentences):
            words = sentence.get("words") or []
            for word_index in sentence.get("asr_boundary_word_indexes") or []:
                if 0 < word_index < len(words):
                    candidates.append(
                        _boundary(
                            words[word_index - 1],
                            words[word_index],
                            "asr_token_boundary",
                            segment_index,
                            sentence_index,
                        )
                    )
            if sentence_index:
                candidates.append(
                    _sentence_boundary(
                        sentences[sentence_index - 1],
                        sentence,
                        "adjacent_sentence_boundary",
                        segment_index,
                        sentence_index,
                    )
                )
        if segment_index and sentences:
            previous_sentences = segments[segment_index - 1].get("sentences") or []
            if previous_sentences:
                candidates.append(
                    _sentence_boundary(
                        previous_sentences[-1],
                        sentences[0],
                        "segment_boundary",
                        segment_index,
                        0,
                    )
                )

    unique: dict[tuple[float, float], dict[str, object]] = {}
    for candidate in candidates:
        gap = float(candidate["gap_seconds"])
        if 0 <= gap <= max_gap_seconds:
            key = (
                round(float(candidate["left_end_seconds"]), 3),
                round(float(candidate["right_start_seconds"]), 3),
            )
            unique.setdefault(key, candidate)
    return sorted(unique.values(), key=lambda item: float(item["left_end_seconds"]))


def _sentence_boundary(
    left: dict[str, Any],
    right: dict[str, Any],
    origin: str,
    segment_index: int,
    sentence_index: int,
) -> dict[str, object]:
    left_words = left.get("words") or []
    right_words = right.get("words") or []
    left_word = left_words[-1] if left_words else {"text": left["text"], "time_range": left["time_range"]}
    right_word = right_words[0] if right_words else {"text": right["text"], "time_range": right["time_range"]}
    boundary = _boundary(left_word, right_word, origin, segment_index, sentence_index)
    boundary["left_context"] = str(left["text"])[-80:]
    boundary["right_context"] = str(right["text"])[:80]
    return boundary


def _boundary(
    left_word: dict[str, Any],
    right_word: dict[str, Any],
    origin: str,
    segment_index: int,
    sentence_index: int,
) -> dict[str, object]:
    left_end = float(left_word["time_range"]["end_seconds"])
    right_start = float(right_word["time_range"]["start_seconds"])
    return {
        "origin": origin,
        "segment_index": segment_index,
        "sentence_index": sentence_index,
        "left_end_seconds": left_end,
        "right_start_seconds": right_start,
        "gap_seconds": max(right_start - left_end, 0.0),
        "left_context": str(left_word["text"]),
        "right_context": str(right_word["text"]),
    }


def _prediction_decisions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    return list((artifact.get("data") or {}).get("cross_segment_merges") or [])


def _matching_prediction(
    boundary: dict[str, object],
    predictions: list[dict[str, Any]],
) -> dict[str, object] | None:
    for prediction in predictions:
        if (
            abs(float(prediction["left_end_seconds"]) - float(boundary["left_end_seconds"]))
            <= DEFAULT_MATCH_TOLERANCE_SECONDS
            and abs(float(prediction["right_start_seconds"]) - float(boundary["right_start_seconds"]))
            <= DEFAULT_MATCH_TOLERANCE_SECONDS
        ):
            return {
                "score": prediction["score"],
                "reason": prediction["reason"],
                "evidence": prediction["evidence"],
            }
    return None


def _boundary_digest(source_path: str, boundary: dict[str, object]) -> str:
    value = (
        f"{source_path}|{float(boundary['left_end_seconds']):.3f}|"
        f"{float(boundary['right_start_seconds']):.3f}"
    )
    return sha256(value.encode("utf-8")).hexdigest()[:8]


def main() -> int:
    parser = ArgumentParser(description="Build a cross-ASR boundary review dataset.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prediction-artifact", type=Path)
    parser.add_argument("--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS)
    args = parser.parse_args()
    dataset = write_cross_asr_boundary_dataset(
        args.artifact,
        args.output,
        prediction_artifact_path=args.prediction_artifact,
        max_gap_seconds=args.max_gap_seconds,
    )
    print(f"samples={len(dataset['samples'])}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

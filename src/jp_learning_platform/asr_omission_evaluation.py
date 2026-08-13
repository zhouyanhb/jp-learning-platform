"""Evaluate ASR omission candidates against time-aligned reference subtitles."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata


DEFAULT_COVERAGE_THRESHOLD = 0.5
DEFAULT_ALIGNMENT_PADDING_SECONDS = 0.25
DEFAULT_REGION_JOIN_GAP_SECONDS = 1.0
DEFAULT_MATCH_IOU_THRESHOLD = 0.1
DEFAULT_MIN_REFERENCE_CUE_SECONDS = 0.1
OMISSION_KINDS = frozenset(
    ("possible_asr_omission", "possible_internal_asr_omission")
)


@dataclass(frozen=True, slots=True)
class _TimedText:
    start_seconds: float
    end_seconds: float
    text: str


def evaluate_asr_omissions(
    reference_path: Path,
    artifact_path: Path,
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    alignment_padding_seconds: float = DEFAULT_ALIGNMENT_PADDING_SECONDS,
    region_join_gap_seconds: float = DEFAULT_REGION_JOIN_GAP_SECONDS,
    match_iou_threshold: float = DEFAULT_MATCH_IOU_THRESHOLD,
    min_reference_cue_seconds: float = DEFAULT_MIN_REFERENCE_CUE_SECONDS,
) -> dict[str, object]:
    raw_reference = _parse_srt(
        Path(reference_path).read_text(encoding="utf-8-sig")
    )
    reference = _prepare_reference_cues(
        raw_reference,
        min_reference_cue_seconds,
    )
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    document = ((artifact.get("context") or {}).get("document") or {})
    transcript = tuple(
        _TimedText(
            float(segment["time_range"]["start_seconds"]),
            float(segment["time_range"]["end_seconds"]),
            str(segment.get("text") or ""),
        )
        for segment in document.get("segments") or ()
    )
    predictions = tuple(
        {
            "kind": str(candidate["kind"]),
            "start_seconds": float(candidate["time_range"]["start_seconds"]),
            "end_seconds": float(candidate["time_range"]["end_seconds"]),
            "confidence": candidate.get("confidence"),
            "evidence": candidate.get("evidence") or [],
        }
        for candidate in (artifact.get("data") or {}).get("candidates") or ()
        if candidate.get("kind") in OMISSION_KINDS
    )
    cue_assessments = tuple(
        _assess_reference_cue(
            cue,
            transcript,
            coverage_threshold,
            alignment_padding_seconds,
        )
        for cue in reference
        if _normalize_text(cue.text)
    )
    gold_regions = _group_omission_regions(
        tuple(item for item in cue_assessments if item["is_omission"]),
        region_join_gap_seconds,
    )
    matches = _match_regions(predictions, gold_regions, match_iou_threshold)
    matched_predictions = {prediction for prediction, _gold, _score in matches}
    matched_gold = {gold for _prediction, gold, _score in matches}
    true_positive = len(matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(gold_regions) - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {
        "schema_version": 1,
        "reference": str(reference_path),
        "artifact": str(artifact_path),
        "label_source": "reference-derived",
        "settings": {
            "coverage_threshold": coverage_threshold,
            "alignment_padding_seconds": alignment_padding_seconds,
            "region_join_gap_seconds": region_join_gap_seconds,
            "match_iou_threshold": match_iou_threshold,
            "min_reference_cue_seconds": min_reference_cue_seconds,
            "omission_kinds": sorted(OMISSION_KINDS),
        },
        "coverage": {
            "raw_reference_cues": len(raw_reference),
            "reference_cues": len(reference),
            "assessed_cues": len(cue_assessments),
            "transcript_segments": len(transcript),
            "predicted_regions": len(predictions),
            "reference_omission_regions": len(gold_regions),
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "matches": [
            {
                "prediction": predictions[prediction],
                "reference_region": gold_regions[gold],
                "temporal_iou": score,
            }
            for prediction, gold, score in matches
        ],
        "errors": {
            "false_positive": [
                predictions[index]
                for index in range(len(predictions))
                if index not in matched_predictions
            ],
            "false_negative": [
                gold_regions[index]
                for index in range(len(gold_regions))
                if index not in matched_gold
            ],
        },
        "reference_cues": list(cue_assessments),
    }


def write_asr_omission_evaluation(
    reference_path: Path,
    artifact_path: Path,
    output_path: Path,
    **settings: float,
) -> dict[str, object]:
    report = evaluate_asr_omissions(reference_path, artifact_path, **settings)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def _parse_srt(content: str) -> tuple[_TimedText, ...]:
    cues: list[_TimedText] = []
    for block in re.split(r"\r?\n\s*\r?\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start_text, end_text = (
            part.strip() for part in lines[timing_index].split("-->", 1)
        )
        text = "".join(lines[timing_index + 1 :])
        if text:
            cues.append(
                _TimedText(
                    _parse_timestamp(start_text),
                    _parse_timestamp(end_text),
                    text,
                )
            )
    return tuple(cues)


def _parse_timestamp(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", value)
    if match is None:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    hours, minutes, seconds, milliseconds = (int(item) for item in match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _prepare_reference_cues(
    cues: tuple[_TimedText, ...],
    minimum_duration_seconds: float,
) -> tuple[_TimedText, ...]:
    prepared: list[_TimedText] = []
    for cue in cues:
        if cue.end_seconds - cue.start_seconds < minimum_duration_seconds:
            continue
        if (
            prepared
            and _normalize_text(prepared[-1].text) == _normalize_text(cue.text)
            and cue.start_seconds - prepared[-1].end_seconds <= 0.25
        ):
            previous = prepared[-1]
            prepared[-1] = _TimedText(
                previous.start_seconds,
                max(previous.end_seconds, cue.end_seconds),
                previous.text,
            )
            continue
        prepared.append(cue)
    return tuple(prepared)


def _assess_reference_cue(
    cue: _TimedText,
    transcript: tuple[_TimedText, ...],
    coverage_threshold: float,
    padding_seconds: float,
) -> dict[str, object]:
    overlapping = tuple(
        segment
        for segment in transcript
        if segment.end_seconds >= cue.start_seconds - padding_seconds
        and segment.start_seconds <= cue.end_seconds + padding_seconds
    )
    hypothesis = "".join(segment.text for segment in overlapping)
    reference_text = _normalize_text(cue.text)
    hypothesis_text = _normalize_text(hypothesis)
    matched = sum(
        block.size
        for block in SequenceMatcher(
            None,
            reference_text,
            hypothesis_text,
            autojunk=False,
        ).get_matching_blocks()
    )
    coverage = _ratio(matched, len(reference_text))
    return {
        "start_seconds": cue.start_seconds,
        "end_seconds": cue.end_seconds,
        "reference_text": cue.text,
        "hypothesis_text": hypothesis,
        "character_coverage": coverage,
        "is_omission": coverage < coverage_threshold,
    }


def _group_omission_regions(
    cues: tuple[dict[str, object], ...],
    join_gap_seconds: float,
) -> tuple[dict[str, object], ...]:
    regions: list[dict[str, object]] = []
    for cue in sorted(cues, key=lambda item: float(item["start_seconds"])):
        if (
            regions
            and float(cue["start_seconds"]) - float(regions[-1]["end_seconds"])
            <= join_gap_seconds
        ):
            regions[-1]["end_seconds"] = cue["end_seconds"]
            regions[-1]["reference_text"] = (
                f"{regions[-1]['reference_text']}{cue['reference_text']}"
            )
            regions[-1]["cue_count"] = int(regions[-1]["cue_count"]) + 1
            regions[-1]["mean_character_coverage"] = (
                float(regions[-1]["mean_character_coverage"])
                * (int(regions[-1]["cue_count"]) - 1)
                + float(cue["character_coverage"])
            ) / int(regions[-1]["cue_count"])
            continue
        regions.append(
            {
                "start_seconds": cue["start_seconds"],
                "end_seconds": cue["end_seconds"],
                "reference_text": cue["reference_text"],
                "cue_count": 1,
                "mean_character_coverage": cue["character_coverage"],
            }
        )
    return tuple(regions)


def _match_regions(
    predictions: tuple[dict[str, object], ...],
    gold_regions: tuple[dict[str, object], ...],
    threshold: float,
) -> tuple[tuple[int, int, float], ...]:
    candidates = sorted(
        (
            (_temporal_iou(prediction, gold), prediction_index, gold_index)
            for prediction_index, prediction in enumerate(predictions)
            for gold_index, gold in enumerate(gold_regions)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_gold: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, prediction_index, gold_index in candidates:
        if score < threshold:
            break
        if prediction_index in used_predictions or gold_index in used_gold:
            continue
        used_predictions.add(prediction_index)
        used_gold.add(gold_index)
        matches.append((prediction_index, gold_index, score))
    return tuple(matches)


def _temporal_iou(left: dict[str, object], right: dict[str, object]) -> float:
    start = max(float(left["start_seconds"]), float(right["start_seconds"]))
    end = min(float(left["end_seconds"]), float(right["end_seconds"]))
    overlap = max(0.0, end - start)
    union = max(float(left["end_seconds"]), float(right["end_seconds"])) - min(
        float(left["start_seconds"]), float(right["start_seconds"])
    )
    return _ratio(overlap, union)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(character for character in normalized if character.isalnum())


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = ArgumentParser(
        description="Evaluate ASR omission candidates against reference SRT."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
    )
    args = parser.parse_args()
    report = write_asr_omission_evaluation(
        args.reference,
        args.artifact,
        args.output,
        coverage_threshold=args.coverage_threshold,
    )
    metrics = report["metrics"]
    print(
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"tp={metrics['true_positive']} "
        f"fp={metrics['false_positive']} "
        f"fn={metrics['false_negative']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

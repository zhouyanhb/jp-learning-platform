"""Attribute evaluated sentence-boundary errors to actionable rule families."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import unicodedata


@dataclass(frozen=True, slots=True)
class _TimedBoundary:
    position: int
    gap_seconds: float


def attribute_boundary_errors(
    evaluation_path: Path,
    artifact_path: Path,
) -> dict[str, object]:
    """Classify false positives and negatives using recorded pipeline evidence."""
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    hypothesis_text, timed_boundaries = _hypothesis_timing(artifact)
    decisions = _decision_positions(artifact, hypothesis_text)

    false_negatives = []
    for error in evaluation["errors"]["false_negative"]:
        start, end = error["projected_hypothesis_range"]
        nearby = [
            item
            for item in timed_boundaries
            if start - 1 <= item.position <= end + 1
        ]
        max_gap = max((item.gap_seconds for item in nearby), default=None)
        category, evidence = _false_negative_category(error, max_gap, end - start)
        false_negatives.append(
            {
                **error,
                "category": category,
                "evidence": evidence,
                "max_nearby_word_gap_seconds": max_gap,
            }
        )

    false_positives = []
    for error in evaluation["errors"]["false_positive"]:
        position = error["hypothesis_position"]
        decision = _nearest_decision(decisions, position)
        category, evidence = _false_positive_category(decision)
        false_positives.append(
            {
                **error,
                "category": category,
                "evidence": evidence,
                "resolver_decision": decision,
            }
        )

    fn_counts = Counter(item["category"] for item in false_negatives)
    fp_counts = Counter(item["category"] for item in false_positives)
    return {
        "schema_version": 1,
        "evaluation": str(evaluation_path),
        "artifact": str(artifact_path),
        "summary": {
            "false_negative_categories": dict(fn_counts.most_common()),
            "false_positive_categories": dict(fp_counts.most_common()),
            "top_actionable_categories": _top_categories(fn_counts, fp_counts),
        },
        "errors": {
            "false_negative": false_negatives,
            "false_positive": false_positives,
            "unaligned_reference": evaluation["errors"]["unaligned_reference"],
        },
    }


def write_boundary_error_attribution(
    evaluation_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, object]:
    report = attribute_boundary_errors(evaluation_path, artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _hypothesis_timing(
    artifact: dict[str, object],
) -> tuple[str, tuple[_TimedBoundary, ...]]:
    segments = artifact["context"]["document"]["segments"]
    text_parts: list[str] = []
    boundaries: list[_TimedBoundary] = []
    position = 0
    previous_word: dict[str, object] | None = None
    for segment in segments:
        for sentence in segment.get("sentences", []):
            for word in sentence.get("words", []):
                text = _semantic_text(str(word.get("text", "")))
                if not text:
                    continue
                if previous_word is not None:
                    previous_end = previous_word["time_range"]["end_seconds"]
                    current_start = word["time_range"]["start_seconds"]
                    boundaries.append(
                        _TimedBoundary(position, float(current_start - previous_end))
                    )
                text_parts.append(text)
                position += len(text)
                previous_word = word
    return "".join(text_parts), tuple(boundaries)


def _decision_positions(
    artifact: dict[str, object],
    hypothesis_text: str,
) -> tuple[dict[str, object], ...]:
    positioned: list[dict[str, object]] = []
    for decision in artifact.get("data", {}).get("decisions", []):
        left = _semantic_text(str(decision.get("left_text", "")))
        right = _semantic_text(str(decision.get("right_text", "")))
        if not left or not right:
            continue
        needle = f"{left[-24:]}{right[:24]}"
        search_start = 0
        while True:
            found = hypothesis_text.find(needle, search_start)
            if found < 0:
                break
            positioned.append(
                {
                    "position": found + min(len(left), 24),
                    "reason": str(decision.get("reason", "unknown")),
                    "gap_seconds": float(decision.get("gap_seconds", 0.0)),
                }
            )
            search_start = found + 1
    return tuple(positioned)


def _nearest_decision(
    decisions: tuple[dict[str, object], ...],
    position: int,
) -> dict[str, object] | None:
    candidates = [
        decision
        for decision in decisions
        if abs(int(decision["position"]) - position) <= 1
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(int(item["position"]) - position))


def _false_negative_category(
    error: dict[str, object],
    max_gap: float | None,
    projection_width: int,
) -> tuple[str, str]:
    role = str(error.get("role", "dialogue"))
    types = set(error.get("boundary_types", []))
    dimensions = set(error.get("boundary_dimensions", ("language_sentence",)))
    if projection_width > 8:
        return "alignment_uncertain", "Projected boundary spans more than 8 characters."
    if "speaker_turn" in types:
        return "speaker_turn_missing", "Reference boundary includes a speaker turn."
    if role == "question":
        return "question_boundary_missing", "Reference sentence is an exam question."
    if max_gap is None:
        return "token_alignment_missing", "No aligned token boundary is nearby."
    if max_gap >= 1.5:
        return "strong_pause_suppressed", f"Nearby token gap is {max_gap:.3f}s."
    if max_gap >= 0.5:
        return "pause_boundary_suppressed", f"Nearby token gap is {max_gap:.3f}s."
    dimension_text = ",".join(sorted(dimensions))
    return (
        "short_pause_semantic_boundary",
        f"Largest nearby gap is {max_gap:.3f}s; dimensions={dimension_text}.",
    )


def _false_positive_category(
    decision: dict[str, object] | None,
) -> tuple[str, str]:
    if decision is None:
        return (
            "upstream_boundary_preserved",
            "Boundary was present before the resolver or has no recorded decision.",
        )
    reason = str(decision["reason"])
    if reason == "strong_pause":
        category = "strong_pause_over_split"
    elif reason in {"pause_after_sentence_final", "terminal_mark"}:
        category = "sentence_final_over_split"
    elif "question_answer" in reason or "response_transition" in reason:
        category = "transition_rule_over_split"
    elif "structural" in reason or "numbering" in reason:
        category = "structure_rule_over_split"
    else:
        category = "resolver_rule_over_split"
    return category, f"Resolver accepted reason {reason}."


def _top_categories(
    false_negatives: Counter[str],
    false_positives: Counter[str],
) -> list[dict[str, object]]:
    combined = [
        {"error_type": "false_negative", "category": category, "count": count}
        for category, count in false_negatives.items()
        if category != "alignment_uncertain"
    ]
    combined.extend(
        {"error_type": "false_positive", "category": category, "count": count}
        for category, count in false_positives.items()
    )
    return sorted(combined, key=lambda item: (-int(item["count"]), str(item["category"])))[:5]


def _semantic_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Attribute sentence-boundary errors.")
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = write_boundary_error_attribution(
        args.evaluation,
        args.artifact,
        args.output,
    )
    for item in report["summary"]["top_actionable_categories"]:
        print(f"{item['error_type']} {item['category']}: {item['count']}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

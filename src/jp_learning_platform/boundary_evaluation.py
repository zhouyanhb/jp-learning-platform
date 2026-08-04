"""Evaluate pipeline sentence boundaries against a reviewed reference dataset."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import unicodedata


DEFAULT_MAX_ANCHOR_DISTANCE = 12


@dataclass(frozen=True, slots=True)
class _ReferenceBoundary:
    position: int
    sample_id: str
    role: str
    types: tuple[str, ...]
    dimensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BoundaryWindow:
    boundary: _ReferenceBoundary
    start: int
    end: int


def evaluate_boundary_artifact(
    dataset_path: Path,
    artifact_path: Path,
    *,
    max_anchor_distance: int = DEFAULT_MAX_ANCHOR_DISTANCE,
) -> dict[str, object]:
    """Compare linguistic sentence boundaries after aligning normalized text."""
    if max_anchor_distance < 1:
        raise ValueError("max_anchor_distance must be at least 1.")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    reference_text, reference_boundaries = _reference_stream(dataset)
    hypothesis_text, hypothesis_boundaries = _hypothesis_stream(artifact)
    reference_to_hypothesis = _aligned_character_map(reference_text, hypothesis_text)
    hypothesis_to_reference = {
        hypothesis: reference
        for reference, hypothesis in reference_to_hypothesis.items()
    }

    language_boundaries = tuple(
        boundary
        for boundary in reference_boundaries
        if "language_sentence" in boundary.dimensions
    )
    windows: list[_BoundaryWindow] = []
    unaligned_reference: list[_ReferenceBoundary] = []
    for boundary in language_boundaries:
        window = _project_reference_boundary(
            boundary,
            reference_to_hypothesis,
            max_anchor_distance,
        )
        if window is None:
            unaligned_reference.append(boundary)
        else:
            windows.append(window)

    structure_only_windows = tuple(
        window
        for boundary in reference_boundaries
        if "content_structure" in boundary.dimensions
        and "language_sentence" not in boundary.dimensions
        if (
            window := _project_reference_boundary(
                boundary,
                reference_to_hypothesis,
                max_anchor_distance,
            )
        )
        is not None
    )
    evaluable_hypothesis = {
        position
        for position in hypothesis_boundaries
        if _is_hypothesis_boundary_evaluable(
            position,
            hypothesis_to_reference,
            max_anchor_distance,
        )
        and not any(
            window.start <= position <= window.end
            for window in structure_only_windows
        )
    }
    matches = _match_boundaries(windows, evaluable_hypothesis)
    matched_reference = {index for index, _position in matches}
    matched_hypothesis = {position for _index, position in matches}
    false_negative_windows = [
        window for index, window in enumerate(windows) if index not in matched_reference
    ]
    false_positive_positions = sorted(evaluable_hypothesis - matched_hypothesis)

    true_positive = len(matches)
    false_positive = len(false_positive_positions)
    false_negative = len(false_negative_windows)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)

    return {
        "schema_version": 2,
        "dataset": str(dataset_path),
        "artifact": str(artifact_path),
        "settings": {"max_anchor_distance": max_anchor_distance},
        "alignment": {
            "reference_characters": len(reference_text),
            "hypothesis_characters": len(hypothesis_text),
            "matched_characters": len(reference_to_hypothesis),
            "reference_character_coverage": _ratio(
                len(reference_to_hypothesis), len(reference_text)
            ),
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "unaligned_reference": len(unaligned_reference),
            "ignored_hypothesis": len(hypothesis_boundaries)
            - len(evaluable_hypothesis),
        },
        "dimensions": {
            "language_sentence": {
                "status": "evaluated",
                "reference_boundaries": len(language_boundaries),
                "metrics": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                },
            },
            "speaker_turn": _reference_dimension_summary(
                reference_boundaries,
                "speaker_turn",
                reference_to_hypothesis,
                max_anchor_distance,
            ),
            "content_structure": _reference_dimension_summary(
                reference_boundaries,
                "content_structure",
                reference_to_hypothesis,
                max_anchor_distance,
            ),
        },
        "errors": {
            "false_negative": [
                {
                    "sample_id": window.boundary.sample_id,
                    "role": window.boundary.role,
                    "boundary_types": list(window.boundary.types),
                    "boundary_dimensions": list(window.boundary.dimensions),
                    "reference_position": window.boundary.position,
                    "reference_context": _boundary_context(
                        reference_text, window.boundary.position
                    ),
                    "projected_hypothesis_range": [window.start, window.end],
                    "hypothesis_context": _range_context(
                        hypothesis_text, window.start, window.end
                    ),
                }
                for window in false_negative_windows
            ],
            "false_positive": [
                {
                    "hypothesis_position": position,
                    "hypothesis_context": _boundary_context(
                        hypothesis_text, position
                    ),
                    "nearest_reference_position": _nearest_reference_position(
                        position, hypothesis_to_reference
                    ),
                }
                for position in false_positive_positions
            ],
            "unaligned_reference": [
                {
                    "sample_id": boundary.sample_id,
                    "role": boundary.role,
                    "boundary_types": list(boundary.types),
                    "boundary_dimensions": list(boundary.dimensions),
                    "reference_position": boundary.position,
                    "reference_context": _boundary_context(
                        reference_text, boundary.position
                    ),
                }
                for boundary in unaligned_reference
            ],
        },
    }


def write_boundary_evaluation(
    dataset_path: Path,
    artifact_path: Path,
    output_path: Path,
    *,
    max_anchor_distance: int = DEFAULT_MAX_ANCHOR_DISTANCE,
) -> dict[str, object]:
    report = evaluate_boundary_artifact(
        dataset_path,
        artifact_path,
        max_anchor_distance=max_anchor_distance,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _reference_stream(
    dataset: dict[str, object],
) -> tuple[str, tuple[_ReferenceBoundary, ...]]:
    text_parts: list[str] = []
    boundaries: list[_ReferenceBoundary] = []
    position = 0
    samples = dataset.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Boundary dataset must contain a samples list.")

    sentences_seen = 0
    flattened: list[
        tuple[str, str, str, tuple[str, ...], tuple[str, ...]]
    ] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each boundary sample must be an object.")
        sample_id = str(sample.get("id", ""))
        sentences = sample.get("sentences")
        if not isinstance(sentences, list):
            raise ValueError(f"Sample {sample_id!r} must contain sentences.")
        sample_boundaries = sample.get("boundaries", [])
        boundary_types = {
            int(boundary["after_char"]): tuple(str(item) for item in boundary["types"])
            for boundary in sample_boundaries
            if isinstance(boundary, dict)
            and isinstance(boundary.get("after_char"), int)
            and isinstance(boundary.get("types"), list)
        }
        boundary_dimensions = {
            int(boundary["after_char"]): tuple(
                str(item) for item in boundary["dimensions"]
            )
            for boundary in sample_boundaries
            if isinstance(boundary, dict)
            and isinstance(boundary.get("after_char"), int)
            and isinstance(boundary.get("dimensions"), list)
        }
        has_explicit_dimensions = bool(boundary_dimensions)
        sample_offset = 0
        for sentence in sentences:
            if not isinstance(sentence, dict):
                raise ValueError(f"Sample {sample_id!r} contains an invalid sentence.")
            normalized = _semantic_text(str(sentence.get("text", "")))
            if normalized:
                sample_offset += len(normalized)
                types = boundary_types.get(sample_offset, ("sample_transition",))
                dimensions = boundary_dimensions.get(sample_offset)
                if dimensions is None:
                    dimensions = (
                        _inferred_dimensions(
                            str(
                                sentence.get(
                                    "source_text",
                                    sentence.get("text", ""),
                                )
                            ),
                            types,
                        )
                        if has_explicit_dimensions
                        else ("language_sentence",)
                    )
                flattened.append(
                    (
                        normalized,
                        sample_id,
                        str(sentence.get("role", "dialogue")),
                        types,
                        dimensions,
                    )
                )

    for index, (text, sample_id, role, types, dimensions) in enumerate(flattened):
        text_parts.append(text)
        position += len(text)
        sentences_seen += 1
        if index < len(flattened) - 1:
            boundaries.append(
                _ReferenceBoundary(position, sample_id, role, types, dimensions)
            )
    if not sentences_seen:
        raise ValueError("Boundary dataset contains no sentence text.")
    return "".join(text_parts), tuple(boundaries)


def _hypothesis_stream(
    artifact: dict[str, object],
) -> tuple[str, frozenset[int]]:
    try:
        segments = artifact["context"]["document"]["segments"]
    except (KeyError, TypeError) as error:
        raise ValueError("Artifact does not contain context.document.segments.") from error
    if not isinstance(segments, list):
        raise ValueError("Artifact segments must be a list.")

    texts: list[str] = []
    for segment in segments:
        sentences = segment.get("sentences", []) if isinstance(segment, dict) else []
        if not isinstance(sentences, list):
            raise ValueError("Artifact sentence collection must be a list.")
        for sentence in sentences:
            if not isinstance(sentence, dict):
                raise ValueError("Artifact contains an invalid sentence.")
            normalized = _semantic_text(str(sentence.get("text", "")))
            if normalized:
                texts.append(normalized)
    if not texts:
        raise ValueError("Artifact contains no sentence text.")

    boundaries: set[int] = set()
    position = 0
    for index, text in enumerate(texts):
        position += len(text)
        if index < len(texts) - 1:
            boundaries.add(position)
    return "".join(texts), frozenset(boundaries)


def _semantic_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def _aligned_character_map(reference: str, hypothesis: str) -> dict[int, int]:
    matcher = SequenceMatcher(None, reference, hypothesis, autojunk=False)
    mapping: dict[int, int] = {}
    for reference_start, hypothesis_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            mapping[reference_start + offset] = hypothesis_start + offset
    return mapping


def _project_reference_boundary(
    boundary: _ReferenceBoundary,
    mapping: dict[int, int],
    max_distance: int,
) -> _BoundaryWindow | None:
    left = next(
        (index for index in range(boundary.position - 1, max(-1, boundary.position - max_distance - 1), -1) if index in mapping),
        None,
    )
    right = next(
        (index for index in range(boundary.position, boundary.position + max_distance) if index in mapping),
        None,
    )
    if left is None or right is None or mapping[left] >= mapping[right]:
        return None
    return _BoundaryWindow(boundary, mapping[left] + 1, mapping[right])


def _is_hypothesis_boundary_evaluable(
    position: int,
    mapping: dict[int, int],
    max_distance: int,
) -> bool:
    left = next(
        (index for index in range(position - 1, max(-1, position - max_distance - 1), -1) if index in mapping),
        None,
    )
    right = next(
        (index for index in range(position, position + max_distance) if index in mapping),
        None,
    )
    return bool(
        left is not None
        and right is not None
        and mapping[left] < mapping[right]
    )


def _match_boundaries(
    windows: list[_BoundaryWindow],
    hypothesis_positions: set[int],
) -> tuple[tuple[int, int], ...]:
    candidates = sorted(
        (
            abs(position - ((window.start + window.end) / 2)),
            index,
            position,
        )
        for index, window in enumerate(windows)
        for position in hypothesis_positions
        if window.start <= position <= window.end
    )
    used_references: set[int] = set()
    used_hypotheses: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _distance, reference_index, hypothesis_position in candidates:
        if (
            reference_index in used_references
            or hypothesis_position in used_hypotheses
        ):
            continue
        used_references.add(reference_index)
        used_hypotheses.add(hypothesis_position)
        matches.append((reference_index, hypothesis_position))
    return tuple(matches)


def _nearest_reference_position(
    hypothesis_position: int,
    mapping: dict[int, int],
) -> int | None:
    if not mapping:
        return None
    nearest = min(mapping, key=lambda position: abs(position - hypothesis_position))
    return mapping[nearest]


def _boundary_context(text: str, position: int, radius: int = 24) -> str:
    start = max(position - radius, 0)
    end = min(position + radius, len(text))
    return f"{text[start:position]}|{text[position:end]}"


def _range_context(text: str, start: int, end: int, radius: int = 24) -> str:
    context_start = max(start - radius, 0)
    context_end = min(end + radius, len(text))
    return f"{text[context_start:start]}|{text[start:end]}|{text[end:context_end]}"


def _inferred_dimensions(
    source_text: str,
    boundary_types: tuple[str, ...],
) -> tuple[str, ...]:
    dimensions: list[str] = []
    types = set(boundary_types)
    stripped = source_text.rstrip()
    if (
        types.intersection({"terminal_punctuation", "speaker_turn"})
        or stripped.endswith(("。", "?", "？", "!", "！"))
    ):
        dimensions.append("language_sentence")
    if "speaker_turn" in types:
        dimensions.append("speaker_turn")
    if "sample_transition" in types:
        dimensions.append("content_structure")
    return tuple(dimensions or ("language_sentence",))


def _reference_dimension_summary(
    boundaries: tuple[_ReferenceBoundary, ...],
    dimension: str,
    mapping: dict[int, int],
    max_distance: int,
) -> dict[str, object]:
    selected = tuple(
        boundary for boundary in boundaries if dimension in boundary.dimensions
    )
    aligned = sum(
        _project_reference_boundary(boundary, mapping, max_distance) is not None
        for boundary in selected
    )
    return {
        "status": "reference_only",
        "reference_boundaries": len(selected),
        "aligned_reference_boundaries": aligned,
        "note": "No structure-layer hypothesis is evaluated at this stage.",
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate Japanese sentence boundaries.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--max-anchor-distance",
        type=int,
        default=DEFAULT_MAX_ANCHOR_DISTANCE,
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = write_boundary_evaluation(
        args.dataset,
        args.artifact,
        args.output,
        max_anchor_distance=args.max_anchor_distance,
    )
    metrics = report["metrics"]
    alignment = report["alignment"]
    print(
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"coverage={alignment['reference_character_coverage']:.4f}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

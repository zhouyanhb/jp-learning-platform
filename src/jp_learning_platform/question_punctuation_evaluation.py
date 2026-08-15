"""Evaluate conservative Japanese question-mark restoration against subtitles."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from statistics import median
import unicodedata


DEFAULT_MIN_TEMPORAL_IOU = 0.05
DEFAULT_MIN_TEXT_SIMILARITY = 0.15
DEFAULT_MIN_TERMINAL_TEXT_SIMILARITY = 0.75
DEFAULT_ALIGNMENT_PADDING_SECONDS = 0.25
DEFAULT_PRECISION_TARGET = 0.95

_CANDIDATE_SCOPE_ATTRIBUTION = "candidate_rule_missing"
_LANGUAGE_SENTENCE_ATTRIBUTION = "language_sentence_error"
_EXCLUDED_MATCH_ATTRIBUTIONS = frozenset(
    {
        "already_visible_question_mark",
        "reference_overlap_duplicate",
        "asr_error",
    }
)
_EXCLUDED_FALSE_NEGATIVE_ATTRIBUTIONS = frozenset(
    {
        "already_visible_question_mark",
        "reference_overlap_duplicate",
        "asr_error",
        "asr_omission_or_misalignment",
    }
)


@dataclass(frozen=True, slots=True)
class _TimedQuestion:
    start_seconds: float
    end_seconds: float
    text: str
    is_question: bool
    segment_position: int | None = None
    sentence_index: int | None = None
    confidence: float | None = None
    evidence: tuple[str, ...] = ()


def evaluate_question_punctuation(
    reference_path: Path,
    artifact_path: Path,
    *,
    min_temporal_iou: float = DEFAULT_MIN_TEMPORAL_IOU,
    min_text_similarity: float = DEFAULT_MIN_TEXT_SIMILARITY,
    min_terminal_text_similarity: float = DEFAULT_MIN_TERMINAL_TEXT_SIMILARITY,
    alignment_padding_seconds: float = DEFAULT_ALIGNMENT_PADDING_SECONDS,
    precision_target: float = DEFAULT_PRECISION_TARGET,
    reference_time_offset_seconds: float | None = None,
    annotation_path: Path | None = None,
) -> dict[str, object]:
    prepared_reference = _prepare_reference_cues(
        _parse_srt(Path(reference_path).read_text(encoding="utf-8-sig"))
    )
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    sentences = _artifact_sentences(artifact)
    alignment_anchors = (
        _reference_alignment_anchors(prepared_reference, sentences)
        if reference_time_offset_seconds is None
        else ()
    )
    reference = _deduplicate_reference_questions(tuple(
        _shift_time(
            item,
            reference_time_offset_seconds
            if reference_time_offset_seconds is not None
            else _local_reference_offset(item, alignment_anchors),
        )
        for cue in prepared_reference
        for item in _reference_questions_in_cue(cue)
    ))
    predictions, prediction_source = _artifact_predictions(artifact, sentences)
    raw_matches = _match_questions(
        predictions,
        reference,
        min_temporal_iou,
        min_text_similarity,
        min_terminal_text_similarity,
        alignment_padding_seconds,
    )
    raw_true_positive = len(raw_matches)
    raw_metrics = _metrics(
        raw_true_positive,
        len(predictions) - raw_true_positive,
        len(reference) - raw_true_positive,
        precision_target,
    )
    reviewed_attributions = _reviewed_reference_attributions(annotation_path)
    excluded_reference_indexes = {
        index
        for index, item in enumerate(reference)
        if reviewed_attributions.get(_annotation_reference_key(_as_dict(item)))
        in _EXCLUDED_MATCH_ATTRIBUTIONS
    }
    active_reference = tuple(
        item
        for index, item in enumerate(reference)
        if index not in excluded_reference_indexes
    )
    active_matches = _match_questions(
        predictions,
        active_reference,
        min_temporal_iou,
        min_text_similarity,
        min_terminal_text_similarity,
        alignment_padding_seconds,
    )
    active_matched_predictions = {item[0] for item in active_matches}
    excluded_reference = tuple(
        reference[index] for index in sorted(excluded_reference_indexes)
    )
    excluded_matches = _match_questions(
        predictions,
        excluded_reference,
        min_temporal_iou,
        min_text_similarity,
        min_terminal_text_similarity,
        alignment_padding_seconds,
    )
    ignored_prediction_indexes = {
        item[0]
        for item in excluded_matches
        if item[0] not in active_matched_predictions
    }
    eligible_prediction_indexes = tuple(
        index
        for index in range(len(predictions))
        if index not in ignored_prediction_indexes
    )
    eligible_predictions = tuple(predictions[index] for index in eligible_prediction_indexes)
    prediction_index = {
        original: scoped for scoped, original in enumerate(eligible_prediction_indexes)
    }
    matches = tuple(
        (prediction_index[prediction], gold, temporal_iou, text_similarity, terminal_text_similarity)
        for prediction, gold, temporal_iou, text_similarity, terminal_text_similarity in active_matches
        if prediction in prediction_index
    )
    matched_predictions = {item[0] for item in matches}
    matched_reference = {item[1] for item in matches}
    true_positive = len(matches)
    false_positive = len(eligible_predictions) - true_positive
    false_negative = len(active_reference) - true_positive
    false_negative_details = [
        {
            **_as_dict(active_reference[index]),
            "overlapping_language_sentences": [
                _as_dict(sentence)
                for sentence in sentences
                if _overlap_seconds(sentence, active_reference[index]) > 0
            ],
        }
        for index in range(len(active_reference))
        if index not in matched_reference
    ]
    attribution = _false_negative_attribution(
        false_negative_details,
        annotation_path,
        excluded_reference,
        reviewed_attributions,
    )
    scoped_false_negative = (
        attribution["candidate_rule_missing"]
        if attribution is not None
        else false_negative
    )
    metrics = _metrics(
        true_positive,
        false_positive,
        scoped_false_negative,
        precision_target,
    )
    metrics_by_candidate_type = _metrics_by_candidate_type(
        eligible_predictions,
        matches,
        len(active_reference),
        precision_target,
    )
    return {
        "schema_version": 2,
        "reference": str(reference_path),
        "artifact": str(artifact_path),
        "label_source": "reference-question-punctuation",
        "prediction_source": prediction_source,
        "settings": {
            "min_temporal_iou": min_temporal_iou,
            "min_text_similarity": min_text_similarity,
            "min_terminal_text_similarity": min_terminal_text_similarity,
            "alignment_padding_seconds": alignment_padding_seconds,
            "precision_target": precision_target,
            "reference_time_offset_seconds": reference_time_offset_seconds,
            "annotation_path": str(annotation_path) if annotation_path else None,
            "time_alignment": (
                "explicit_global_offset"
                if reference_time_offset_seconds is not None
                else "local_text_anchors"
            ),
            "time_alignment_anchor_count": len(alignment_anchors),
        },
        "coverage": {
            "reference_questions": len(reference),
            "evaluated_reference_questions": len(active_reference),
            "excluded_reference_questions": len(excluded_reference),
            "language_sentences": len(sentences),
            "predicted_questions": len(predictions),
            "evaluated_predicted_questions": len(eligible_predictions),
            "ignored_predictions": len(ignored_prediction_indexes),
        },
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "evaluation_scope": (
            "reviewed_candidate_rule"
            if attribution is not None
            else "all_reference_questions"
        ),
        "error_attribution": attribution,
        "metrics_by_candidate_type": metrics_by_candidate_type,
        "matches": [
            {
                "prediction": _as_dict(eligible_predictions[prediction]),
                "reference": _as_dict(active_reference[gold]),
                "temporal_iou": temporal_iou,
                "text_similarity": text_similarity,
                "terminal_text_similarity": terminal_text_similarity,
            }
            for prediction, gold, temporal_iou, text_similarity, terminal_text_similarity in matches
        ],
        "errors": {
            "false_positive": [
                _as_dict(eligible_predictions[index])
                for index in range(len(eligible_predictions))
                if index not in matched_predictions
            ],
            "false_negative": false_negative_details,
            "ignored_prediction": [
                _as_dict(predictions[index])
                for index in sorted(ignored_prediction_indexes)
            ],
        },
    }


def _metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    precision_target: float,
) -> dict[str, object]:
    precision = _optional_ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        _ratio(2 * precision * recall, precision + recall)
        if precision is not None
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision_target_met": (
            precision >= precision_target if precision is not None else None
        ),
    }


def _false_negative_attribution(
    false_negatives: list[dict[str, object]],
    annotation_path: Path | None,
    excluded_references: tuple[_TimedQuestion, ...] = (),
    reviewed_attributions: dict[tuple[float, float, str], str] | None = None,
) -> dict[str, object] | None:
    if annotation_path is None:
        return None
    reviewed = (
        reviewed_attributions
        if reviewed_attributions is not None
        else _reviewed_reference_attributions(annotation_path)
    )
    details: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for item in false_negatives:
        reason = reviewed.get(
            _annotation_reference_key(item),
            "unreviewed",
        )
        counts[reason] = counts.get(reason, 0) + 1
        details.append(
            {
                "reference": {
                    key: value
                    for key, value in item.items()
                    if key != "overlapping_language_sentences"
                },
                "error_attribution": reason,
                "evaluation_disposition": _attribution_disposition(reason),
            }
        )
    for item in excluded_references:
        item_dict = _as_dict(item)
        reason = reviewed.get(
            _annotation_reference_key(item_dict),
            "reviewed_without_attribution",
        )
        counts[reason] = counts.get(reason, 0) + 1
        details.append(
            {
                "reference": item_dict,
                "error_attribution": reason,
                "evaluation_disposition": "excluded_before_matching",
            }
        )
    excluded = sum(
        count
        for reason, count in counts.items()
        if reason in _EXCLUDED_FALSE_NEGATIVE_ATTRIBUTIONS
    )
    return {
        "candidate_rule_missing": counts.get(_CANDIDATE_SCOPE_ATTRIBUTION, 0),
        "language_sentence_error": counts.get(_LANGUAGE_SENTENCE_ATTRIBUTION, 0),
        "excluded_false_negative": excluded,
        "unreviewed": counts.get("unreviewed", 0),
        "counts_by_reason": dict(sorted(counts.items())),
        "details": details,
    }


def _reviewed_reference_attributions(
    annotation_path: Path | None,
) -> dict[tuple[float, float, str], str]:
    if annotation_path is None:
        return {}
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    return {
        _annotation_reference_key(sample["reference"]): str(
            sample.get("error_attribution") or "reviewed_without_attribution"
        )
        for sample in annotation.get("samples") or ()
        if sample.get("sample_kind") == "missed_reference"
        and sample.get("review_status") == "reviewed"
        and isinstance(sample.get("reference"), dict)
    }


def _annotation_reference_key(item: dict[str, object]) -> tuple[float, float, str]:
    return (
        round(float(item["start_seconds"]), 4),
        round(float(item["end_seconds"]), 4),
        _normalize_text(str(item.get("text") or "")),
    )


def _attribution_disposition(reason: str) -> str:
    if reason == _CANDIDATE_SCOPE_ATTRIBUTION:
        return "candidate_evaluation_false_negative"
    if reason == _LANGUAGE_SENTENCE_ATTRIBUTION:
        return "language_sentence_evaluation"
    if reason in _EXCLUDED_FALSE_NEGATIVE_ATTRIBUTIONS:
        return "excluded"
    return "pending_review"


def write_question_punctuation_evaluation(
    reference_path: Path,
    artifact_path: Path,
    output_path: Path,
    **settings: float,
) -> dict[str, object]:
    report = evaluate_question_punctuation(reference_path, artifact_path, **settings)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def _parse_srt(content: str) -> tuple[_TimedQuestion, ...]:
    cues: list[_TimedQuestion] = []
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
                _TimedQuestion(
                    _parse_timestamp(start_text),
                    _parse_timestamp(end_text),
                    text,
                    _contains_question_mark(text),
                )
            )
    return tuple(cues)


def _reference_questions_in_cue(
    cue: _TimedQuestion,
) -> tuple[_TimedQuestion, ...]:
    question_mark_indexes = tuple(
        index for index, character in enumerate(cue.text) if character in "?？"
    )
    if not question_mark_indexes:
        return ()
    questions: list[_TimedQuestion] = []
    clause_start = 0
    cue_duration = cue.end_seconds - cue.start_seconds
    text_length = len(cue.text)
    for question_mark_index in question_mark_indexes:
        text = cue.text[clause_start : question_mark_index + 1].strip()
        if text:
            question_start = cue.start_seconds + cue_duration * (
                clause_start / text_length
            )
            question_end = cue.start_seconds + cue_duration * (
                (question_mark_index + 1) / text_length
            )
            questions.append(
                _TimedQuestion(
                    question_start,
                    question_end,
                    text,
                    True,
                )
            )
        clause_start = question_mark_index + 1
    return tuple(questions)


def _artifact_sentences(artifact: dict[str, object]) -> tuple[_TimedQuestion, ...]:
    document = ((artifact.get("context") or {}).get("document") or {})
    results: list[_TimedQuestion] = []
    for segment in document.get("segments") or ():
        for sentence_index, sentence in enumerate(segment.get("sentences") or ()):
            time_range = sentence["time_range"]
            text = str(sentence.get("text") or "")
            results.append(
                _TimedQuestion(
                    float(time_range["start_seconds"]),
                    float(time_range["end_seconds"]),
                    text,
                    _contains_question_mark(text),
                    int(segment.get("position", 0)),
                    sentence_index,
                )
            )
    return tuple(results)


def _artifact_predictions(
    artifact: dict[str, object],
    sentences: tuple[_TimedQuestion, ...],
) -> tuple[tuple[_TimedQuestion, ...], str]:
    data = artifact.get("data") or {}
    if "candidates" not in data:
        return tuple(item for item in sentences if item.is_question), "visible_question_marks"
    candidates = tuple(
        _TimedQuestion(
            float(item["time_range"]["start_seconds"]),
            float(item["time_range"]["end_seconds"]),
            str(item.get("text") or ""),
            True,
            int(item["segment_position"]),
            int(item["sentence_index"]),
            (
                float(item["confidence"])
                if item.get("confidence") is not None
                else None
            ),
            tuple(str(value) for value in item.get("evidence") or ()),
        )
        for item in data.get("candidates") or ()
    )
    return candidates, "question_punctuation_candidates"


def _deduplicate_reference_questions(
    questions: tuple[_TimedQuestion, ...],
) -> tuple[_TimedQuestion, ...]:
    deduplicated: list[_TimedQuestion] = []
    for question in questions:
        signature = _question_signature(question.text)
        duplicate_index = next(
            (
                index
                for index in range(len(deduplicated) - 1, max(-1, len(deduplicated) - 6), -1)
                if _overlap_seconds(deduplicated[index], question) > 0
                and abs(_center_seconds(deduplicated[index]) - _center_seconds(question))
                <= 6.0
                and SequenceMatcher(
                    None,
                    _question_signature(deduplicated[index].text),
                    signature,
                    autojunk=False,
                ).ratio()
                >= 0.6
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(question)
            continue
        previous = deduplicated[duplicate_index]
        deduplicated[duplicate_index] = _TimedQuestion(
            min(previous.start_seconds, question.start_seconds),
            max(previous.end_seconds, question.end_seconds),
            question.text if len(question.text) > len(previous.text) else previous.text,
            True,
        )
    return tuple(deduplicated)


def _question_signature(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    question_index = max(normalized.rfind("?"), normalized.rfind("？"))
    if question_index < 0:
        return _normalize_text(normalized)
    return _normalize_text(normalized[max(0, question_index - 30) : question_index])


def _prepare_reference_cues(
    cues: tuple[_TimedQuestion, ...],
) -> tuple[_TimedQuestion, ...]:
    prepared: list[_TimedQuestion] = []
    for cue in cues:
        if (
            prepared
            and _normalize_text(prepared[-1].text) == _normalize_text(cue.text)
            and cue.start_seconds - prepared[-1].end_seconds <= 0.25
        ):
            previous = prepared[-1]
            prepared[-1] = _TimedQuestion(
                previous.start_seconds,
                max(previous.end_seconds, cue.end_seconds),
                previous.text,
                previous.is_question or cue.is_question,
            )
            continue
        prepared.append(cue)
    return tuple(prepared)


def _reference_alignment_anchors(
    reference: tuple[_TimedQuestion, ...],
    sentences: tuple[_TimedQuestion, ...],
) -> tuple[tuple[float, float], ...]:
    anchors: list[tuple[float, float]] = []
    for gold in reference:
        normalized_gold = _normalize_text(gold.text)
        if len(normalized_gold) < 6:
            continue
        best: tuple[float, _TimedQuestion] | None = None
        for sentence in sentences:
            center_distance = abs(_center_seconds(gold) - _center_seconds(sentence))
            if center_distance > 30.0:
                continue
            similarity = _text_similarity(gold.text, sentence.text)
            if best is None or similarity > best[0]:
                best = (similarity, sentence)
        if best is None or best[0] < 0.65:
            continue
        sentence = best[1]
        anchors.append(
            (
                _center_seconds(gold),
                _center_seconds(sentence) - _center_seconds(gold),
            )
        )
    return tuple(anchors)


def _local_reference_offset(
    item: _TimedQuestion,
    anchors: tuple[tuple[float, float], ...],
) -> float:
    if not anchors:
        return 0.0
    center = _center_seconds(item)
    nearest = sorted(anchors, key=lambda anchor: abs(anchor[0] - center))[:5]
    return median(anchor[1] for anchor in nearest)


def _center_seconds(item: _TimedQuestion) -> float:
    return (item.start_seconds + item.end_seconds) / 2


def _shift_time(item: _TimedQuestion, offset_seconds: float) -> _TimedQuestion:
    return _TimedQuestion(
        item.start_seconds + offset_seconds,
        item.end_seconds + offset_seconds,
        item.text,
        item.is_question,
        item.segment_position,
        item.sentence_index,
        item.confidence,
        item.evidence,
    )


def _match_questions(
    predictions: tuple[_TimedQuestion, ...],
    reference: tuple[_TimedQuestion, ...],
    min_temporal_iou: float,
    min_text_similarity: float,
    min_terminal_text_similarity: float,
    padding_seconds: float,
) -> tuple[tuple[int, int, float, float, float], ...]:
    candidates: list[tuple[float, int, int, float, float, float]] = []
    for prediction_index, prediction in enumerate(predictions):
        for reference_index, gold in enumerate(reference):
            temporal_iou = _temporal_match_score(
                prediction,
                gold,
                padding_seconds,
            )
            text_similarity = _text_similarity(prediction.text, gold.text)
            terminal_text_similarity = _question_endpoint_similarity(
                prediction,
                gold,
            )
            terminal_clause_alignment = _matches_reference_terminal_clause(
                prediction,
                gold,
                padding_seconds,
            )
            if (
                (
                    temporal_iou >= min_temporal_iou
                    or terminal_clause_alignment
                )
                and text_similarity >= min_text_similarity
                and (
                    terminal_text_similarity >= min_terminal_text_similarity
                    or terminal_clause_alignment
                )
            ):
                score = (
                    0.5 * temporal_iou
                    + 0.2 * text_similarity
                    + 0.3 * terminal_text_similarity
                )
                candidates.append(
                    (
                        score,
                        prediction_index,
                        reference_index,
                        temporal_iou,
                        text_similarity,
                        terminal_text_similarity,
                    )
                )
    matched_predictions: set[int] = set()
    matched_reference: set[int] = set()
    matches: list[tuple[int, int, float, float, float]] = []
    for _score, prediction, gold, temporal_iou, text_similarity, terminal_text_similarity in sorted(
        candidates, reverse=True
    ):
        if prediction in matched_predictions or gold in matched_reference:
            continue
        matched_predictions.add(prediction)
        matched_reference.add(gold)
        matches.append(
            (prediction, gold, temporal_iou, text_similarity, terminal_text_similarity)
        )
    return tuple(matches)


def _contains_question_mark(text: str) -> bool:
    return "?" in text or "？" in text


def _metrics_by_candidate_type(
    predictions: tuple[_TimedQuestion, ...],
    matches: tuple[tuple[int, int, float, float, float], ...],
    reference_count: int,
    precision_target: float,
) -> dict[str, dict[str, object]]:
    matched_predictions = {item[0] for item in matches}
    types = (
        "sentence_terminal_question",
        "embedded_quoted_question",
        "elliptical_question",
    )
    return {
        candidate_type: _candidate_type_metrics(
            candidate_type,
            predictions,
            matched_predictions,
            reference_count,
            precision_target,
        )
        for candidate_type in types
    }


def _candidate_type_metrics(
    candidate_type: str,
    predictions: tuple[_TimedQuestion, ...],
    matched_predictions: set[int],
    reference_count: int,
    precision_target: float,
) -> dict[str, object]:
    indexes = tuple(
        index
        for index, item in enumerate(predictions)
        if _candidate_type(item) == candidate_type
    )
    true_positive = sum(index in matched_predictions for index in indexes)
    false_positive = len(indexes) - true_positive
    precision = _optional_ratio(true_positive, len(indexes))
    return {
        "predicted_questions": len(indexes),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "precision": precision,
        "recall_contribution": _ratio(true_positive, reference_count),
        "precision_target_met": (
            precision >= precision_target if precision is not None else None
        ),
    }


def _candidate_type(item: _TimedQuestion) -> str:
    evidence = set(item.evidence)
    if "embedded_quoted_question" in evidence:
        return "embedded_quoted_question"
    if "short_pronominal_case_phrase" in evidence:
        return "elliptical_question"
    return "sentence_terminal_question"


def _parse_timestamp(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", value)
    if match is None:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    hours, minutes, seconds, milliseconds = (int(item) for item in match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _temporal_iou(
    left: _TimedQuestion,
    right: _TimedQuestion,
    padding_seconds: float,
) -> float:
    right_start = right.start_seconds - padding_seconds
    right_end = right.end_seconds + padding_seconds
    overlap = max(
        0.0,
        min(left.end_seconds, right_end) - max(left.start_seconds, right_start),
    )
    union = max(left.end_seconds, right_end) - min(left.start_seconds, right_start)
    return _ratio(overlap, union)


def _temporal_match_score(
    prediction: _TimedQuestion,
    reference: _TimedQuestion,
    padding_seconds: float,
) -> float:
    temporal_iou = _temporal_iou(prediction, reference, padding_seconds)
    if _candidate_type(prediction) != "embedded_quoted_question":
        return temporal_iou
    padded_reference = _TimedQuestion(
        reference.start_seconds - padding_seconds,
        reference.end_seconds + padding_seconds,
        reference.text,
        reference.is_question,
    )
    overlap = _overlap_seconds(prediction, padded_reference)
    prediction_duration = prediction.end_seconds - prediction.start_seconds
    return max(temporal_iou, _ratio(overlap, prediction_duration))


def _overlap_seconds(left: _TimedQuestion, right: _TimedQuestion) -> float:
    return max(
        0.0,
        min(left.end_seconds, right.end_seconds)
        - max(left.start_seconds, right.start_seconds),
    )


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        _normalize_text(left),
        _normalize_text(right),
        autojunk=False,
    ).ratio()


def _terminal_text_similarity(prediction: str, reference: str) -> float:
    predicted_terminal = _normalize_text(prediction)[-16:]
    reference_prefix = reference[: _last_question_mark_index(reference)]
    reference_terminal = _normalize_text(reference_prefix)[-16:]
    if not predicted_terminal or not reference_terminal:
        return 0.0
    return SequenceMatcher(
        None,
        predicted_terminal,
        reference_terminal,
        autojunk=False,
    ).ratio()


def _question_endpoint_similarity(
    prediction: _TimedQuestion,
    reference: _TimedQuestion,
) -> float:
    terminal_similarity = _terminal_text_similarity(
        prediction.text,
        reference.text,
    )
    if _candidate_type(prediction) != "embedded_quoted_question":
        return terminal_similarity
    predicted = _normalize_text(prediction.text)
    reference_prefix = _normalize_text(
        reference.text[: _last_question_mark_index(reference.text)]
    )
    if predicted and predicted in reference_prefix:
        return 1.0
    return terminal_similarity


def _matches_reference_terminal_clause(
    prediction: _TimedQuestion,
    reference: _TimedQuestion,
    padding_seconds: float,
) -> bool:
    if _candidate_type(prediction) == "embedded_quoted_question":
        return False
    predicted = _normalize_text(prediction.text)
    reference_prefix = _normalize_text(
        reference.text[: _last_question_mark_index(reference.text)]
    )
    if not predicted or not reference_prefix.endswith(predicted):
        return False
    return abs(prediction.start_seconds - reference.end_seconds) <= padding_seconds


def _last_question_mark_index(text: str) -> int:
    positions = (text.rfind("?"), text.rfind("？"))
    position = max(positions)
    return position if position >= 0 else len(text)


def _normalize_text(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text)
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def _as_dict(item: _TimedQuestion) -> dict[str, object]:
    result: dict[str, object] = {
        "start_seconds": item.start_seconds,
        "end_seconds": item.end_seconds,
        "text": item.text,
        "is_question": item.is_question,
    }
    if item.segment_position is not None:
        result["segment_position"] = item.segment_position
    if item.sentence_index is not None:
        result["sentence_index"] = item.sentence_index
    if item.confidence is not None:
        result["confidence"] = item.confidence
    if item.evidence:
        result["evidence"] = list(item.evidence)
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    parser = ArgumentParser(
        description="Evaluate conservative Japanese question-mark restoration."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Reviewed classified dataset used to scope false negatives.",
    )
    args = parser.parse_args()
    report = write_question_punctuation_evaluation(
        args.reference,
        args.artifact,
        args.output,
        annotation_path=args.annotations,
    )
    metrics = report["metrics"]
    precision_text = (
        f"{metrics['precision']:.4f}"
        if metrics["precision"] is not None
        else "n/a"
    )
    print(
        f"precision={precision_text} "
        f"recall={metrics['recall']:.4f} "
        f"tp={metrics['true_positive']} "
        f"fp={metrics['false_positive']} "
        f"fn={metrics['false_negative']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

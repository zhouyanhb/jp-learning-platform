"""Non-destructive candidates for ASR coverage and recognition anomalies."""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import fmean
import unicodedata

from jp_learning_platform.domain import Segment, TimeRange
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    SudachiMorphologicalAnalyzer,
    morphological_particle_chain_penalty,
)
from jp_learning_platform.workflow.transcript_anomaly_stage import (
    TranscriptAnomalyCandidate,
    TranscriptAnomalyRequest,
)


@dataclass(frozen=True, slots=True)
class ConservativeTranscriptAnomalyDetector:
    min_coverage_gap_seconds: float = 1.5
    min_stable_context_gap_seconds: float = 8.0
    min_stable_context_duration_seconds: float = 3.0
    min_stable_context_characters: int = 12
    uncertain_edge_confidence: float = 0.65
    max_secondary_speech_seconds: float = 3.0
    secondary_speech_confidence: float = 0.55
    stable_context_confidence: float = 0.8
    min_internal_word_gap_seconds: float = 1.0
    uncertain_internal_edge_confidence: float = 0.65
    morphological_word_confidence_threshold: float = 0.35

    def detect(
        self,
        request: TranscriptAnomalyRequest,
    ) -> tuple[TranscriptAnomalyCandidate, ...]:
        candidates: list[TranscriptAnomalyCandidate] = []
        segments = request.segments
        morphological_analyzer = None
        for segment in segments:
            for sentence_index, sentence in enumerate(segment.sentences):
                if not _has_lexical_alignment(sentence):
                    candidates.append(
                        TranscriptAnomalyCandidate(
                            kind="possible_alignment_failure",
                            time_range=sentence.time_range,
                            segment_positions=(segment.position,),
                            sentence_indexes=(sentence_index,),
                            confidence=0.95,
                            evidence=(
                                "recognized_text_without_lexical_alignment",
                            ),
                        )
                    )
            if _is_repeated_laughter(segment.text):
                candidates.append(
                    TranscriptAnomalyCandidate(
                        kind="possible_repeated_laughter",
                        time_range=segment.time_range,
                        segment_positions=(segment.position,),
                        confidence=0.9,
                        evidence=(
                            "non_lexical_utterance",
                            "repeated_laughter_syllables",
                        ),
                    )
                )
            elif (
                not any(
                    _has_lexical_alignment(sentence)
                    for sentence in segment.sentences
                )
                and _is_repeated_vocalization(segment.text)
            ):
                candidates.append(
                    TranscriptAnomalyCandidate(
                        kind="possible_repeated_vocalization",
                        time_range=segment.time_range,
                        segment_positions=(segment.position,),
                        confidence=0.9,
                        evidence=(
                            "missing_lexical_alignment",
                            "repeated_short_text_unit",
                        ),
                    )
                )
            elif _is_background_sound_annotation(segment.text):
                candidates.append(
                    TranscriptAnomalyCandidate(
                        kind="possible_background_sound",
                        time_range=segment.time_range,
                        segment_positions=(segment.position,),
                        confidence=0.95,
                        evidence=(
                            "non_speech_annotation",
                            "background_sound_label",
                        ),
                    )
                )
            word_confidences = tuple(
                word.confidence
                for sentence in segment.sentences
                for word in sentence.words
                if word.confidence is not None
            )
            if (
                word_confidences
                and min(word_confidences)
                <= self.morphological_word_confidence_threshold
            ):
                morphological_analyzer = (
                    morphological_analyzer or SudachiMorphologicalAnalyzer()
                )
                if morphological_particle_chain_penalty(
                    segment.text,
                    morphological_analyzer,
                ):
                    candidates.append(
                        TranscriptAnomalyCandidate(
                            kind="possible_morphological_asr_error",
                            time_range=segment.time_range,
                            segment_positions=(segment.position,),
                            confidence=round(1.0 - min(word_confidences), 3),
                            evidence=(
                                "low_confidence_word",
                                "unlikely_morphological_particle_chain",
                            ),
                        )
                    )
            for sentence in segment.sentences:
                for left, right in zip(sentence.words, sentence.words[1:]):
                    gap = (
                        right.time_range.start_seconds
                        - left.time_range.end_seconds
                    )
                    edge_confidences = tuple(
                        value
                        for value in (left.confidence, right.confidence)
                        if value is not None
                    )
                    if (
                        gap >= self.min_internal_word_gap_seconds
                        and edge_confidences
                        and min(edge_confidences)
                        <= self.uncertain_internal_edge_confidence
                    ):
                        candidates.append(
                            TranscriptAnomalyCandidate(
                                kind="possible_internal_asr_omission",
                                time_range=TimeRange(
                                    left.time_range.end_seconds,
                                    right.time_range.start_seconds,
                                ),
                                segment_positions=(segment.position,),
                                confidence=round(
                                    1.0 - min(edge_confidences),
                                    3,
                                ),
                                evidence=(
                                    "large_internal_word_gap",
                                    "low_confidence_gap_edge",
                                ),
                            )
                        )
        for left, right in zip(segments, segments[1:]):
            gap = right.time_range.start_seconds - left.time_range.end_seconds
            edge_confidence = _edge_confidence(left, right)
            uncertain_edges = gap >= self.min_coverage_gap_seconds and (
                edge_confidence is not None
                and edge_confidence <= self.uncertain_edge_confidence
            )
            stable_context_discontinuity = (
                gap >= self.min_stable_context_gap_seconds
                and _has_substantial_context(
                    left,
                    self.min_stable_context_duration_seconds,
                    self.min_stable_context_characters,
                )
                and _has_substantial_context(
                    right,
                    self.min_stable_context_duration_seconds,
                    self.min_stable_context_characters,
                )
            )
            if uncertain_edges or stable_context_discontinuity:
                evidence = (
                    ("uncovered_time_range", "low_confidence_edges")
                    if uncertain_edges
                    else (
                        "long_uncovered_time_range",
                        "substantial_stable_context",
                    )
                )
                confidence = (
                    round(1.0 - edge_confidence, 3)
                    if uncertain_edges and edge_confidence is not None
                    else _stable_gap_confidence(
                        gap,
                        self.min_stable_context_gap_seconds,
                    )
                )
                candidates.append(
                    TranscriptAnomalyCandidate(
                        kind="possible_asr_omission",
                        time_range=TimeRange(
                            left.time_range.end_seconds,
                            right.time_range.start_seconds,
                        ),
                        segment_positions=(left.position, right.position),
                        confidence=confidence,
                        evidence=evidence,
                    )
                )

        for index, segment in enumerate(segments):
            if index == 0 or index + 1 >= len(segments):
                continue
            confidence = _segment_confidence(segment)
            before = _segment_confidence(segments[index - 1])
            after = _segment_confidence(segments[index + 1])
            if (
                confidence is not None
                and before is not None
                and after is not None
                and segment.time_range.duration_seconds
                <= self.max_secondary_speech_seconds
                and confidence <= self.secondary_speech_confidence
                and min(before, after) >= self.stable_context_confidence
            ):
                candidates.append(
                    TranscriptAnomalyCandidate(
                        kind="possible_background_speech",
                        time_range=segment.time_range,
                        segment_positions=(segment.position,),
                        confidence=round(min(before, after) - confidence, 3),
                        evidence=("short_low_confidence_insert", "stable_neighbors"),
                    )
                )
        return tuple(candidates)


def _segment_confidence(segment: Segment) -> float | None:
    values = [
        word.confidence
        for sentence in segment.sentences
        for word in sentence.words
        if word.confidence is not None
    ]
    return fmean(values) if values else None


def _edge_confidence(left: Segment, right: Segment) -> float | None:
    left_values = [word.confidence for word in left.sentences[-1].words[-3:]] if left.sentences else []
    right_values = [word.confidence for word in right.sentences[0].words[:3]] if right.sentences else []
    values = [value for value in (*left_values, *right_values) if value is not None]
    return fmean(values) if values else None


def _has_substantial_context(
    segment: Segment,
    minimum_duration_seconds: float,
    minimum_characters: int,
) -> bool:
    lexical_characters = sum(character.isalnum() for character in segment.text)
    return (
        segment.time_range.duration_seconds >= minimum_duration_seconds
        and lexical_characters >= minimum_characters
        and _segment_confidence(segment) is not None
    )


def _stable_gap_confidence(gap: float, minimum_gap: float) -> float:
    return round(min(0.9, 0.7 + (gap - minimum_gap) / 100.0), 3)


def _is_repeated_laughter(text: str) -> bool:
    normalized = "".join(
        character.lower()
        for character in unicodedata.normalize("NFKC", text)
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )
    if re.fullmatch(r"(?:w{3,}|(?:ha){2,}|(?:he){2,}|(?:lol)+)", normalized):
        return True
    if not re.fullmatch(r"[あいうえおはひふへほっー]+", normalized):
        return False
    laughter_consonants = "".join(
        character for character in normalized if character in "はひふへほ"
    )
    return (
        len(normalized) >= 2
        and len(laughter_consonants) >= 2
        and len(laughter_consonants) / len(normalized.replace("ー", "")) >= 0.6
        and len(set(normalized.replace("ー", ""))) <= 3
    )


def _has_lexical_alignment(sentence) -> bool:
    return any(
        any(character.isalnum() for character in word.text)
        for word in sentence.words
    )


def _is_repeated_vocalization(text: str) -> bool:
    normalized = "".join(
        character.lower()
        for character in unicodedata.normalize("NFKC", text)
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )
    if len(normalized) < 8:
        return False
    if len(set(normalized)) <= 2:
        return True
    for unit_length in range(1, min(6, len(normalized) // 4 + 1)):
        unit = normalized[:unit_length]
        repetitions, remainder = divmod(len(normalized), unit_length)
        if remainder == 0 and repetitions >= 4 and unit * repetitions == normalized:
            return True
    return False


def _is_background_sound_annotation(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if normalized and all(character in "♪♫♬♩~〜 " for character in normalized):
        return True
    match = re.fullmatch(r"[\[(【<〈《「『（](.+?)[\])】>〉》」』）]", normalized)
    if match is None:
        return False
    label = re.sub(r"\s+", "", match.group(1)).lower()
    sound_labels = (
        "音楽",
        "bgm",
        "効果音",
        "環境音",
        "雑音",
        "拍手",
        "歓声",
        "笑い声",
        "笑声",
        "ざわめき",
        "ノイズ",
        "music",
        "applause",
        "laughter",
    )
    return any(label == value or label.startswith(f"{value}:") for value in sound_labels)

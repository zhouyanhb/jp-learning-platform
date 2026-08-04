"""Text-independent candidates for ASR coverage gaps and secondary speech."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from jp_learning_platform.domain import Segment, TimeRange
from jp_learning_platform.workflow.transcript_anomaly_stage import (
    TranscriptAnomalyCandidate,
    TranscriptAnomalyRequest,
)


@dataclass(frozen=True, slots=True)
class ConservativeTranscriptAnomalyDetector:
    min_coverage_gap_seconds: float = 1.5
    uncertain_edge_confidence: float = 0.65
    max_secondary_speech_seconds: float = 3.0
    secondary_speech_confidence: float = 0.55
    stable_context_confidence: float = 0.8
    min_internal_word_gap_seconds: float = 1.0
    uncertain_internal_edge_confidence: float = 0.65

    def detect(
        self,
        request: TranscriptAnomalyRequest,
    ) -> tuple[TranscriptAnomalyCandidate, ...]:
        candidates: list[TranscriptAnomalyCandidate] = []
        segments = request.segments
        for segment in segments:
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
            if gap >= self.min_coverage_gap_seconds and (
                edge_confidence is not None
                and edge_confidence <= self.uncertain_edge_confidence
            ):
                candidates.append(
                    TranscriptAnomalyCandidate(
                        kind="possible_asr_omission",
                        time_range=TimeRange(
                            left.time_range.end_seconds,
                            right.time_range.start_seconds,
                        ),
                        segment_positions=(left.position, right.position),
                        confidence=round(1.0 - edge_confidence, 3),
                        evidence=("uncovered_time_range", "low_confidence_edges"),
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

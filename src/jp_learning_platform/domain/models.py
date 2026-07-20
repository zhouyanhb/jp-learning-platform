"""Core domain models for the subtitle pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TypeVar

MIN_SECONDS = 0.0
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_SEGMENT_POSITION = 0
MIN_SUBTITLE_INDEX = 1

T = TypeVar("T")


def _normalize_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


def _normalize_optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None

    return _normalize_text(value, field_name)


def _normalize_seconds(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number of seconds.")

    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a number of seconds.") from error

    if not isfinite(seconds):
        raise ValueError(f"{field_name} must be finite.")

    if seconds < MIN_SECONDS:
        raise ValueError(f"{field_name} must be non-negative.")

    return seconds


def _normalize_position(value: int, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")

    return value


def _normalize_optional_confidence(value: float | None) -> float | None:
    if value is None:
        return None

    confidence = _normalize_seconds(value, "confidence")
    if confidence < MIN_CONFIDENCE or confidence > MAX_CONFIDENCE:
        raise ValueError("confidence must be between 0.0 and 1.0.")

    return confidence


def _tuple_of_type(
    values: Iterable[T],
    item_type: type[T],
    field_name: str,
) -> tuple[T, ...]:
    try:
        tuple_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field_name} must be iterable.") from error

    for value in tuple_values:
        if not isinstance(value, item_type):
            raise TypeError(f"{field_name} must contain {item_type.__name__} values.")

    return tuple_values


@dataclass(frozen=True, slots=True)
class TimeRange:
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        start_seconds = _normalize_seconds(self.start_seconds, "start_seconds")
        end_seconds = _normalize_seconds(self.end_seconds, "end_seconds")

        if end_seconds < start_seconds:
            raise ValueError(
                "end_seconds must be greater than or equal to start_seconds."
            )

        object.__setattr__(self, "start_seconds", start_seconds)
        object.__setattr__(self, "end_seconds", end_seconds)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def contains(self, other: TimeRange) -> bool:
        if not isinstance(other, TimeRange):
            raise TypeError("other must be a TimeRange.")

        return (
            self.start_seconds <= other.start_seconds
            and other.end_seconds <= self.end_seconds
        )


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    time_range: TimeRange
    confidence: float | None = None
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")

        object.__setattr__(self, "text", _normalize_text(self.text, "text"))
        object.__setattr__(
            self,
            "confidence",
            _normalize_optional_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "speaker_id",
            _normalize_optional_text(self.speaker_id, "speaker_id"),
        )


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    time_range: TimeRange
    words: tuple[Word, ...] = ()
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")

        words = _tuple_of_type(self.words, Word, "words")
        for word in words:
            if not self.time_range.contains(word.time_range):
                raise ValueError("words must fall within the sentence time range.")

        object.__setattr__(self, "text", _normalize_text(self.text, "text"))
        object.__setattr__(self, "words", words)
        object.__setattr__(
            self,
            "speaker_id",
            _normalize_optional_text(self.speaker_id, "speaker_id"),
        )


@dataclass(frozen=True, slots=True)
class Segment:
    position: int
    text: str
    time_range: TimeRange
    sentences: tuple[Sentence, ...] = ()
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")

        sentences = _tuple_of_type(self.sentences, Sentence, "sentences")
        for sentence in sentences:
            if not self.time_range.contains(sentence.time_range):
                raise ValueError("sentences must fall within the segment time range.")

        object.__setattr__(
            self,
            "position",
            _normalize_position(self.position, "position", MIN_SEGMENT_POSITION),
        )
        object.__setattr__(self, "text", _normalize_text(self.text, "text"))
        object.__setattr__(self, "sentences", sentences)
        object.__setattr__(
            self,
            "speaker_id",
            _normalize_optional_text(self.speaker_id, "speaker_id"),
        )


@dataclass(frozen=True, slots=True)
class Subtitle:
    index: int
    text: str
    time_range: TimeRange
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")

        object.__setattr__(
            self,
            "index",
            _normalize_position(self.index, "index", MIN_SUBTITLE_INDEX),
        )
        object.__setattr__(self, "text", _normalize_text(self.text, "text"))
        object.__setattr__(
            self,
            "speaker_id",
            _normalize_optional_text(self.speaker_id, "speaker_id"),
        )


@dataclass(frozen=True, slots=True)
class SentenceBoundaryCandidate:
    segment_position: int
    after_word_index: int
    boundary_time_seconds: float
    pause_time_range: TimeRange
    acoustic_score: float
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.pause_time_range, TimeRange):
            raise TypeError("pause_time_range must be a TimeRange.")

        boundary_time_seconds = _normalize_seconds(
            self.boundary_time_seconds,
            "boundary_time_seconds",
        )
        if not self.pause_time_range.contains(
            TimeRange(boundary_time_seconds, boundary_time_seconds)
        ):
            raise ValueError(
                "boundary_time_seconds must fall within the pause time range."
            )

        object.__setattr__(
            self,
            "segment_position",
            _normalize_position(
                self.segment_position,
                "segment_position",
                MIN_SEGMENT_POSITION,
            ),
        )
        object.__setattr__(
            self,
            "after_word_index",
            _normalize_position(self.after_word_index, "after_word_index", 0),
        )
        object.__setattr__(
            self,
            "boundary_time_seconds",
            boundary_time_seconds,
        )
        acoustic_score = _normalize_optional_confidence(self.acoustic_score)
        if acoustic_score is None:
            raise TypeError("acoustic_score must be a number.")

        object.__setattr__(self, "acoustic_score", acoustic_score)
        object.__setattr__(self, "source", _normalize_text(self.source, "source"))


@dataclass(frozen=True, slots=True)
class Document:
    source_path: Path
    segments: tuple[Segment, ...] = ()
    subtitles: tuple[Subtitle, ...] = ()
    sentence_boundary_candidates: tuple[SentenceBoundaryCandidate, ...] = ()

    def __post_init__(self) -> None:
        source_path = Path(self.source_path)
        segments = _tuple_of_type(self.segments, Segment, "segments")
        subtitles = _tuple_of_type(self.subtitles, Subtitle, "subtitles")
        sentence_boundary_candidates = _tuple_of_type(
            self.sentence_boundary_candidates,
            SentenceBoundaryCandidate,
            "sentence_boundary_candidates",
        )

        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "subtitles", subtitles)
        object.__setattr__(
            self,
            "sentence_boundary_candidates",
            sentence_boundary_candidates,
        )


@dataclass(frozen=True, slots=True)
class PipelineContext:
    run_id: str
    document: Document
    working_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.document, Document):
            raise TypeError("document must be a Document.")

        object.__setattr__(self, "run_id", _normalize_text(self.run_id, "run_id"))
        object.__setattr__(self, "working_directory", Path(self.working_directory))


__all__ = [
    "Document",
    "PipelineContext",
    "Segment",
    "Sentence",
    "SentenceBoundaryCandidate",
    "Subtitle",
    "TimeRange",
    "Word",
]

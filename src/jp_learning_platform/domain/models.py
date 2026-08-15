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

    def __post_init__(self) -> None:
        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")

        object.__setattr__(self, "text", _normalize_text(self.text, "text"))
        object.__setattr__(
            self,
            "confidence",
            _normalize_optional_confidence(self.confidence),
        )


@dataclass(frozen=True, slots=True)
class LearningWord:
    text: str
    start_char: int
    end_char: int
    aligned_word_indexes: tuple[int, ...]
    time_range: TimeRange
    timing_estimated: bool = False
    is_structure: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")
        if not isinstance(self.timing_estimated, bool):
            raise TypeError("timing_estimated must be a bool.")
        if not isinstance(self.is_structure, bool):
            raise TypeError("is_structure must be a bool.")

        start_char = _normalize_position(self.start_char, "start_char", 0)
        end_char = _normalize_position(self.end_char, "end_char", 1)
        if end_char <= start_char:
            raise ValueError("end_char must be greater than start_char.")

        indexes = tuple(self.aligned_word_indexes)
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in indexes
        ):
            raise TypeError("aligned_word_indexes must contain integers.")
        if any(index < 0 for index in indexes):
            raise ValueError("aligned_word_indexes must be non-negative.")
        if tuple(sorted(set(indexes))) != indexes:
            raise ValueError(
                "aligned_word_indexes must be unique and strictly increasing."
            )
        if indexes and indexes != tuple(range(indexes[0], indexes[-1] + 1)):
            raise ValueError("aligned_word_indexes must be contiguous.")

        object.__setattr__(self, "text", _normalize_text(self.text, "text"))
        object.__setattr__(self, "start_char", start_char)
        object.__setattr__(self, "end_char", end_char)
        object.__setattr__(self, "aligned_word_indexes", indexes)


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    time_range: TimeRange
    words: tuple[Word, ...] = ()
    learning_words: tuple[LearningWord, ...] = ()
    is_question: bool = False
    asr_boundary_word_indexes: tuple[int, ...] = ()
    anomaly_kinds: tuple[str, ...] = ()
    excluded_from_language_evaluation: bool = False
    learning_words_suppressed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")
        if not isinstance(self.is_question, bool):
            raise TypeError("is_question must be a bool.")
        if not isinstance(self.excluded_from_language_evaluation, bool):
            raise TypeError("excluded_from_language_evaluation must be a bool.")
        if not isinstance(self.learning_words_suppressed, bool):
            raise TypeError("learning_words_suppressed must be a bool.")

        words = _tuple_of_type(self.words, Word, "words")
        for word in words:
            if not self.time_range.contains(word.time_range):
                raise ValueError("words must fall within the sentence time range.")
        learning_words = _tuple_of_type(
            self.learning_words,
            LearningWord,
            "learning_words",
        )
        for learning_word in learning_words:
            if not self.time_range.contains(learning_word.time_range):
                raise ValueError(
                    "learning_words must fall within the sentence time range."
                )
            if learning_word.end_char > len(self.text.strip()):
                raise ValueError("learning_words must fall within sentence text.")
            if any(index >= len(words) for index in learning_word.aligned_word_indexes):
                raise ValueError("learning_words must reference existing words.")
        asr_boundaries = tuple(self.asr_boundary_word_indexes)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in asr_boundaries):
            raise TypeError("asr_boundary_word_indexes must contain integers.")
        if any(index <= 0 or index >= len(words) for index in asr_boundaries):
            raise ValueError("asr_boundary_word_indexes must fall between words.")
        if tuple(sorted(set(asr_boundaries))) != asr_boundaries:
            raise ValueError("asr_boundary_word_indexes must be unique and increasing.")
        anomaly_kinds = tuple(self.anomaly_kinds)
        if any(not isinstance(kind, str) or not kind.strip() for kind in anomaly_kinds):
            raise TypeError("anomaly_kinds must contain non-empty strings.")
        if tuple(dict.fromkeys(anomaly_kinds)) != anomaly_kinds:
            raise ValueError("anomaly_kinds must be unique.")

        object.__setattr__(self, "text", _normalize_text(self.text, "text"))
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "learning_words", learning_words)
        object.__setattr__(self, "asr_boundary_word_indexes", asr_boundaries)
        object.__setattr__(self, "anomaly_kinds", anomaly_kinds)


@dataclass(frozen=True, slots=True)
class Segment:
    position: int
    text: str
    time_range: TimeRange
    sentences: tuple[Sentence, ...] = ()

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


@dataclass(frozen=True, slots=True)
class Subtitle:
    index: int
    text: str
    time_range: TimeRange
    source_sentence_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")

        object.__setattr__(
            self,
            "index",
            _normalize_position(self.index, "index", MIN_SUBTITLE_INDEX),
        )
        object.__setattr__(self, "text", _normalize_text(self.text, "text"))
        if self.source_sentence_index is not None:
            object.__setattr__(
                self,
                "source_sentence_index",
                _normalize_position(
                    self.source_sentence_index,
                    "source_sentence_index",
                    0,
                ),
            )


@dataclass(frozen=True, slots=True)
class Document:
    source_path: Path
    segments: tuple[Segment, ...] = ()
    subtitles: tuple[Subtitle, ...] = ()

    def __post_init__(self) -> None:
        source_path = Path(self.source_path)
        segments = _tuple_of_type(self.segments, Segment, "segments")
        subtitles = _tuple_of_type(self.subtitles, Subtitle, "subtitles")

        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "subtitles", subtitles)


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
    "LearningWord",
    "PipelineContext",
    "Segment",
    "Sentence",
    "Subtitle",
    "TimeRange",
    "Word",
]

"""Sentence boundary resolution workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment
from jp_learning_platform.workflow.runtime import StageResult

SENTENCE_BOUNDARY_STAGE_NAME = "sentence-boundary-resolution"

T = TypeVar("T")


def _normalize_name(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


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


def _normalize_position(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")

    return value


def _normalize_seconds(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")

    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a number.") from error

    if seconds < 0:
        raise ValueError(f"{field_name} must be non-negative.")

    return seconds


@dataclass(frozen=True, slots=True)
class SentenceBoundaryResolutionRequest:
    """Input passed to sentence boundary resolvers."""

    source_path: Path
    working_directory: Path
    run_id: str
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "working_directory", Path(self.working_directory))
        object.__setattr__(self, "run_id", _normalize_name(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "segments",
            _tuple_of_type(self.segments, Segment, "segments"),
        )


@dataclass(frozen=True, slots=True)
class SentenceBoundaryDecision:
    """Diagnostic record for one accepted sentence boundary."""

    segment_position: int
    sentence_index: int
    word_index: int
    gap_seconds: float
    reason: str
    left_text: str
    right_text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_position",
            _normalize_position(self.segment_position, "segment_position"),
        )
        object.__setattr__(
            self,
            "sentence_index",
            _normalize_position(self.sentence_index, "sentence_index"),
        )
        object.__setattr__(
            self,
            "word_index",
            _normalize_position(self.word_index, "word_index"),
        )
        object.__setattr__(
            self,
            "gap_seconds",
            _normalize_seconds(self.gap_seconds, "gap_seconds"),
        )
        object.__setattr__(self, "reason", _normalize_name(self.reason, "reason"))
        object.__setattr__(self, "left_text", _normalize_name(self.left_text, "left_text"))
        object.__setattr__(
            self,
            "right_text",
            _normalize_name(self.right_text, "right_text"),
        )


@dataclass(frozen=True, slots=True)
class SentenceBoundaryResolution:
    """Transcript segments after sentence boundary resolution."""

    source_path: Path
    segments: tuple[Segment, ...]
    decisions: tuple[SentenceBoundaryDecision, ...] = ()

    def __post_init__(self) -> None:
        segments = _tuple_of_type(self.segments, Segment, "segments")
        if not segments:
            raise ValueError("segments must not be empty.")

        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "segments", segments)
        object.__setattr__(
            self,
            "decisions",
            _tuple_of_type(
                self.decisions,
                SentenceBoundaryDecision,
                "decisions",
            ),
        )


class SentenceBoundaryResolver(Protocol):
    """Contract implemented by sentence boundary resolver adapters."""

    def resolve(
        self,
        request: SentenceBoundaryResolutionRequest,
    ) -> SentenceBoundaryResolution:
        """Split segment sentences into sentence-level units."""


class SentenceBoundaryStageError(RuntimeError):
    """Base error for sentence boundary workflow failures."""


class InvalidSentenceBoundaryResolverError(SentenceBoundaryStageError):
    """Raised when a resolver does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__(
            "Sentence boundary resolver must define a callable resolve method."
        )


class MissingSentenceBoundarySegmentsError(SentenceBoundaryStageError):
    """Raised when sentence boundary resolution has no segments to inspect."""

    def __init__(self) -> None:
        super().__init__(
            "Sentence boundary resolution requires existing document segments."
        )


class InvalidSentenceBoundaryResolutionError(SentenceBoundaryStageError):
    """Raised when a resolver returns an invalid resolution result."""


@dataclass(frozen=True, slots=True)
class SentenceBoundaryResolutionStage:
    """Workflow stage that applies sentence boundary resolution."""

    resolver: SentenceBoundaryResolver
    name: str = SENTENCE_BOUNDARY_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.resolver, "resolve", None)):
            raise InvalidSentenceBoundaryResolverError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.segments:
            raise MissingSentenceBoundarySegmentsError()

        request = SentenceBoundaryResolutionRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
        )
        resolution = self.resolver.resolve(request)
        if not isinstance(resolution, SentenceBoundaryResolution):
            raise InvalidSentenceBoundaryResolutionError(
                "Sentence boundary resolver must return SentenceBoundaryResolution."
            )

        if resolution.source_path != request.source_path:
            raise InvalidSentenceBoundaryResolutionError(
                "Sentence boundary resolution source path must match the request."
            )

        document = Document(
            source_path=context.document.source_path,
            segments=resolution.segments,
            subtitles=context.document.subtitles,
        )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=document,
            working_directory=context.working_directory,
        )
        return StageResult(
            stage_name=self.name,
            context=next_context,
            data={"decisions": resolution.decisions},
        )


__all__ = [
    "InvalidSentenceBoundaryResolutionError",
    "InvalidSentenceBoundaryResolverError",
    "MissingSentenceBoundarySegmentsError",
    "SENTENCE_BOUNDARY_STAGE_NAME",
    "SentenceBoundaryDecision",
    "SentenceBoundaryResolution",
    "SentenceBoundaryResolutionRequest",
    "SentenceBoundaryResolutionStage",
    "SentenceBoundaryResolver",
    "SentenceBoundaryStageError",
]

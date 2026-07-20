"""Sentence boundary candidate detection and resolution workflow stages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    SentenceBoundaryCandidate,
)
from jp_learning_platform.workflow.runtime import StageResult

SENTENCE_BOUNDARY_DETECTION_STAGE_NAME = "sentence-boundary-detection"
SENTENCE_BOUNDARY_RESOLVER_STAGE_NAME = "sentence-boundary-resolver"

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


@dataclass(frozen=True, slots=True)
class SentenceBoundaryDetectionRequest:
    """Input passed to acoustic sentence boundary detectors."""

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
class SentenceBoundaryDetection:
    """Candidate boundary positions detected from speech and word timing."""

    source_path: Path
    candidates: tuple[SentenceBoundaryCandidate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "candidates",
            _tuple_of_type(
                self.candidates,
                SentenceBoundaryCandidate,
                "candidates",
            ),
        )


class SentenceBoundaryDetector(Protocol):
    """Contract implemented by acoustic boundary detector adapters."""

    def detect(
        self,
        request: SentenceBoundaryDetectionRequest,
    ) -> SentenceBoundaryDetection:
        """Return candidate sentence boundaries for aligned segments."""


@dataclass(frozen=True, slots=True)
class SentenceBoundaryResolutionRequest:
    """Input passed to final sentence boundary resolvers."""

    source_path: Path
    working_directory: Path
    run_id: str
    segments: tuple[Segment, ...]
    candidates: tuple[SentenceBoundaryCandidate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "working_directory", Path(self.working_directory))
        object.__setattr__(self, "run_id", _normalize_name(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "segments",
            _tuple_of_type(self.segments, Segment, "segments"),
        )
        object.__setattr__(
            self,
            "candidates",
            _tuple_of_type(
                self.candidates,
                SentenceBoundaryCandidate,
                "candidates",
            ),
        )


@dataclass(frozen=True, slots=True)
class SentenceBoundaryResolution:
    """Resolved segments with final sentence boundaries applied."""

    source_path: Path
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        segments = _tuple_of_type(self.segments, Segment, "segments")
        if not segments:
            raise ValueError("segments must not be empty.")

        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "segments", segments)


class SentenceBoundaryResolver(Protocol):
    """Contract implemented by final sentence boundary resolvers."""

    def resolve(
        self,
        request: SentenceBoundaryResolutionRequest,
    ) -> SentenceBoundaryResolution:
        """Apply candidate boundaries to current segments."""


class SentenceBoundaryStageError(RuntimeError):
    """Base error for sentence boundary workflow failures."""


class InvalidSentenceBoundaryDetectorError(SentenceBoundaryStageError):
    """Raised when a detector does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__(
            "Sentence boundary detector must define a callable detect method."
        )


class InvalidSentenceBoundaryResolverError(SentenceBoundaryStageError):
    """Raised when a resolver does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__(
            "Sentence boundary resolver must define a callable resolve method."
        )


class MissingSentenceBoundarySegmentsError(SentenceBoundaryStageError):
    """Raised when sentence boundary work has no segments to inspect."""

    def __init__(self) -> None:
        super().__init__(
            "Sentence boundary detection requires existing document segments."
        )


class InvalidSentenceBoundaryDetectionError(SentenceBoundaryStageError):
    """Raised when a detector returns an invalid detection result."""


class InvalidSentenceBoundaryResolutionError(SentenceBoundaryStageError):
    """Raised when a resolver returns an invalid resolution result."""


@dataclass(frozen=True, slots=True)
class SentenceBoundaryDetectionStage:
    """Workflow stage that records acoustic sentence boundary candidates."""

    detector: SentenceBoundaryDetector
    name: str = SENTENCE_BOUNDARY_DETECTION_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.detector, "detect", None)):
            raise InvalidSentenceBoundaryDetectorError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.segments:
            raise MissingSentenceBoundarySegmentsError()

        request = SentenceBoundaryDetectionRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
        )
        detection = self.detector.detect(request)
        if not isinstance(detection, SentenceBoundaryDetection):
            raise InvalidSentenceBoundaryDetectionError(
                "Sentence boundary detector must return SentenceBoundaryDetection."
            )

        if detection.source_path != request.source_path:
            raise InvalidSentenceBoundaryDetectionError(
                "Sentence boundary detection source path must match the request."
            )

        document = Document(
            source_path=context.document.source_path,
            segments=context.document.segments,
            subtitles=context.document.subtitles,
            sentence_boundary_candidates=detection.candidates,
        )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=document,
            working_directory=context.working_directory,
        )
        return StageResult(stage_name=self.name, context=next_context)


@dataclass(frozen=True, slots=True)
class SentenceBoundaryResolverStage:
    """Workflow stage that applies final sentence boundaries after repair."""

    resolver: SentenceBoundaryResolver
    name: str = SENTENCE_BOUNDARY_RESOLVER_STAGE_NAME

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
            candidates=context.document.sentence_boundary_candidates,
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
            sentence_boundary_candidates=context.document.sentence_boundary_candidates,
        )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=document,
            working_directory=context.working_directory,
        )
        return StageResult(stage_name=self.name, context=next_context)


__all__ = [
    "InvalidSentenceBoundaryDetectionError",
    "InvalidSentenceBoundaryDetectorError",
    "InvalidSentenceBoundaryResolutionError",
    "InvalidSentenceBoundaryResolverError",
    "MissingSentenceBoundarySegmentsError",
    "SENTENCE_BOUNDARY_DETECTION_STAGE_NAME",
    "SENTENCE_BOUNDARY_RESOLVER_STAGE_NAME",
    "SentenceBoundaryDetection",
    "SentenceBoundaryDetectionRequest",
    "SentenceBoundaryDetectionStage",
    "SentenceBoundaryDetector",
    "SentenceBoundaryResolution",
    "SentenceBoundaryResolutionRequest",
    "SentenceBoundaryResolver",
    "SentenceBoundaryResolverStage",
    "SentenceBoundaryStageError",
]

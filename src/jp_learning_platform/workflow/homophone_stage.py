"""Homophone semantic resolution workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment
from jp_learning_platform.workflow.runtime import StageResult

HOMOPHONE_RESOLUTION_STAGE_NAME = "homophone-resolution"

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


def _normalize_optional_score(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")

    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a number.") from error


@dataclass(frozen=True, slots=True)
class HomophoneResolutionRequest:
    """Input passed to homophone semantic resolvers."""

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
class HomophoneCandidateScore:
    """One same-reading candidate considered by the resolver."""

    text: str
    reading: str
    score: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _normalize_name(self.text, "text"))
        object.__setattr__(self, "reading", _normalize_name(self.reading, "reading"))
        object.__setattr__(
            self,
            "score",
            _normalize_optional_score(self.score, "score"),
        )


@dataclass(frozen=True, slots=True)
class HomophoneResolutionDecision:
    """Diagnostic record for one homophone replacement decision."""

    segment_position: int
    sentence_index: int
    original_text: str
    selected_text: str
    reading: str
    accepted: bool
    reason: str
    original_score: float | None = None
    selected_score: float | None = None
    candidates: tuple[HomophoneCandidateScore, ...] = ()

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
            "original_text",
            _normalize_name(self.original_text, "original_text"),
        )
        object.__setattr__(
            self,
            "selected_text",
            _normalize_name(self.selected_text, "selected_text"),
        )
        object.__setattr__(self, "reading", _normalize_name(self.reading, "reading"))
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool.")

        object.__setattr__(self, "reason", _normalize_name(self.reason, "reason"))
        object.__setattr__(
            self,
            "original_score",
            _normalize_optional_score(self.original_score, "original_score"),
        )
        object.__setattr__(
            self,
            "selected_score",
            _normalize_optional_score(self.selected_score, "selected_score"),
        )
        object.__setattr__(
            self,
            "candidates",
            _tuple_of_type(
                self.candidates,
                HomophoneCandidateScore,
                "candidates",
            ),
        )


@dataclass(frozen=True, slots=True)
class HomophoneResolution:
    """Resolved transcript segments after safe homophone replacements."""

    source_path: Path
    segments: tuple[Segment, ...]
    decisions: tuple[HomophoneResolutionDecision, ...] = ()

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
                HomophoneResolutionDecision,
                "decisions",
            ),
        )


class HomophoneResolver(Protocol):
    """Contract implemented by homophone semantic resolver adapters."""

    def resolve(
        self,
        request: HomophoneResolutionRequest,
    ) -> HomophoneResolution:
        """Resolve same-reading ASR word confusions without free rewriting."""


class HomophoneStageError(RuntimeError):
    """Base error for homophone resolution workflow failures."""


class InvalidHomophoneResolverError(HomophoneStageError):
    """Raised when a resolver does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__(
            "Homophone resolver must define a callable resolve method."
        )


class MissingHomophoneSegmentsError(HomophoneStageError):
    """Raised when homophone resolution has no segments to inspect."""

    def __init__(self) -> None:
        super().__init__(
            "Homophone resolution requires existing document segments."
        )


class InvalidHomophoneResolutionError(HomophoneStageError):
    """Raised when a resolver returns an invalid resolution result."""


@dataclass(frozen=True, slots=True)
class HomophoneResolutionStage:
    """Workflow stage that applies constrained homophone semantic resolution."""

    resolver: HomophoneResolver
    name: str = HOMOPHONE_RESOLUTION_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.resolver, "resolve", None)):
            raise InvalidHomophoneResolverError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.segments:
            raise MissingHomophoneSegmentsError()

        request = HomophoneResolutionRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
        )
        resolution = self.resolver.resolve(request)
        if not isinstance(resolution, HomophoneResolution):
            raise InvalidHomophoneResolutionError(
                "Homophone resolver must return HomophoneResolution."
            )

        if resolution.source_path != request.source_path:
            raise InvalidHomophoneResolutionError(
                "Homophone resolution source path must match the request."
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
    "HOMOPHONE_RESOLUTION_STAGE_NAME",
    "HomophoneCandidateScore",
    "HomophoneResolution",
    "HomophoneResolutionDecision",
    "HomophoneResolutionRequest",
    "HomophoneResolutionStage",
    "HomophoneResolver",
    "HomophoneStageError",
    "InvalidHomophoneResolutionError",
    "InvalidHomophoneResolverError",
    "MissingHomophoneSegmentsError",
]

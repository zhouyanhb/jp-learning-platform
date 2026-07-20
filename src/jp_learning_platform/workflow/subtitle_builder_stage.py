"""Subtitle builder workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment, Subtitle
from jp_learning_platform.workflow.runtime import StageResult

SUBTITLE_BUILDER_STAGE_NAME = "subtitle-builder"

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
class SubtitleBuildRequest:
    """Input passed from the workflow stage to a subtitle builder."""

    source_path: Path
    working_directory: Path
    run_id: str
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "working_directory",
            Path(self.working_directory),
        )
        object.__setattr__(self, "run_id", _normalize_name(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "segments",
            _tuple_of_type(self.segments, Segment, "segments"),
        )


@dataclass(frozen=True, slots=True)
class SubtitleBuild:
    """Normalized subtitle builder output."""

    source_path: Path
    subtitles: tuple[Subtitle, ...]

    def __post_init__(self) -> None:
        subtitles = _tuple_of_type(self.subtitles, Subtitle, "subtitles")
        if not subtitles:
            raise ValueError("subtitles must not be empty.")

        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "subtitles", subtitles)


class SubtitleBuilder(Protocol):
    """Contract implemented by subtitle building adapters."""

    def build(self, request: SubtitleBuildRequest) -> SubtitleBuild:
        """Build subtitle entries from repaired transcript segments."""


class SubtitleBuilderStageError(RuntimeError):
    """Base error for subtitle builder stage failures."""


class InvalidSubtitleBuilderError(SubtitleBuilderStageError):
    """Raised when a configured builder does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__("Subtitle builder must define a callable build method.")


class MissingSubtitleBuildSegmentsError(SubtitleBuilderStageError):
    """Raised when the document has no repaired segments to build from."""

    def __init__(self) -> None:
        super().__init__("Subtitle building requires existing document segments.")


class InvalidSubtitleBuildError(SubtitleBuilderStageError):
    """Raised when a builder returns an invalid subtitle build result."""


@dataclass(frozen=True, slots=True)
class SubtitleBuilderStage:
    """Workflow stage that coordinates subtitle construction."""

    builder: SubtitleBuilder
    name: str = SUBTITLE_BUILDER_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.builder, "build", None)):
            raise InvalidSubtitleBuilderError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.segments:
            raise MissingSubtitleBuildSegmentsError()

        request = SubtitleBuildRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
        )
        build = self.builder.build(request)
        if not isinstance(build, SubtitleBuild):
            raise InvalidSubtitleBuildError(
                "Subtitle builder must return a SubtitleBuild."
            )

        if build.source_path != request.source_path:
            raise InvalidSubtitleBuildError(
                "Subtitle build source path must match the request source path."
            )

        document = Document(
            source_path=context.document.source_path,
            segments=context.document.segments,
            subtitles=build.subtitles,
            sentence_boundary_candidates=(
                context.document.sentence_boundary_candidates
            ),
        )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=document,
            working_directory=context.working_directory,
        )

        return StageResult(stage_name=self.name, context=next_context)


__all__ = [
    "InvalidSubtitleBuildError",
    "InvalidSubtitleBuilderError",
    "MissingSubtitleBuildSegmentsError",
    "SUBTITLE_BUILDER_STAGE_NAME",
    "SubtitleBuild",
    "SubtitleBuildRequest",
    "SubtitleBuilder",
    "SubtitleBuilderStage",
    "SubtitleBuilderStageError",
]

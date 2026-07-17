"""Subtitle merger workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment, Subtitle
from jp_learning_platform.workflow.runtime import StageResult

SUBTITLE_MERGER_STAGE_NAME = "subtitle-merger"

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
class SubtitleMergeRequest:
    """Input passed from the workflow stage to a subtitle merger."""

    source_path: Path
    working_directory: Path
    run_id: str
    segments: tuple[Segment, ...]
    subtitles: tuple[Subtitle, ...]

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
        object.__setattr__(
            self,
            "subtitles",
            _tuple_of_type(self.subtitles, Subtitle, "subtitles"),
        )


@dataclass(frozen=True, slots=True)
class SubtitleMerge:
    """Normalized subtitle merger output."""

    source_path: Path
    subtitles: tuple[Subtitle, ...]

    def __post_init__(self) -> None:
        subtitles = _tuple_of_type(self.subtitles, Subtitle, "subtitles")
        if not subtitles:
            raise ValueError("subtitles must not be empty.")

        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "subtitles", subtitles)


class SubtitleMerger(Protocol):
    """Contract implemented by subtitle merge adapters."""

    def merge(self, request: SubtitleMergeRequest) -> SubtitleMerge:
        """Merge subtitle entries into normalized subtitle output."""


class SubtitleMergerStageError(RuntimeError):
    """Base error for subtitle merger stage failures."""


class InvalidSubtitleMergerError(SubtitleMergerStageError):
    """Raised when a configured merger does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__("Subtitle merger must define a callable merge method.")


class MissingSubtitlesToMergeError(SubtitleMergerStageError):
    """Raised when the document has no subtitles to merge."""

    def __init__(self) -> None:
        super().__init__("Subtitle merging requires existing document subtitles.")


class InvalidSubtitleMergeError(SubtitleMergerStageError):
    """Raised when a merger returns an invalid subtitle merge result."""


@dataclass(frozen=True, slots=True)
class SubtitleMergerStage:
    """Workflow stage that coordinates subtitle merging."""

    merger: SubtitleMerger
    name: str = SUBTITLE_MERGER_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.merger, "merge", None)):
            raise InvalidSubtitleMergerError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.subtitles:
            raise MissingSubtitlesToMergeError()

        request = SubtitleMergeRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
            subtitles=context.document.subtitles,
        )
        merge = self.merger.merge(request)
        if not isinstance(merge, SubtitleMerge):
            raise InvalidSubtitleMergeError(
                "Subtitle merger must return a SubtitleMerge."
            )

        if merge.source_path != request.source_path:
            raise InvalidSubtitleMergeError(
                "Subtitle merge source path must match the request source path."
            )

        document = Document(
            source_path=context.document.source_path,
            segments=context.document.segments,
            subtitles=merge.subtitles,
        )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=document,
            working_directory=context.working_directory,
        )

        return StageResult(stage_name=self.name, context=next_context)


__all__ = [
    "InvalidSubtitleMergeError",
    "InvalidSubtitleMergerError",
    "MissingSubtitlesToMergeError",
    "SUBTITLE_MERGER_STAGE_NAME",
    "SubtitleMerge",
    "SubtitleMergeRequest",
    "SubtitleMerger",
    "SubtitleMergerStage",
    "SubtitleMergerStageError",
]

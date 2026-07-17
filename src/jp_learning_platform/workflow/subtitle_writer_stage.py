"""Subtitle writer workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import PipelineContext, Segment, Subtitle
from jp_learning_platform.workflow.runtime import StageResult

SUBTITLE_WRITER_STAGE_NAME = "subtitle-writer"

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
class SubtitleWriteRequest:
    """Input passed from the workflow stage to a subtitle writer."""

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
class SubtitleWrite:
    """Normalized subtitle writer output."""

    source_path: Path
    output_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "output_path", Path(self.output_path))


class SubtitleWriter(Protocol):
    """Contract implemented by subtitle writer adapters."""

    def write(self, request: SubtitleWriteRequest) -> SubtitleWrite:
        """Write validated subtitles to an output artifact."""


class SubtitleWriterStageError(RuntimeError):
    """Base error for subtitle writer stage failures."""


class InvalidSubtitleWriterError(SubtitleWriterStageError):
    """Raised when a configured writer does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__("Subtitle writer must define a callable write method.")


class MissingSubtitlesToWriteError(SubtitleWriterStageError):
    """Raised when the document has no subtitles to write."""

    def __init__(self) -> None:
        super().__init__("Subtitle writing requires existing document subtitles.")


class InvalidSubtitleWriteError(SubtitleWriterStageError):
    """Raised when a writer returns an invalid write result."""


@dataclass(frozen=True, slots=True)
class SubtitleWriterStage:
    """Workflow stage that coordinates subtitle writing."""

    writer: SubtitleWriter
    name: str = SUBTITLE_WRITER_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.writer, "write", None)):
            raise InvalidSubtitleWriterError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.subtitles:
            raise MissingSubtitlesToWriteError()

        request = SubtitleWriteRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
            subtitles=context.document.subtitles,
        )
        write = self.writer.write(request)
        if not isinstance(write, SubtitleWrite):
            raise InvalidSubtitleWriteError(
                "Subtitle writer must return a SubtitleWrite."
            )

        if write.source_path != request.source_path:
            raise InvalidSubtitleWriteError(
                "Subtitle write source path must match the request source path."
            )

        return StageResult(stage_name=self.name, context=context)


__all__ = [
    "InvalidSubtitleWriteError",
    "InvalidSubtitleWriterError",
    "MissingSubtitlesToWriteError",
    "SUBTITLE_WRITER_STAGE_NAME",
    "SubtitleWrite",
    "SubtitleWriteRequest",
    "SubtitleWriter",
    "SubtitleWriterStage",
    "SubtitleWriterStageError",
]

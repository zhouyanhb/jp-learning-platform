"""WhisperX alignment workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment
from jp_learning_platform.workflow.runtime import StageResult

WHISPERX_ALIGNMENT_STAGE_NAME = "whisperx-alignment"

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
class WhisperXAlignmentRequest:
    """Input passed from the workflow stage to a WhisperX aligner."""

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
class WhisperXAlignment:
    """Normalized WhisperX alignment output."""

    source_path: Path
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        segments = _tuple_of_type(self.segments, Segment, "segments")
        if not segments:
            raise ValueError("segments must not be empty.")

        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "segments", segments)


class WhisperXAligner(Protocol):
    """Contract implemented by infrastructure or plugin WhisperX adapters."""

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        """Align transcribed segments into normalized domain segments."""


class WhisperXAlignmentStageError(RuntimeError):
    """Base error for WhisperX alignment stage failures."""


class InvalidWhisperXAlignerError(WhisperXAlignmentStageError):
    """Raised when a configured aligner does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__("WhisperX aligner must define a callable align method.")


class MissingWhisperSegmentsError(WhisperXAlignmentStageError):
    """Raised when the document has no Whisper segments to align."""

    def __init__(self) -> None:
        super().__init__("WhisperX alignment requires existing document segments.")


class InvalidWhisperXAlignmentError(WhisperXAlignmentStageError):
    """Raised when an aligner returns an invalid alignment."""


@dataclass(frozen=True, slots=True)
class WhisperXAlignmentStage:
    """Workflow stage that coordinates WhisperX alignment."""

    aligner: WhisperXAligner
    name: str = WHISPERX_ALIGNMENT_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.aligner, "align", None)):
            raise InvalidWhisperXAlignerError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.segments:
            raise MissingWhisperSegmentsError()

        request = WhisperXAlignmentRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
        )
        alignment = self.aligner.align(request)
        if not isinstance(alignment, WhisperXAlignment):
            raise InvalidWhisperXAlignmentError(
                "WhisperX aligner must return a WhisperXAlignment."
            )

        if alignment.source_path != request.source_path:
            raise InvalidWhisperXAlignmentError(
                "WhisperX alignment source path must match the request source path."
            )

        document = Document(
            source_path=context.document.source_path,
            segments=alignment.segments,
            subtitles=context.document.subtitles,
        )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=document,
            working_directory=context.working_directory,
        )

        return StageResult(stage_name=self.name, context=next_context)


__all__ = [
    "InvalidWhisperXAlignerError",
    "InvalidWhisperXAlignmentError",
    "MissingWhisperSegmentsError",
    "WHISPERX_ALIGNMENT_STAGE_NAME",
    "WhisperXAligner",
    "WhisperXAlignment",
    "WhisperXAlignmentRequest",
    "WhisperXAlignmentStage",
    "WhisperXAlignmentStageError",
]

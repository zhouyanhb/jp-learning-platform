"""Whisper transcription workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment
from jp_learning_platform.workflow.runtime import StageResult

WHISPER_STAGE_NAME = "whisper"

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
class WhisperTranscriptionRequest:
    """Input passed from the workflow stage to a Whisper transcriber."""

    source_path: Path
    working_directory: Path
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "working_directory",
            Path(self.working_directory),
        )
        object.__setattr__(self, "run_id", _normalize_name(self.run_id, "run_id"))


@dataclass(frozen=True, slots=True)
class WhisperTranscript:
    """Normalized Whisper transcription output."""

    source_path: Path
    segments: tuple[Segment, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "segments",
            _tuple_of_type(self.segments, Segment, "segments"),
        )


class WhisperTranscriber(Protocol):
    """Contract implemented by infrastructure or plugin Whisper adapters."""

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        """Transcribe a source audio file into normalized domain segments."""


class WhisperStageError(RuntimeError):
    """Base error for Whisper stage failures."""


class InvalidWhisperTranscriberError(WhisperStageError):
    """Raised when a configured transcriber does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__(
            "Whisper transcriber must define a callable transcribe method."
        )


class InvalidWhisperTranscriptError(WhisperStageError):
    """Raised when a transcriber returns an invalid transcript."""


@dataclass(frozen=True, slots=True)
class WhisperStage:
    """Workflow stage that coordinates Whisper transcription."""

    transcriber: WhisperTranscriber
    name: str = WHISPER_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.transcriber, "transcribe", None)):
            raise InvalidWhisperTranscriberError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        request = WhisperTranscriptionRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
        )
        transcript = self.transcriber.transcribe(request)
        if not isinstance(transcript, WhisperTranscript):
            raise InvalidWhisperTranscriptError(
                "Whisper transcriber must return a WhisperTranscript."
            )

        if transcript.source_path != request.source_path:
            raise InvalidWhisperTranscriptError(
                "Whisper transcript source path must match the request source path."
            )

        document = Document(
            source_path=context.document.source_path,
            segments=transcript.segments,
            subtitles=context.document.subtitles,
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
    "InvalidWhisperTranscriberError",
    "InvalidWhisperTranscriptError",
    "WHISPER_STAGE_NAME",
    "WhisperStage",
    "WhisperStageError",
    "WhisperTranscriber",
    "WhisperTranscript",
    "WhisperTranscriptionRequest",
]

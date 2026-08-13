"""Whisper transcription workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment, TimeRange
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
class WhisperRetryDecision:
    """Auditable acceptance decision for one bounded ASR retry candidate."""

    time_range: TimeRange
    accepted: bool
    reasons: tuple[str, ...]
    original_text: str
    candidate_text: str
    original_confidence: float | None
    candidate_confidence: float | None
    text_similarity: float
    original_character_coverage: float
    original_language_model_score: float | None
    candidate_language_model_score: float | None
    original_grammar_penalty: int
    candidate_grammar_penalty: int
    deletes_content_or_particle: bool
    passed_validation: bool = False
    selected: bool = False
    candidate_support_count: int = 1
    selection_score: float | None = None


@dataclass(frozen=True, slots=True)
class ShortAnomalyRetryAudit:
    """Trace one short malformed utterance through bounded ASR retries."""

    segment_position: int
    time_range: TimeRange
    original_text: str
    short_anomaly_detected: bool
    retry_attempted: bool
    short_window_candidate_texts: tuple[str, ...] = ()
    extracted_candidate_texts: tuple[str, ...] = ()
    left_anchor_matches: tuple[bool, ...] = ()
    right_anchor_matches: tuple[bool, ...] = ()
    anchor_orders_valid: tuple[bool, ...] = ()
    accepted: bool = False
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ShortUtteranceAnalysisAudit:
    """Record the morphological evidence used to classify one short ASR segment."""

    segment_position: int
    time_range: TimeRange
    original_text: str
    normalized_text: str
    morpheme_surfaces: tuple[str, ...]
    morpheme_part_of_speech: tuple[tuple[str, ...], ...]
    morpheme_conjugation_types: tuple[str, ...]
    structure_penalty: int
    short_anomaly_detected: bool


@dataclass(frozen=True, slots=True)
class WhisperTranscript:
    """Normalized Whisper transcription output."""

    source_path: Path
    segments: tuple[Segment, ...] = ()
    retry_decisions: tuple[WhisperRetryDecision, ...] = ()
    short_anomaly_retry_audits: tuple[ShortAnomalyRetryAudit, ...] = ()
    short_utterance_analysis_audits: tuple[ShortUtteranceAnalysisAudit, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "segments",
            _tuple_of_type(self.segments, Segment, "segments"),
        )
        object.__setattr__(
            self,
            "retry_decisions",
            _tuple_of_type(
                self.retry_decisions,
                WhisperRetryDecision,
                "retry_decisions",
            ),
        )
        object.__setattr__(
            self,
            "short_anomaly_retry_audits",
            _tuple_of_type(
                self.short_anomaly_retry_audits,
                ShortAnomalyRetryAudit,
                "short_anomaly_retry_audits",
            ),
        )
        object.__setattr__(
            self,
            "short_utterance_analysis_audits",
            _tuple_of_type(
                self.short_utterance_analysis_audits,
                ShortUtteranceAnalysisAudit,
                "short_utterance_analysis_audits",
            ),
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
        )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=document,
            working_directory=context.working_directory,
        )

        return StageResult(
            stage_name=self.name,
            context=next_context,
            data={
                "retry_decisions": transcript.retry_decisions,
                "short_anomaly_retry_audits": transcript.short_anomaly_retry_audits,
                "short_utterance_analysis_audits": (
                    transcript.short_utterance_analysis_audits
                ),
            },
        )


__all__ = [
    "InvalidWhisperTranscriberError",
    "InvalidWhisperTranscriptError",
    "WHISPER_STAGE_NAME",
    "WhisperStage",
    "WhisperStageError",
    "ShortAnomalyRetryAudit",
    "ShortUtteranceAnalysisAudit",
    "WhisperTranscriber",
    "WhisperTranscript",
    "WhisperRetryDecision",
    "WhisperTranscriptionRequest",
]

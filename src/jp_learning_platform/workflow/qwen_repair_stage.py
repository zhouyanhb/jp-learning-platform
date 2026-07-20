"""Qwen repair workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment
from jp_learning_platform.workflow.runtime import StageResult

QWEN_REPAIR_STAGE_NAME = "qwen-repair"

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
class QwenRepairRequest:
    """Input passed from the workflow stage to a Qwen repairer."""

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
class QwenRepairDecision:
    """Diagnostic record for one Qwen repair candidate and safety decision."""

    segment_position: int
    original_text: str
    raw_text: str
    candidate_text: str
    selected_text: str
    accepted: bool
    reason: str
    length_delta_ratio: float
    content_change_ratio: float

    def __post_init__(self) -> None:
        if isinstance(self.segment_position, bool) or not isinstance(
            self.segment_position,
            int,
        ):
            raise TypeError("segment_position must be an integer.")

        if self.segment_position < 0:
            raise ValueError("segment_position must be non-negative.")

        for field_name in (
            "original_text",
            "raw_text",
            "candidate_text",
            "selected_text",
            "reason",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string.")

        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool.")

        object.__setattr__(
            self,
            "length_delta_ratio",
            float(self.length_delta_ratio),
        )
        object.__setattr__(
            self,
            "content_change_ratio",
            float(self.content_change_ratio),
        )


@dataclass(frozen=True, slots=True)
class QwenRepair:
    """Normalized Qwen repair output."""

    source_path: Path
    segments: tuple[Segment, ...]
    decisions: tuple[QwenRepairDecision, ...] = ()

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
                QwenRepairDecision,
                "decisions",
            ),
        )


class QwenRepairer(Protocol):
    """Contract implemented by infrastructure or plugin Qwen adapters."""

    def repair(self, request: QwenRepairRequest) -> QwenRepair:
        """Repair aligned transcription segments into normalized domain segments."""


class QwenRepairStageError(RuntimeError):
    """Base error for Qwen repair stage failures."""


class InvalidQwenRepairerError(QwenRepairStageError):
    """Raised when a configured repairer does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__("Qwen repairer must define a callable repair method.")


class MissingAlignedSegmentsError(QwenRepairStageError):
    """Raised when the document has no aligned segments to repair."""

    def __init__(self) -> None:
        super().__init__("Qwen repair requires existing aligned document segments.")


class InvalidQwenRepairError(QwenRepairStageError):
    """Raised when a repairer returns an invalid repair result."""


@dataclass(frozen=True, slots=True)
class QwenRepairStage:
    """Workflow stage that coordinates Qwen transcript repair."""

    repairer: QwenRepairer
    name: str = QWEN_REPAIR_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.repairer, "repair", None)):
            raise InvalidQwenRepairerError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.segments:
            raise MissingAlignedSegmentsError()

        request = QwenRepairRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
        )
        repair = self.repairer.repair(request)
        if not isinstance(repair, QwenRepair):
            raise InvalidQwenRepairError(
                "Qwen repairer must return a QwenRepair."
            )

        if repair.source_path != request.source_path:
            raise InvalidQwenRepairError(
                "Qwen repair source path must match the request source path."
            )

        document = Document(
            source_path=context.document.source_path,
            segments=repair.segments,
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

        return StageResult(
            stage_name=self.name,
            context=next_context,
            data={"decisions": repair.decisions},
        )


__all__ = [
    "InvalidQwenRepairError",
    "InvalidQwenRepairerError",
    "MissingAlignedSegmentsError",
    "QWEN_REPAIR_STAGE_NAME",
    "QwenRepair",
    "QwenRepairDecision",
    "QwenRepairRequest",
    "QwenRepairStage",
    "QwenRepairStageError",
    "QwenRepairer",
]

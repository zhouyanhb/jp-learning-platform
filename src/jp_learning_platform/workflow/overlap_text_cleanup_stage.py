"""Auditable cleanup of proven cross-segment ASR overlap."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment, TimeRange
from jp_learning_platform.workflow.runtime import StageResult

OVERLAP_TEXT_CLEANUP_STAGE_NAME = "overlap-text-cleanup"
T = TypeVar("T")


def _items(values: Iterable[T], item_type: type[T], name: str) -> tuple[T, ...]:
    result = tuple(values)
    if any(not isinstance(value, item_type) for value in result):
        raise TypeError(f"{name} must contain {item_type.__name__} values.")
    return result


@dataclass(frozen=True, slots=True)
class OverlapTextCleanupRequest:
    source_path: Path
    working_directory: Path
    run_id: str
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "working_directory", Path(self.working_directory))
        object.__setattr__(self, "segments", _items(self.segments, Segment, "segments"))


@dataclass(frozen=True, slots=True)
class OverlapTextCleanupDecision:
    previous_segment_position: int
    segment_position: int
    original_text: str
    deleted_text: str
    deletion_start: int
    deletion_end: int
    deleted_time_range: TimeRange
    time_overlap_seconds: float
    boundary_gap_seconds: float
    reason: str
    evidence: tuple[str, ...] = ()
    transferred_punctuation_text: str = ""
    transferred_word_indexes: tuple[int, ...] = ()
    transferred_time_range: TimeRange | None = None


@dataclass(frozen=True, slots=True)
class OverlapTextCleanup:
    source_path: Path
    segments: tuple[Segment, ...]
    decisions: tuple[OverlapTextCleanupDecision, ...] = ()


class OverlapTextCleaner(Protocol):
    def clean(self, request: OverlapTextCleanupRequest) -> OverlapTextCleanup: ...


@dataclass(frozen=True, slots=True)
class OverlapTextCleanupStage:
    cleaner: OverlapTextCleaner
    name: str = OVERLAP_TEXT_CLEANUP_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.cleaner, "clean", None)):
            raise TypeError("Overlap text cleaner must define a callable clean method.")

    def run(self, context: PipelineContext) -> StageResult:
        if not context.document.segments:
            raise ValueError("Overlap text cleanup requires existing segments.")
        request = OverlapTextCleanupRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
        )
        cleanup = self.cleaner.clean(request)
        if not isinstance(cleanup, OverlapTextCleanup):
            raise TypeError("Overlap text cleaner must return OverlapTextCleanup.")
        if cleanup.source_path != request.source_path:
            raise ValueError("Overlap text cleanup source path must match the request.")
        next_context = PipelineContext(
            run_id=context.run_id,
            document=Document(
                source_path=context.document.source_path,
                segments=cleanup.segments,
                subtitles=context.document.subtitles,
            ),
            working_directory=context.working_directory,
        )
        return StageResult(
            stage_name=self.name,
            context=next_context,
            data={"decisions": cleanup.decisions},
        )


__all__ = [
    "OVERLAP_TEXT_CLEANUP_STAGE_NAME",
    "OverlapTextCleaner",
    "OverlapTextCleanup",
    "OverlapTextCleanupDecision",
    "OverlapTextCleanupRequest",
    "OverlapTextCleanupStage",
]

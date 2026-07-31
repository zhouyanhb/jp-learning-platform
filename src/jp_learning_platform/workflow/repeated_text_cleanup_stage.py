"""Auditable cleanup of immediate repeated text inside ASR segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jp_learning_platform.domain import Document, PipelineContext, Segment, TimeRange
from jp_learning_platform.workflow.runtime import StageResult

REPEATED_TEXT_CLEANUP_STAGE_NAME = "repeated-text-cleanup"


@dataclass(frozen=True, slots=True)
class RepeatedTextCleanupRequest:
    source_path: Path
    segments: tuple[Segment, ...]


@dataclass(frozen=True, slots=True)
class RepeatedTextCleanupDecision:
    segment_position: int
    sentence_index: int
    original_text: str
    deleted_text: str
    deletion_start: int
    deletion_end: int
    deleted_word_indexes: tuple[int, ...]
    retained_time_range: TimeRange
    deleted_time_range: TimeRange
    repetition_gap_seconds: float
    reason: str = "adjacent_exact_aligned_word_sequence"


@dataclass(frozen=True, slots=True)
class RepeatedTextCleanup:
    source_path: Path
    segments: tuple[Segment, ...]
    decisions: tuple[RepeatedTextCleanupDecision, ...] = ()


class RepeatedTextCleaner(Protocol):
    def clean(self, request: RepeatedTextCleanupRequest) -> RepeatedTextCleanup: ...


@dataclass(frozen=True, slots=True)
class RepeatedTextCleanupStage:
    cleaner: RepeatedTextCleaner
    name: str = REPEATED_TEXT_CLEANUP_STAGE_NAME

    def run(self, context: PipelineContext) -> StageResult:
        result = self.cleaner.clean(
            RepeatedTextCleanupRequest(
                source_path=context.document.source_path,
                segments=context.document.segments,
            )
        )
        if not isinstance(result, RepeatedTextCleanup):
            raise TypeError("Repeated text cleaner must return RepeatedTextCleanup.")
        next_context = PipelineContext(
            run_id=context.run_id,
            document=Document(
                source_path=context.document.source_path,
                segments=result.segments,
                subtitles=context.document.subtitles,
            ),
            working_directory=context.working_directory,
        )
        return StageResult(
            stage_name=self.name,
            context=next_context,
            data={"decisions": result.decisions},
        )


__all__ = [
    "REPEATED_TEXT_CLEANUP_STAGE_NAME",
    "RepeatedTextCleaner",
    "RepeatedTextCleanup",
    "RepeatedTextCleanupDecision",
    "RepeatedTextCleanupRequest",
    "RepeatedTextCleanupStage",
]

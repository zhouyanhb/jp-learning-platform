"""Local subtitle quality adapters."""

from __future__ import annotations

from dataclasses import dataclass
import re

from jp_learning_platform.domain import (
    Document,
    DocumentValidator,
    Subtitle,
    TimeRange,
)
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_READABILITY_CONFIG,
    DEFAULT_SUBTITLE_MERGE_CONFIG,
)
from jp_learning_platform.infrastructure.japanese_text import (
    JAPANESE_TERMINAL_MARKS,
    text_ends_with_complete_predicate,
    text_ends_with_terminal_sentence_mark,
)
from jp_learning_platform.workflow.readability_optimizer_stage import (
    ReadabilityOptimization,
    ReadabilityOptimizationRequest,
)
from jp_learning_platform.workflow.subtitle_merger_stage import (
    SubtitleMerge,
    SubtitleMergeRequest,
)
from jp_learning_platform.workflow.subtitle_validator_stage import (
    SubtitleValidation,
    SubtitleValidationRequest,
)

DEFAULT_MERGE_GAP_SECONDS = DEFAULT_SUBTITLE_MERGE_CONFIG.max_gap_seconds
DEFAULT_MERGE_MAX_CHARS = DEFAULT_SUBTITLE_MERGE_CONFIG.max_chars


@dataclass(frozen=True, slots=True)
class ConservativeSubtitleMerger:
    """Merge only short adjacent subtitles that look like one cue."""

    max_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS
    max_chars: int = DEFAULT_MERGE_MAX_CHARS

    def merge(self, request: SubtitleMergeRequest) -> SubtitleMerge:
        if not isinstance(request, SubtitleMergeRequest):
            raise TypeError("request must be a SubtitleMergeRequest.")

        merged: list[Subtitle] = []
        index = 0
        while index < len(request.subtitles):
            current = request.subtitles[index]
            if index + 1 >= len(request.subtitles):
                merged.append(current)
                break

            nxt = request.subtitles[index + 1]
            if self._should_merge(current, nxt):
                merged.append(self._merge_pair(current, nxt))
                index += 2
                continue

            merged.append(current)
            index += 1

        return SubtitleMerge(
            source_path=request.source_path,
            subtitles=_reindex(merged),
        )

    def _should_merge(self, current: Subtitle, nxt: Subtitle) -> bool:
        gap = nxt.time_range.start_seconds - current.time_range.end_seconds
        combined_text = current.text + nxt.text
        return (
            0 <= gap <= self.max_gap_seconds
            and len(combined_text) <= self.max_chars
            and not _ends_with_terminal_or_complete_sentence(current.text)
            and not _speaker_boundary(current, nxt)
        )

    def _merge_pair(self, current: Subtitle, nxt: Subtitle) -> Subtitle:
        return Subtitle(
            index=current.index,
            text=current.text + nxt.text,
            time_range=TimeRange(
                current.time_range.start_seconds,
                nxt.time_range.end_seconds,
            ),
            speaker_id=current.speaker_id,
        )


@dataclass(frozen=True, slots=True)
class LocalReadabilityOptimizer:
    """Normalize punctuation and spacing for Japanese subtitle text."""

    def optimize(
        self,
        request: ReadabilityOptimizationRequest,
    ) -> ReadabilityOptimization:
        if not isinstance(request, ReadabilityOptimizationRequest):
            raise TypeError("request must be a ReadabilityOptimizationRequest.")

        comma = DEFAULT_READABILITY_CONFIG.japanese_comma
        period = DEFAULT_READABILITY_CONFIG.japanese_period
        subtitles = tuple(
            Subtitle(
                index=subtitle.index,
                text=self._normalize_text(subtitle.text, comma, period),
                time_range=subtitle.time_range,
                speaker_id=subtitle.speaker_id,
            )
            for subtitle in request.subtitles
        )
        return ReadabilityOptimization(
            source_path=request.source_path,
            subtitles=_resolve_subtitle_overlaps(subtitles),
        )

    def _normalize_text(self, text: str, comma: str, period: str) -> str:
        normalized = text.strip()
        normalized = normalized.replace(",", comma)
        normalized = normalized.replace("，", comma)
        normalized = normalized.replace(".", period)
        normalized = normalized.replace("．", period)
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(rf"{re.escape(period)}{{2,}}", period, normalized)
        normalized = re.sub(rf"{re.escape(comma)}{{2,}}", comma, normalized)
        normalized = re.sub(r"？{2,}", "？", normalized)
        normalized = re.sub(r"！{2,}", "！", normalized)
        return normalized


@dataclass(frozen=True, slots=True)
class DomainSubtitleValidator:
    """Validate final subtitles with the domain document validator."""

    validator: DocumentValidator = DocumentValidator()

    def validate(self, request: SubtitleValidationRequest) -> SubtitleValidation:
        if not isinstance(request, SubtitleValidationRequest):
            raise TypeError("request must be a SubtitleValidationRequest.")

        result = self.validator.validate(
            Document(
                source_path=request.source_path,
                segments=request.segments,
                subtitles=request.subtitles,
            )
        )
        return SubtitleValidation(
            source_path=request.source_path,
            result=result,
        )


def _reindex(subtitles: list[Subtitle]) -> tuple[Subtitle, ...]:
    return tuple(
        Subtitle(
            index=index,
            text=subtitle.text,
            time_range=subtitle.time_range,
            speaker_id=subtitle.speaker_id,
        )
        for index, subtitle in enumerate(subtitles, start=1)
    )


def _resolve_subtitle_overlaps(
    subtitles: tuple[Subtitle, ...],
) -> tuple[Subtitle, ...]:
    if not subtitles:
        return ()

    resolved: list[Subtitle] = [subtitles[0]]
    for subtitle in subtitles[1:]:
        previous = resolved[-1]
        if subtitle.time_range.start_seconds < previous.time_range.end_seconds:
            previous, subtitle = _split_subtitle_overlap(previous, subtitle)
            resolved[-1] = previous

        resolved.append(subtitle)

    return tuple(resolved)


def _split_subtitle_overlap(
    previous: Subtitle,
    current: Subtitle,
) -> tuple[Subtitle, Subtitle]:
    midpoint_seconds = (
        current.time_range.start_seconds + previous.time_range.end_seconds
    ) / 2.0
    boundary_seconds = min(
        current.time_range.end_seconds,
        max(previous.time_range.start_seconds, midpoint_seconds),
    )
    return (
        _subtitle_with_time_range(
            previous,
            TimeRange(
                previous.time_range.start_seconds,
                boundary_seconds,
            ),
        ),
        _subtitle_with_time_range(
            current,
            TimeRange(
                boundary_seconds,
                current.time_range.end_seconds,
            ),
        ),
    )


def _subtitle_with_time_range(
    subtitle: Subtitle,
    time_range: TimeRange,
) -> Subtitle:
    return Subtitle(
        index=subtitle.index,
        text=subtitle.text,
        time_range=time_range,
        speaker_id=subtitle.speaker_id,
    )


def _speaker_boundary(current: Subtitle, nxt: Subtitle) -> bool:
    if current.speaker_id is None and nxt.speaker_id is None:
        return False

    return current.speaker_id != nxt.speaker_id


def _ends_with_terminal_or_complete_sentence(text: str) -> bool:
    return text_ends_with_terminal_sentence_mark(
        text
    ) or text_ends_with_complete_predicate(text)


__all__ = [
    "ConservativeSubtitleMerger",
    "DEFAULT_MERGE_GAP_SECONDS",
    "DEFAULT_MERGE_MAX_CHARS",
    "DomainSubtitleValidator",
    "LocalReadabilityOptimizer",
]

"""Factories for constructing domain models from primitive values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from jp_learning_platform.domain.models import (
    Document,
    LearningWord,
    PipelineContext,
    Segment,
    Sentence,
    Subtitle,
    TimeRange,
    Word,
)


@dataclass(frozen=True, slots=True)
class DomainModelFactory:
    """Create immutable subtitle pipeline domain models."""

    def create_time_range(self, start_seconds: float, end_seconds: float) -> TimeRange:
        return TimeRange(start_seconds=start_seconds, end_seconds=end_seconds)

    def create_word(
        self,
        text: str,
        start_seconds: float,
        end_seconds: float,
        confidence: float | None = None,
    ) -> Word:
        return Word(
            text=text,
            time_range=self.create_time_range(start_seconds, end_seconds),
            confidence=confidence,
        )

    def create_sentence(
        self,
        text: str,
        start_seconds: float,
        end_seconds: float,
        words: Iterable[Word] = (),
        learning_words: Iterable[LearningWord] = (),
        is_question: bool = False,
        asr_boundary_word_indexes: Iterable[int] = (),
    ) -> Sentence:
        return Sentence(
            text=text,
            time_range=self.create_time_range(start_seconds, end_seconds),
            words=tuple(words),
            learning_words=tuple(learning_words),
            is_question=is_question,
            asr_boundary_word_indexes=tuple(asr_boundary_word_indexes),
        )

    def create_segment(
        self,
        position: int,
        text: str,
        start_seconds: float,
        end_seconds: float,
        sentences: Iterable[Sentence] = (),
    ) -> Segment:
        return Segment(
            position=position,
            text=text,
            time_range=self.create_time_range(start_seconds, end_seconds),
            sentences=tuple(sentences),
        )

    def create_subtitle(
        self,
        index: int,
        text: str,
        start_seconds: float,
        end_seconds: float,
        source_sentence_index: int | None = None,
    ) -> Subtitle:
        return Subtitle(
            index=index,
            text=text,
            time_range=self.create_time_range(start_seconds, end_seconds),
            source_sentence_index=source_sentence_index,
        )

    def create_document(
        self,
        source_path: Path,
        segments: Iterable[Segment] = (),
        subtitles: Iterable[Subtitle] = (),
    ) -> Document:
        return Document(
            source_path=source_path,
            segments=tuple(segments),
            subtitles=tuple(subtitles),
        )

    def create_pipeline_context(
        self,
        run_id: str,
        document: Document,
        working_directory: Path,
    ) -> PipelineContext:
        return PipelineContext(
            run_id=run_id,
            document=document,
            working_directory=working_directory,
        )


__all__ = ["DomainModelFactory"]

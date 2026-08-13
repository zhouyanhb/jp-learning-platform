"""Assign standalone trailing punctuation to its preceding sentence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import unicodedata

from jp_learning_platform.domain import Document, PipelineContext, Segment, Sentence, TimeRange
from jp_learning_platform.workflow.runtime import StageResult

PUNCTUATION_ATTRIBUTION_STAGE_NAME = "punctuation-attribution"


@dataclass(frozen=True, slots=True)
class PunctuationAttributionDecision:
    segment_position: int
    sentence_index: int
    attributed_text: str
    original_sentence_text: str
    resulting_sentence_text: str
    is_question: bool
    reason: str = "standalone_trailing_punctuation"

    @property
    def original_question_text(self) -> str:
        """Return the former field name for callers migrating old artifacts."""
        return self.original_sentence_text

    @property
    def resulting_question_text(self) -> str:
        """Return the former field name for callers migrating old artifacts."""
        return self.resulting_sentence_text


@dataclass(frozen=True, slots=True)
class InternalPunctuationRestoration:
    sentence: Sentence
    attributed_text: str
    original_sentence_text: str
    reason: str


class InternalPunctuationRestorer(Protocol):
    def restore(self, sentence: Sentence) -> InternalPunctuationRestoration | None: ...


@dataclass(frozen=True, slots=True)
class PunctuationAttributionStage:
    internal_restorer: InternalPunctuationRestorer | None = None
    name: str = PUNCTUATION_ATTRIBUTION_STAGE_NAME

    def run(self, context: PipelineContext) -> StageResult:
        segments: list[Segment] = []
        decisions: list[PunctuationAttributionDecision] = []
        for segment in context.document.segments:
            sentences: list[Sentence] = []
            for sentence_index, sentence in enumerate(segment.sentences):
                if (
                    sentences
                    and _is_trailing_punctuation_only(sentence.text)
                ):
                    previous = sentences.pop()
                    merged = Sentence(
                        text=f"{previous.text}{sentence.text}",
                        time_range=TimeRange(
                            previous.time_range.start_seconds,
                            max(
                                previous.time_range.end_seconds,
                                sentence.time_range.end_seconds,
                            ),
                        ),
                        words=(*previous.words, *sentence.words),
                        is_question=previous.is_question,
                        asr_boundary_word_indexes=(
                            *previous.asr_boundary_word_indexes,
                            len(previous.words),
                            *(
                                len(previous.words) + index
                                for index in sentence.asr_boundary_word_indexes
                            ),
                        ),
                    )
                    sentences.append(merged)
                    decisions.append(
                        PunctuationAttributionDecision(
                            segment_position=segment.position,
                            sentence_index=sentence_index,
                            attributed_text=sentence.text,
                            original_sentence_text=previous.text,
                            resulting_sentence_text=merged.text,
                            is_question=previous.is_question,
                        )
                    )
                else:
                    sentences.append(sentence)
                if self.internal_restorer is not None:
                    current = sentences[-1]
                    restoration = self.internal_restorer.restore(current)
                    if restoration is not None:
                        sentences[-1] = restoration.sentence
                        decisions.append(
                            PunctuationAttributionDecision(
                                segment_position=segment.position,
                                sentence_index=sentence_index,
                                attributed_text=restoration.attributed_text,
                                original_sentence_text=(
                                    restoration.original_sentence_text
                                ),
                                resulting_sentence_text=restoration.sentence.text,
                                is_question=restoration.sentence.is_question,
                                reason=restoration.reason,
                            )
                        )
            segments.append(
                Segment(
                    position=segment.position,
                    text="".join(sentence.text for sentence in sentences),
                    time_range=segment.time_range,
                    sentences=tuple(sentences),
                )
            )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=Document(
                source_path=context.document.source_path,
                segments=tuple(segments),
                subtitles=context.document.subtitles,
            ),
            working_directory=context.working_directory,
        )
        return StageResult(
            stage_name=self.name,
            context=next_context,
            data={"decisions": tuple(decisions)},
        )


def _is_trailing_punctuation_only(text: str) -> bool:
    characters = tuple(character for character in text if not character.isspace())
    return (
        bool(characters)
        and all(
            unicodedata.category(character).startswith("P")
            for character in characters
        )
        and unicodedata.category(characters[0]) not in {"Ps", "Pi"}
    )


__all__ = [
    "InternalPunctuationRestoration",
    "InternalPunctuationRestorer",
    "PUNCTUATION_ATTRIBUTION_STAGE_NAME",
    "PunctuationAttributionDecision",
    "PunctuationAttributionStage",
]

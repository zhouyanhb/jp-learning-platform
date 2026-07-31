"""Assign standalone punctuation to its question sentence without changing words."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata

from jp_learning_platform.domain import Document, PipelineContext, Segment, Sentence, TimeRange
from jp_learning_platform.workflow.runtime import StageResult

PUNCTUATION_ATTRIBUTION_STAGE_NAME = "punctuation-attribution"


@dataclass(frozen=True, slots=True)
class PunctuationAttributionDecision:
    segment_position: int
    sentence_index: int
    attributed_text: str
    original_question_text: str
    resulting_question_text: str
    reason: str = "standalone_punctuation_after_question"


@dataclass(frozen=True, slots=True)
class PunctuationAttributionStage:
    name: str = PUNCTUATION_ATTRIBUTION_STAGE_NAME

    def run(self, context: PipelineContext) -> StageResult:
        segments: list[Segment] = []
        decisions: list[PunctuationAttributionDecision] = []
        for segment in context.document.segments:
            sentences: list[Sentence] = []
            for sentence_index, sentence in enumerate(segment.sentences):
                if (
                    sentences
                    and sentences[-1].is_question
                    and _is_punctuation_only(sentence.text)
                ):
                    previous = sentences.pop()
                    merged = Sentence(
                        text=f"{previous.text}{sentence.text}",
                        time_range=TimeRange(
                            previous.time_range.start_seconds,
                            sentence.time_range.end_seconds,
                        ),
                        words=(*previous.words, *sentence.words),
                        is_question=True,
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
                            original_question_text=previous.text,
                            resulting_question_text=merged.text,
                        )
                    )
                else:
                    sentences.append(sentence)
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


def _is_punctuation_only(text: str) -> bool:
    characters = tuple(character for character in text if not character.isspace())
    return bool(characters) and all(
        unicodedata.category(character).startswith("P") for character in characters
    )


__all__ = [
    "PUNCTUATION_ATTRIBUTION_STAGE_NAME",
    "PunctuationAttributionDecision",
    "PunctuationAttributionStage",
]

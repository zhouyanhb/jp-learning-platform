"""Normalize question punctuation in subtitle display text with an audit trail."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from jp_learning_platform.domain import Document, PipelineContext, Subtitle
from jp_learning_platform.workflow.runtime import StageResult

SUBTITLE_DISPLAY_NORMALIZATION_STAGE_NAME = "subtitle-display-normalization"


@dataclass(frozen=True, slots=True)
class SubtitleDisplayNormalizationDecision:
    subtitle_index: int
    original_text: str
    display_text: str
    original_terminal: str
    display_terminal: str
    reason: str = "question_terminal_normalization"


@dataclass(frozen=True, slots=True)
class SubtitleDisplayNormalizationStage:
    name: str = SUBTITLE_DISPLAY_NORMALIZATION_STAGE_NAME

    def run(self, context: PipelineContext) -> StageResult:
        sentences = tuple(
            sentence
            for segment in context.document.segments
            for sentence in segment.sentences
        )
        subtitles: list[Subtitle] = []
        decisions: list[SubtitleDisplayNormalizationDecision] = []
        sentence_index = 0
        sentence_cursor = 0
        for subtitle in context.document.subtitles:
            if sentence_index >= len(sentences):
                raise ValueError("Display subtitle does not belong to a sentence.")
            sentence = sentences[sentence_index]
            sentence_text = _compact(sentence.text)
            subtitle_text = _compact(subtitle.text)
            end_cursor = sentence_cursor + len(subtitle_text)
            if sentence_text[sentence_cursor:end_cursor] != subtitle_text:
                raise ValueError("Display subtitle text does not match its sentence.")
            is_sentence_final_cue = end_cursor == len(sentence_text)
            display_text, original_terminal = _question_display_text(
                subtitle.text, sentence.is_question and is_sentence_final_cue
            )
            subtitles.append(
                Subtitle(
                    index=subtitle.index,
                    text=display_text,
                    time_range=subtitle.time_range,
                    source_sentence_index=subtitle.source_sentence_index,
                )
            )
            if display_text != subtitle.text:
                decisions.append(
                    SubtitleDisplayNormalizationDecision(
                        subtitle_index=subtitle.index,
                        original_text=subtitle.text,
                        display_text=display_text,
                        original_terminal=original_terminal,
                        display_terminal="？",
                    )
                )
            if is_sentence_final_cue:
                sentence_index += 1
                sentence_cursor = 0
            else:
                sentence_cursor = end_cursor
        if sentence_index != len(sentences):
            raise ValueError("Not every sentence has display subtitles.")
        next_context = PipelineContext(
            run_id=context.run_id,
            document=Document(
                source_path=context.document.source_path,
                segments=context.document.segments,
                subtitles=tuple(subtitles),
            ),
            working_directory=context.working_directory,
        )
        return StageResult(
            stage_name=self.name,
            context=next_context,
            data={"decisions": tuple(decisions)},
        )


def _question_display_text(text: str, is_question: bool) -> tuple[str, str]:
    if not is_question:
        return text, ""
    characters = list(text)
    index = len(characters) - 1
    while index >= 0 and unicodedata.category(characters[index]) in {"Pe", "Pf"}:
        index -= 1
    if index >= 0 and characters[index] in {"?", "？"}:
        return text, characters[index]
    original = characters[index] if index >= 0 else ""
    if index >= 0 and unicodedata.category(characters[index]).startswith("P"):
        characters[index] = "？"
    else:
        characters.insert(index + 1, "？")
    return "".join(characters), original


def _compact(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


__all__ = [
    "SUBTITLE_DISPLAY_NORMALIZATION_STAGE_NAME",
    "SubtitleDisplayNormalizationDecision",
    "SubtitleDisplayNormalizationStage",
]

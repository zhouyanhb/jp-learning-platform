"""Word-aware subtitle builder adapter."""

from __future__ import annotations

from dataclasses import dataclass

from jp_learning_platform.domain import LearningWord, Sentence, Subtitle, TimeRange
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_SUBTITLE_DISPLAY_CONFIG,
    SubtitleDisplayConfig,
)
from jp_learning_platform.workflow.subtitle_builder_stage import (
    SubtitleBuild,
    SubtitleBuildRequest,
)


@dataclass(frozen=True, slots=True)
class WordSubtitleBuilder:
    """Build subtitle cues from segment sentences while preserving word timing."""

    config: SubtitleDisplayConfig = DEFAULT_SUBTITLE_DISPLAY_CONFIG

    def build(self, request: SubtitleBuildRequest) -> SubtitleBuild:
        if not isinstance(request, SubtitleBuildRequest):
            raise TypeError("request must be a SubtitleBuildRequest.")

        subtitles: list[Subtitle] = []
        source_sentence_index = 0
        for segment in request.segments:
            sentences = segment.sentences or (
                Sentence(
                    text=segment.text,
                    time_range=segment.time_range,
                    words=(),
                ),
            )
            for sentence in sentences:
                for text, time_range in self._display_parts(sentence):
                    subtitles.append(
                        Subtitle(
                            index=len(subtitles) + 1,
                            text=text,
                            time_range=time_range,
                            source_sentence_index=source_sentence_index,
                        )
                    )
                source_sentence_index += 1

        return SubtitleBuild(
            source_path=request.source_path,
            subtitles=tuple(subtitles),
        )

    def _display_parts(
        self, sentence: Sentence
    ) -> tuple[tuple[str, TimeRange], ...]:
        if (
            len(sentence.text) <= self.config.max_chars
            and sentence.time_range.duration_seconds
            <= self.config.max_duration_seconds
        ):
            return ((sentence.text, sentence.time_range),)

        boundaries = _timed_boundaries(sentence)
        if len(boundaries) < 2:
            return ((sentence.text, sentence.time_range),)

        parts: list[tuple[str, TimeRange]] = []
        start_index = 0
        while start_index < len(boundaries) - 1:
            end_index = _furthest_display_boundary(
                boundaries,
                start_index,
                self.config.max_chars,
                self.config.max_duration_seconds,
            )
            if end_index == start_index:
                end_index += 1
            start_char, start_time = boundaries[start_index]
            end_char, end_time = boundaries[end_index]
            display_end_time = min(
                end_time,
                start_time + self.config.max_duration_seconds,
            )
            parts.append(
                (
                    sentence.text[start_char:end_char],
                    TimeRange(start_time, display_end_time),
                )
            )
            start_index = end_index
        return tuple(parts)


def _timed_boundaries(sentence: Sentence) -> tuple[tuple[int, float], ...]:
    units = sentence.learning_words
    if not units:
        units = _aligned_learning_units(sentence)
    if not units:
        return ()

    raw_offsets = _raw_offsets(sentence.text)
    if units[-1].end_char >= len(raw_offsets):
        return ()
    boundaries: list[tuple[int, float]] = [
        (0, sentence.time_range.start_seconds)
    ]
    for index, unit in enumerate(units):
        if index + 1 < len(units):
            following = units[index + 1]
            time = (
                unit.time_range.end_seconds
                + following.time_range.start_seconds
            ) / 2
        else:
            time = sentence.time_range.end_seconds
        boundaries.append((raw_offsets[unit.end_char], time))
    if boundaries[-1][0] != len(sentence.text):
        return ()
    return tuple(boundaries)


def _raw_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    for index, character in enumerate(text, start=1):
        if not character.isspace():
            offsets.append(index)
    if offsets[-1] != len(text):
        offsets[-1] = len(text)
    return tuple(offsets)


def _aligned_learning_units(sentence: Sentence) -> tuple[LearningWord, ...]:
    units: list[LearningWord] = []
    cursor = 0
    for index, word in enumerate(sentence.words):
        text = word.text.replace(" ", "")
        if not text:
            continue
        end = cursor + len(text)
        units.append(
            LearningWord(
                text=text,
                start_char=cursor,
                end_char=end,
                aligned_word_indexes=(index,),
                time_range=word.time_range,
            )
        )
        cursor = end
    return tuple(units) if cursor == len(sentence.text) else ()


def _furthest_display_boundary(
    boundaries: tuple[tuple[int, float], ...],
    start_index: int,
    max_chars: int,
    max_duration_seconds: float,
) -> int:
    start_char, start_time = boundaries[start_index]
    selected = start_index
    for index in range(start_index + 1, len(boundaries)):
        end_char, end_time = boundaries[index]
        if (
            end_char - start_char > max_chars
            or end_time - start_time > max_duration_seconds
        ):
            break
        selected = index
    return selected


__all__ = ["WordSubtitleBuilder"]

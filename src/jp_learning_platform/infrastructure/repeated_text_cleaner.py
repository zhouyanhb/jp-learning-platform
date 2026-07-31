"""Conservative aligned-word repetition cleanup inside ASR segments."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseMorphologicalAnalyzer,
)
from jp_learning_platform.workflow.repeated_text_cleanup_stage import (
    RepeatedTextCleanup,
    RepeatedTextCleanupDecision,
    RepeatedTextCleanupRequest,
)


@dataclass(frozen=True, slots=True)
class AuditableRepeatedTextCleaner:
    min_duplicate_characters: int = 2
    max_sequence_words: int = 12
    max_repetition_gap_seconds: float = 0.1
    morphological_analyzer: JapaneseMorphologicalAnalyzer | None = None

    def clean(self, request: RepeatedTextCleanupRequest) -> RepeatedTextCleanup:
        segments: list[Segment] = []
        decisions: list[RepeatedTextCleanupDecision] = []
        for segment in request.segments:
            sentences: list[Sentence] = []
            for sentence_index, sentence in enumerate(segment.sentences):
                cleaned, sentence_decisions = self._clean_sentence(
                    segment.position, sentence_index, sentence
                )
                sentences.append(cleaned)
                decisions.extend(sentence_decisions)
            segments.append(
                Segment(
                    position=segment.position,
                    text="".join(sentence.text for sentence in sentences),
                    time_range=segment.time_range,
                    sentences=tuple(sentences),
                )
            )
        return RepeatedTextCleanup(
            request.source_path, tuple(segments), tuple(decisions)
        )

    def _clean_sentence(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
    ) -> tuple[Sentence, tuple[RepeatedTextCleanupDecision, ...]]:
        words = list(sentence.words)
        original_word_indexes = list(range(len(words)))
        original_words = tuple(words)
        decisions: list[RepeatedTextCleanupDecision] = []
        cursor = 0
        original_text = _words_text(tuple(words))
        while cursor < len(words):
            size = _longest_repeated_sequence(
                words,
                cursor,
                self.min_duplicate_characters,
                self.max_sequence_words,
                self.max_repetition_gap_seconds,
                self.morphological_analyzer,
            )
            if size == 0:
                cursor += 1
                continue
            first = tuple(words[cursor : cursor + size])
            second = tuple(words[cursor + size : cursor + (2 * size)])
            deleted_indexes = tuple(
                original_word_indexes[cursor + size : cursor + (2 * size)]
            )
            deletion_start = len(_words_text(original_words[: deleted_indexes[0]]))
            deleted_text = _words_text(second)
            decisions.append(
                RepeatedTextCleanupDecision(
                    segment_position=segment_position,
                    sentence_index=sentence_index,
                    original_text=original_text,
                    deleted_text=deleted_text,
                    deletion_start=deletion_start,
                    deletion_end=deletion_start + len(deleted_text),
                    deleted_word_indexes=deleted_indexes,
                    retained_time_range=TimeRange(
                        first[0].time_range.start_seconds,
                        first[-1].time_range.end_seconds,
                    ),
                    deleted_time_range=TimeRange(
                        second[0].time_range.start_seconds,
                        second[-1].time_range.end_seconds,
                    ),
                    repetition_gap_seconds=max(
                        second[0].time_range.start_seconds
                        - first[-1].time_range.end_seconds,
                        0.0,
                    ),
                )
            )
            del words[cursor + size : cursor + (2 * size)]
            del original_word_indexes[cursor + size : cursor + (2 * size)]
            cursor += size

        if not decisions:
            return sentence, ()
        cleaned_words = tuple(words)
        retained_boundaries = tuple(
            original_word_indexes.index(index)
            for index in sentence.asr_boundary_word_indexes
            if index in original_word_indexes
            and original_word_indexes.index(index) > 0
        )
        return (
            Sentence(
                text=_words_text(cleaned_words),
                time_range=sentence.time_range,
                words=cleaned_words,
                is_question=sentence.is_question,
                asr_boundary_word_indexes=retained_boundaries,
            ),
            tuple(decisions),
        )


def _longest_repeated_sequence(
    words: list[Word],
    start: int,
    minimum_characters: int,
    maximum_words: int,
    maximum_gap_seconds: float,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> int:
    limit = min(maximum_words, (len(words) - start) // 2)
    for size in range(limit, 0, -1):
        left = words[start : start + size]
        right = words[start + size : start + (2 * size)]
        if tuple(_word_text(word) for word in left) != tuple(
            _word_text(word) for word in right
        ):
            continue
        if len(_words_text(tuple(left))) < minimum_characters:
            continue
        repeated_text = _words_text(tuple((*left, *right)))
        if analyzer is not None:
            analysis = analyzer.analyze(repeated_text)
            if len(analysis) == 1 and analysis[0].surface == repeated_text:
                continue
        gap = right[0].time_range.start_seconds - left[-1].time_range.end_seconds
        if gap < 0 or gap > maximum_gap_seconds:
            continue
        return size
    return 0


def _words_text(words: tuple[Word, ...]) -> str:
    return "".join(_word_text(word) for word in words)


def _word_text(word: Word) -> str:
    return unicodedata.normalize("NFKC", word.text).strip()


__all__ = ["AuditableRepeatedTextCleaner"]

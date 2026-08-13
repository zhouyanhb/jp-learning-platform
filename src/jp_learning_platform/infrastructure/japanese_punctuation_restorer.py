"""Pause-aware restoration of missing Japanese internal commas."""

from __future__ import annotations

from dataclasses import dataclass

from jp_learning_platform.domain import Sentence, Word
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseMorpheme,
    JapaneseMorphologicalAnalyzer,
)
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG,
)
from jp_learning_platform.workflow.punctuation_attribution_stage import (
    InternalPunctuationRestoration,
)


@dataclass(frozen=True, slots=True)
class PauseAwareJapaneseCommaRestorer:
    analyzer: JapaneseMorphologicalAnalyzer

    def restore(self, sentence: Sentence) -> InternalPunctuationRestoration | None:
        if not sentence.words:
            return None
        morphemes = self.analyzer.analyze(sentence.text)
        cursor = 0
        for index, morpheme in enumerate(morphemes[:-1]):
            cursor += len(morpheme.surface)
            if not _is_connective_before_independent_predicate(
                morpheme,
                morphemes[index + 1],
            ):
                continue
            word_index = _word_boundary_index(sentence.words, cursor)
            if word_index is None or not _has_pause_evidence(sentence.words, word_index):
                continue
            restored = _insert_comma(sentence, word_index, cursor)
            return InternalPunctuationRestoration(
                sentence=restored,
                attributed_text="、",
                original_sentence_text=sentence.text,
                reason="pause_supported_connective_comma",
            )
        return None


def _is_connective_before_independent_predicate(
    morpheme: JapaneseMorpheme,
    following: JapaneseMorpheme,
) -> bool:
    return bool(
        morpheme.part_of_speech[:2] == ("助詞", "接続助詞")
        and morpheme.dictionary_form in {"て", "で"}
        and following.part_of_speech
        and following.part_of_speech[0] in {"動詞", "形容詞"}
        and len(following.part_of_speech) > 1
        and following.part_of_speech[1] == "一般"
    )


def _word_boundary_index(words: tuple[Word, ...], character_offset: int) -> int | None:
    cursor = 0
    for index, word in enumerate(words):
        cursor += len("".join(word.text.split()))
        if cursor == character_offset:
            return index
        if cursor > character_offset:
            return None
    return None


def _has_pause_evidence(words: tuple[Word, ...], left_index: int) -> bool:
    if left_index + 1 >= len(words):
        return False
    left = words[left_index]
    right = words[left_index + 1]
    gap = right.time_range.start_seconds - left.time_range.end_seconds
    compact_length = max(len("".join(left.text.split())), 1)
    seconds_per_character = left.time_range.duration_seconds / compact_length
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    return bool(
        gap >= config.min_pause_seconds
        or (
            left.time_range.duration_seconds
            >= config.extended_word_duration_seconds
            and seconds_per_character
            >= config.extended_word_seconds_per_character
        )
    )


def _insert_comma(
    sentence: Sentence,
    left_word_index: int,
    character_offset: int,
) -> Sentence:
    words = list(sentence.words)
    left = words[left_word_index]
    words[left_word_index] = Word(
        text=f"{left.text}、",
        time_range=left.time_range,
        confidence=left.confidence,
    )
    return Sentence(
        text=f"{sentence.text[:character_offset]}、{sentence.text[character_offset:]}",
        time_range=sentence.time_range,
        words=tuple(words),
        is_question=sentence.is_question,
        asr_boundary_word_indexes=sentence.asr_boundary_word_indexes,
    )


__all__ = ["PauseAwareJapaneseCommaRestorer"]

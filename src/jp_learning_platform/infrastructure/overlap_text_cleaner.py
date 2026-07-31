"""Conservative cross-segment overlap cleanup adapter."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseMorpheme,
    JapaneseMorphologicalAnalyzer,
)
from jp_learning_platform.workflow.overlap_text_cleanup_stage import (
    OverlapTextCleanup,
    OverlapTextCleanupDecision,
    OverlapTextCleanupRequest,
)


@dataclass(frozen=True, slots=True)
class AuditableOverlapTextCleaner:
    """Remove only complete leading words proven to repeat overlapping audio."""

    min_duplicate_characters: int = 2
    max_contiguous_gap_seconds: float = 0.1
    max_single_character_gap_seconds: float = 0.25
    anomalous_seconds_per_character: float = 1.5
    morphological_analyzer: JapaneseMorphologicalAnalyzer | None = None

    def clean(self, request: OverlapTextCleanupRequest) -> OverlapTextCleanup:
        segments: list[Segment] = []
        decisions: list[OverlapTextCleanupDecision] = []
        for segment in request.segments:
            if not segments:
                segments.append(segment)
                continue
            previous, cleaned, decision = self._clean_pair(segments[-1], segment)
            segments[-1] = previous
            segments.append(cleaned)
            if decision is not None:
                decisions.append(decision)
        return OverlapTextCleanup(request.source_path, tuple(segments), tuple(decisions))

    def _clean_pair(
        self, previous: Segment, current: Segment
    ) -> tuple[Segment, Segment, OverlapTextCleanupDecision | None]:
        if not previous.sentences or not current.sentences:
            return previous, current, None
        left = previous.sentences[-1]
        right = current.sentences[0]
        if not right.words:
            return previous, current, None

        prefix_count, reason, evidence = _proven_prefix(
            left,
            right,
            self.min_duplicate_characters,
            self.max_contiguous_gap_seconds,
            self.anomalous_seconds_per_character,
            self.morphological_analyzer,
            self.max_single_character_gap_seconds,
        )
        if prefix_count == 0 or prefix_count == len(right.words):
            return previous, current, None

        removed_words = right.words[:prefix_count]
        punctuation_count = _leading_punctuation_word_count(
            right.words[prefix_count:]
        )
        if prefix_count + punctuation_count == len(right.words):
            return previous, current, None
        punctuation_words = right.words[
            prefix_count : prefix_count + punctuation_count
        ]
        remaining_words = right.words[prefix_count + punctuation_count :]
        deleted_text = _words_text(removed_words)
        original_text = current.text
        removed_count = prefix_count + punctuation_count
        replacement = _sentence_from_words(
            remaining_words,
            asr_boundary_word_indexes=tuple(
                index - removed_count
                for index in right.asr_boundary_word_indexes
                if index > removed_count
            ),
        )
        sentences = (replacement, *current.sentences[1:])
        cleaned = Segment(
            position=current.position,
            text="".join(sentence.text for sentence in sentences),
            time_range=TimeRange(
                replacement.time_range.start_seconds,
                current.time_range.end_seconds,
            ),
            sentences=sentences,
        )
        overlap = min(
            left.time_range.end_seconds, removed_words[-1].time_range.end_seconds
        ) - max(
            left.time_range.start_seconds, removed_words[0].time_range.start_seconds
        )
        updated_previous = (
            _append_punctuation_words(previous, punctuation_words)
            if punctuation_words
            else previous
        )
        return updated_previous, cleaned, OverlapTextCleanupDecision(
            previous_segment_position=previous.position,
            segment_position=current.position,
            original_text=original_text,
            deleted_text=deleted_text,
            deletion_start=0,
            deletion_end=len(deleted_text),
            deleted_time_range=TimeRange(
                removed_words[0].time_range.start_seconds,
                removed_words[-1].time_range.end_seconds,
            ),
            time_overlap_seconds=max(overlap, 0.0),
            boundary_gap_seconds=max(
                right.time_range.start_seconds - left.time_range.end_seconds,
                0.0,
            ),
            reason=reason,
            evidence=evidence,
            transferred_punctuation_text=_words_text(punctuation_words),
            transferred_word_indexes=tuple(
                range(prefix_count, prefix_count + punctuation_count)
            ),
            transferred_time_range=(
                TimeRange(
                    punctuation_words[0].time_range.start_seconds,
                    punctuation_words[-1].time_range.end_seconds,
                )
                if punctuation_words
                else None
            ),
        )


def _proven_prefix(
    left: Sentence,
    right: Sentence,
    minimum: int,
    max_contiguous_gap_seconds: float,
    anomalous_seconds_per_character: float,
    analyzer: JapaneseMorphologicalAnalyzer | None,
    max_single_character_gap_seconds: float,
) -> tuple[int, str, tuple[str, ...]]:
    left_text = _compact(left.text)
    prefix = ""
    candidates: list[int] = []
    for index, word in enumerate(right.words, start=1):
        prefix += _word_text(word)
        if len(prefix) >= minimum and left_text.endswith(prefix):
            candidates.append(index)

    for count in reversed(candidates):
        words = right.words[:count]
        if all(_overlap_seconds(left.time_range, word.time_range) > 0 for word in words):
            return count, "exact_boundary_text_with_aligned_time_overlap", (
                "exact_boundary_text",
                "aligned_time_overlap",
            )

        boundary_gap = words[0].time_range.start_seconds - left.time_range.end_seconds
        if (
            0 <= boundary_gap <= max_contiguous_gap_seconds
            and any(
                word.time_range.duration_seconds / max(len(_word_text(word)), 1)
                >= anomalous_seconds_per_character
                for word in words
            )
        ):
            return count, "exact_boundary_text_with_contiguous_anomalous_alignment", (
                "exact_boundary_text",
                "contiguous_timing",
                "anomalous_alignment_duration",
            )

        if 0 <= boundary_gap <= max_contiguous_gap_seconds:
            return count, "exact_boundary_words_with_contiguous_timing", (
                "exact_boundary_words",
                "contiguous_timing",
            )

    if _deletion_improves_single_character_morphology(
        left, right, max_single_character_gap_seconds, analyzer
    ):
        return 1, "single_character_boundary_echo_with_repaired_morphology", (
            "exact_single_aligned_word_boundary",
            "contiguous_timing",
            "standalone_function_morpheme_before_deletion",
            "independent_morpheme_preserved_after_deletion",
            "morphological_completeness_improved",
        )

    if _is_high_evidence_single_character_echo(
        left, right, max_single_character_gap_seconds, analyzer
    ):
        return 1, "single_character_boundary_echo_with_morphological_evidence", (
            "exact_single_aligned_word_boundary",
            "contiguous_timing",
            "matching_function_morphemes_at_boundary",
            "right_character_is_independent_morpheme",
        )

    return 0, "", ()


def _is_high_evidence_single_character_echo(
    left: Sentence,
    right: Sentence,
    max_contiguous_gap_seconds: float,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> bool:
    if analyzer is None or not left.words or len(right.words) < 2:
        return False
    repeated = _word_text(right.words[0])
    if (
        len(repeated) != 1
        or unicodedata.category(repeated) != "Lo"
        or not _lexical_text(left.text).endswith(repeated)
    ):
        return False
    gap = right.words[0].time_range.start_seconds - left.time_range.end_seconds
    if gap < 0 or gap > max_contiguous_gap_seconds:
        return False

    left_text = _compact(left.text)
    right_text = _compact(right.text)
    analysis = analyzer.analyze(f"{left_text}{right_text}")
    boundary = len(left_text)
    left_morpheme, right_morpheme = _morphemes_at_boundary(analysis, boundary)
    return bool(
        left_morpheme
        and right_morpheme
        and left_morpheme.surface == repeated
        and right_morpheme.surface == repeated
        and _is_function_morpheme(left_morpheme)
        and _is_function_morpheme(right_morpheme)
    )


def _deletion_improves_single_character_morphology(
    left: Sentence,
    right: Sentence,
    max_contiguous_gap_seconds: float,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> bool:
    """Accept deletion only when it exposes an intact independent morpheme."""
    if analyzer is None or not left.words or len(right.words) < 2:
        return False
    repeated = _word_text(right.words[0])
    if (
        len(repeated) != 1
        or unicodedata.category(repeated) != "Lo"
        or not _lexical_text(left.text).endswith(repeated)
    ):
        return False
    gap = right.words[0].time_range.start_seconds - left.time_range.end_seconds
    if gap < 0 or gap > max_contiguous_gap_seconds:
        return False

    original = analyzer.analyze(_compact(right.text))
    repaired = analyzer.analyze(_words_text(right.words[1:]))
    if len(original) < 2 or not repaired:
        return False
    original_first, original_remainder = original[0], original[1]
    repaired_first = repaired[0]
    return bool(
        original_first.surface == repeated
        and _is_function_morpheme(original_first)
        and _is_independent_start_morpheme(original_remainder)
        and repaired_first.surface == original_remainder.surface
        and repaired_first.part_of_speech == original_remainder.part_of_speech
    )


def _is_independent_start_morpheme(morpheme: JapaneseMorpheme) -> bool:
    return bool(morpheme.part_of_speech) and morpheme.part_of_speech[0] in {
        "感動詞", "接続詞", "副詞", "代名詞", "名詞", "連体詞", "接頭辞",
    }


def _morphemes_at_boundary(
    morphemes: tuple[JapaneseMorpheme, ...], boundary: int
) -> tuple[JapaneseMorpheme | None, JapaneseMorpheme | None]:
    cursor = 0
    before: JapaneseMorpheme | None = None
    after: JapaneseMorpheme | None = None
    for morpheme in morphemes:
        start = cursor
        cursor += len(_compact(morpheme.surface))
        if cursor == boundary:
            before = morpheme
        if start == boundary:
            after = morpheme
            break
    return before, after


def _is_function_morpheme(morpheme: JapaneseMorpheme) -> bool:
    return bool(morpheme.part_of_speech) and morpheme.part_of_speech[0] in {
        "助詞",
        "助動詞",
        "接続詞",
    }


def _overlap_seconds(left: TimeRange, right: TimeRange) -> float:
    return min(left.end_seconds, right.end_seconds) - max(
        left.start_seconds, right.start_seconds
    )


def _leading_punctuation_word_count(words: tuple[Word, ...]) -> int:
    count = 0
    for word in words:
        text = _word_text(word)
        if not text or not all(
            unicodedata.category(character).startswith("P") for character in text
        ):
            break
        count += 1
    return count


def _append_punctuation_words(
    segment: Segment,
    punctuation_words: tuple[Word, ...],
) -> Segment:
    sentence = segment.sentences[-1]
    punctuation_text = _words_text(punctuation_words)
    updated_sentence = Sentence(
        text=f"{sentence.text}{punctuation_text}",
        time_range=TimeRange(
            sentence.time_range.start_seconds,
            punctuation_words[-1].time_range.end_seconds,
        ),
        words=(*sentence.words, *punctuation_words),
        is_question=(
            sentence.is_question
            or any(
                "QUESTION MARK" in unicodedata.name(character, "")
                for character in punctuation_text
            )
        ),
        asr_boundary_word_indexes=sentence.asr_boundary_word_indexes,
    )
    sentences = (*segment.sentences[:-1], updated_sentence)
    return Segment(
        position=segment.position,
        text="".join(item.text for item in sentences),
        time_range=TimeRange(
            segment.time_range.start_seconds,
            punctuation_words[-1].time_range.end_seconds,
        ),
        sentences=sentences,
    )


def _sentence_from_words(
    words: tuple[Word, ...],
    *,
    asr_boundary_word_indexes: tuple[int, ...] = (),
) -> Sentence:
    return Sentence(
        text=_words_text(words),
        time_range=TimeRange(
            words[0].time_range.start_seconds,
            words[-1].time_range.end_seconds,
        ),
        words=words,
        asr_boundary_word_indexes=asr_boundary_word_indexes,
    )


def _words_text(words: tuple[Word, ...]) -> str:
    return "".join(_word_text(word) for word in words)


def _word_text(word: Word) -> str:
    return unicodedata.normalize("NFKC", word.text).strip()


def _lexical_text(text: str) -> str:
    normalized = _compact(text)
    return normalized.rstrip(
        "".join(
            character
            for character in normalized
            if unicodedata.category(character).startswith("P")
        )
    )


def _compact(text: str) -> str:
    return "".join(character for character in text.strip() if not character.isspace())


__all__ = ["AuditableOverlapTextCleaner"]

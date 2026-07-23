"""Pause-aware Japanese sentence boundary resolver."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG,
)
from jp_learning_platform.workflow.sentence_boundary_stage import (
    SentenceBoundaryDecision,
    SentenceBoundaryResolution,
    SentenceBoundaryResolutionRequest,
)

DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS = (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG.min_pause_seconds
)
DEFAULT_SENTENCE_BOUNDARY_TERMINAL_MARKS = (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG.terminal_marks
)
DEFAULT_SENTENCE_BOUNDARY_FINAL_SUFFIXES = (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG.sentence_final_suffixes
)

_CLOSING_QUOTES = frozenset(("」", "』", "）", ")", "】", "］", "]", "〉", "》"))
_STRONG_PAUSE_SECONDS = 1.5
_SENTENCE_FINAL_PARTICLES = frozenset(("か", "ね", "よ", "な"))


@dataclass(frozen=True, slots=True)
class JapaneseSentenceBoundaryResolver:
    """Split sentence-sized ASR segments using punctuation and word-level pauses."""

    min_pause_seconds: float = DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS
    terminal_marks: tuple[str, ...] = DEFAULT_SENTENCE_BOUNDARY_TERMINAL_MARKS
    sentence_final_suffixes: tuple[str, ...] = DEFAULT_SENTENCE_BOUNDARY_FINAL_SUFFIXES

    def resolve(
        self,
        request: SentenceBoundaryResolutionRequest,
    ) -> SentenceBoundaryResolution:
        if not isinstance(request, SentenceBoundaryResolutionRequest):
            raise TypeError("request must be a SentenceBoundaryResolutionRequest.")

        segments = _merge_adjacent_sentence_overlaps(request.segments)
        segments = _remove_adjacent_boundary_echoes(segments)
        segments = _restore_inter_segment_commas(segments)
        resolved_segments: list[Segment] = []
        decisions: list[SentenceBoundaryDecision] = []
        for segment in segments:
            resolved_segment, segment_decisions = self._resolve_segment(segment)
            resolved_segments.append(resolved_segment)
            decisions.extend(segment_decisions)

        return SentenceBoundaryResolution(
            source_path=request.source_path,
            segments=tuple(resolved_segments),
            decisions=tuple(decisions),
        )

    def _resolve_segment(
        self,
        segment: Segment,
    ) -> tuple[Segment, tuple[SentenceBoundaryDecision, ...]]:
        sentences = segment.sentences or (
            Sentence(
                text=segment.text,
                time_range=segment.time_range,
                words=(),
                speaker_id=segment.speaker_id,
            ),
        )

        resolved_sentences: list[Sentence] = []
        decisions: list[SentenceBoundaryDecision] = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_parts, sentence_decisions = self._split_sentence(
                segment.position,
                sentence_index,
                sentence,
            )
            resolved_sentences.extend(sentence_parts)
            decisions.extend(sentence_decisions)

        if tuple(resolved_sentences) == sentences:
            return segment, tuple(decisions)

        return (
            Segment(
                position=segment.position,
                text="".join(sentence.text for sentence in resolved_sentences),
                time_range=segment.time_range,
                sentences=tuple(resolved_sentences),
                speaker_id=segment.speaker_id,
            ),
            tuple(decisions),
        )

    def _split_sentence(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
    ) -> tuple[tuple[Sentence, ...], tuple[SentenceBoundaryDecision, ...]]:
        if len(sentence.words) < 2:
            return (sentence,), ()

        boundaries: list[int] = []
        decisions: list[SentenceBoundaryDecision] = []
        chunk_start = 0
        for word_index in range(len(sentence.words) - 1):
            reason = self._boundary_reason(
                sentence.words,
                chunk_start,
                word_index,
            )
            if reason is None:
                continue

            left_text = _words_text(sentence.words[chunk_start : word_index + 1])
            right_text = _words_text(sentence.words[word_index + 1 :])
            if not left_text or not right_text:
                continue

            gap_seconds = _word_gap_seconds(
                sentence.words[word_index],
                sentence.words[word_index + 1],
            )
            boundaries.append(word_index)
            decisions.append(
                SentenceBoundaryDecision(
                    segment_position=segment_position,
                    sentence_index=sentence_index,
                    word_index=word_index,
                    gap_seconds=max(gap_seconds, 0.0),
                    reason=reason,
                    left_text=left_text,
                    right_text=right_text,
                )
            )
            chunk_start = word_index + 1

        if not boundaries:
            return (sentence,), ()

        parts: list[Sentence] = []
        start_index = 0
        for boundary in boundaries:
            parts.append(
                _sentence_from_words(
                    sentence.words[start_index : boundary + 1],
                    speaker_id=sentence.speaker_id,
                )
            )
            start_index = boundary + 1

        parts.append(
            _sentence_from_words(
                sentence.words[start_index:],
                speaker_id=sentence.speaker_id,
            )
        )
        return tuple(parts), tuple(decisions)

    def _boundary_reason(
        self,
        words: tuple[Word, ...],
        chunk_start: int,
        word_index: int,
    ) -> str | None:
        current_text = _word_text(words[word_index])
        left_text = _words_text(words[chunk_start : word_index + 1])
        if _ends_with_terminal_mark(current_text, self.terminal_marks):
            return "terminal_mark"

        if _ends_with_terminal_mark(left_text, self.terminal_marks):
            return "terminal_mark"

        gap_seconds = _word_gap_seconds(words[word_index], words[word_index + 1])
        right_text = _words_text(words[word_index + 1 :])
        if _starts_with_sentence_final_particle(right_text):
            return None

        if right_text[:1].isdigit() and not _looks_sentence_final(
            left_text,
            self.sentence_final_suffixes,
        ):
            return None

        if gap_seconds >= _STRONG_PAUSE_SECONDS:
            return "strong_pause"

        if len(left_text) >= 2 and right_text.startswith(left_text):
            return "repeated_heading"

        if _looks_sentence_final(left_text, self.sentence_final_suffixes):
            if gap_seconds >= self.min_pause_seconds:
                return "pause_after_sentence_final"
            return "sentence_final_expression"

        return None


def _sentence_from_words(
    words: tuple[Word, ...],
    *,
    speaker_id: str | None,
) -> Sentence:
    if not words:
        raise ValueError("words must not be empty.")

    return Sentence(
        text=_words_text(words),
        time_range=TimeRange(
            words[0].time_range.start_seconds,
            words[-1].time_range.end_seconds,
        ),
        words=words,
        speaker_id=speaker_id or _common_speaker_id(words),
    )


def _word_gap_seconds(current: Word, nxt: Word) -> float:
    return nxt.time_range.start_seconds - current.time_range.end_seconds


def _words_text(words: tuple[Word, ...]) -> str:
    return "".join(_word_text(word) for word in words).strip()


def _word_text(word: Word) -> str:
    return unicodedata.normalize("NFKC", word.text).strip()


def _looks_sentence_final(text: str, suffixes: tuple[str, ...]) -> bool:
    normalized = _trim_closing_quotes(_compact_text(text))
    if not normalized:
        return False

    return any(normalized.endswith(suffix) for suffix in suffixes)


def _ends_with_terminal_mark(text: str, terminal_marks: tuple[str, ...]) -> bool:
    normalized = _trim_closing_quotes(_compact_text(text))
    if not normalized:
        return False

    return normalized.endswith(terminal_marks)


def _trim_closing_quotes(text: str) -> str:
    normalized = text
    while normalized and normalized[-1] in _CLOSING_QUOTES:
        normalized = normalized[:-1]
    return normalized


def _compact_text(text: str) -> str:
    return "".join(character for character in text.strip() if not character.isspace())


def _starts_with_sentence_final_particle(text: str) -> bool:
    normalized = _compact_text(text)
    return bool(normalized) and normalized[0] in _SENTENCE_FINAL_PARTICLES


def _common_speaker_id(words: tuple[Word, ...]) -> str | None:
    speaker_ids = tuple(
        dict.fromkeys(word.speaker_id for word in words if word.speaker_id is not None)
    )
    if len(speaker_ids) == 1:
        return speaker_ids[0]

    return None


def _remove_adjacent_boundary_echoes(
    segments: tuple[Segment, ...],
) -> tuple[Segment, ...]:
    """Drop a word repeated at the start of the next ASR segment.

    Whisper commonly carries the final word (for example ``です``) into the
    following segment.  The aligned word timing makes this safe to identify.
    """
    normalized: list[Segment] = []
    for index, segment in enumerate(segments):
        if index == 0 or not segment.sentences or not segment.sentences[0].words:
            normalized.append(segment)
            continue

        previous_text = _compact_text(segments[index - 1].text)
        sentence = segment.sentences[0]
        first_word = sentence.words[0]
        echo = _word_text(first_word)
        if not echo or not previous_text.endswith(echo):
            normalized.append(segment)
            continue

        remaining_words = sentence.words[1:]
        if not remaining_words or not _compact_text(sentence.text).startswith(echo):
            normalized.append(segment)
            continue

        replacement = _sentence_from_words(
            remaining_words,
            speaker_id=sentence.speaker_id,
        )
        sentences = (replacement, *segment.sentences[1:])
        normalized.append(
            Segment(
                position=segment.position,
                text="".join(item.text for item in sentences),
                time_range=TimeRange(
                    replacement.time_range.start_seconds,
                    segment.time_range.end_seconds,
                ),
                sentences=sentences,
                speaker_id=segment.speaker_id,
            )
        )

    return tuple(normalized)


def _merge_adjacent_sentence_overlaps(
    segments: tuple[Segment, ...],
) -> tuple[Segment, ...]:
    """Join an unfinished sentence split across ASR segments with repeated audio."""
    merged: list[Segment] = []
    for segment in segments:
        if not merged or len(segment.sentences) != 1:
            merged.append(segment)
            continue

        previous = merged[-1]
        if not previous.sentences:
            merged.append(segment)
            continue

        left = previous.sentences[-1]
        right = segment.sentences[0]
        overlap = _boundary_overlap(left.text, right.text)
        first_word_length = len(_word_text(right.words[0])) if right.words else 0
        if (
            overlap < 2
            or overlap == first_word_length
            or _ends_with_terminal_mark(left.text, ("。", "？", "！", "?", "!"))
        ):
            merged.append(segment)
            continue

        skipped = 0
        consumed = 0
        for word in right.words:
            consumed += len(_word_text(word))
            skipped += 1
            if consumed >= overlap:
                break
        remaining_words = right.words[skipped:]
        combined_words = (*left.words, *remaining_words)
        combined = Sentence(
            text=f"{left.text}{right.text[overlap:]}",
            time_range=TimeRange(
                left.time_range.start_seconds,
                right.time_range.end_seconds,
            ),
            words=combined_words,
            speaker_id=left.speaker_id or right.speaker_id,
        )
        sentences = (*previous.sentences[:-1], combined)
        merged[-1] = Segment(
            position=previous.position,
            text="".join(item.text for item in sentences),
            time_range=TimeRange(
                previous.time_range.start_seconds,
                segment.time_range.end_seconds,
            ),
            sentences=sentences,
            speaker_id=previous.speaker_id,
        )

    return tuple(
        Segment(
            position=position,
            text=segment.text,
            time_range=segment.time_range,
            sentences=segment.sentences,
            speaker_id=segment.speaker_id,
        )
        for position, segment in enumerate(merged)
    )


def _boundary_overlap(left: str, right: str) -> int:
    left_compact = _compact_text(left)
    right_compact = _compact_text(right)
    limit = min(len(left_compact), len(right_compact))
    for size in range(limit, 1, -1):
        if left_compact.endswith(right_compact[:size]):
            return size
    return 0


def _restore_inter_segment_commas(
    segments: tuple[Segment, ...],
) -> tuple[Segment, ...]:
    """Mark a connective te-form at an ASR boundary without splitting it."""
    restored: list[Segment] = []
    for index, segment in enumerate(segments):
        if index + 1 >= len(segments) or not segment.sentences:
            restored.append(segment)
            continue

        sentence = segment.sentences[-1]
        if not _ends_with_connective_te(sentence.text):
            restored.append(segment)
            continue

        punctuated = _append_comma(sentence)
        sentences = (*segment.sentences[:-1], punctuated)
        restored.append(
            Segment(
                position=segment.position,
                text="".join(item.text for item in sentences),
                time_range=segment.time_range,
                sentences=sentences,
                speaker_id=segment.speaker_id,
            )
        )

    return tuple(restored)


def _ends_with_connective_te(text: str) -> bool:
    normalized = _compact_text(text)
    if not normalized or normalized.endswith(("、", ",")):
        return False
    return normalized.endswith(("て", "で"))


def _append_comma(sentence: Sentence) -> Sentence:
    words = sentence.words
    if words:
        last = words[-1]
        words = (
            *words[:-1],
            Word(
                text=f"{_word_text(last)}、",
                time_range=last.time_range,
                confidence=last.confidence,
                speaker_id=last.speaker_id,
            ),
        )
    return Sentence(
        text=f"{sentence.text.rstrip()}、",
        time_range=sentence.time_range,
        words=words,
        speaker_id=sentence.speaker_id,
    )


__all__ = [
    "DEFAULT_SENTENCE_BOUNDARY_FINAL_SUFFIXES",
    "DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS",
    "DEFAULT_SENTENCE_BOUNDARY_TERMINAL_MARKS",
    "JapaneseSentenceBoundaryResolver",
]

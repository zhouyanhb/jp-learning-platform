"""Qwen transcript repair adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_QWEN_REPAIR_CONFIG,
    DEFAULT_QWEN_REPAIR_SAFETY_CONFIG,
)
from jp_learning_platform.infrastructure.japanese_text import (
    text_ends_with_complete_predicate,
)
from jp_learning_platform.workflow.qwen_repair_stage import (
    QwenRepair,
    QwenRepairDecision,
    QwenRepairRequest,
)

DEFAULT_QWEN_CONTEXT = DEFAULT_QWEN_REPAIR_CONFIG.context_size
DEFAULT_QWEN_THREADS = DEFAULT_QWEN_REPAIR_CONFIG.threads
DEFAULT_QWEN_GPU_LAYERS = DEFAULT_QWEN_REPAIR_CONFIG.gpu_layers
DEFAULT_QWEN_MAX_TOKENS = DEFAULT_QWEN_REPAIR_CONFIG.max_tokens
DEFAULT_QWEN_TEMPERATURE = DEFAULT_QWEN_REPAIR_CONFIG.temperature
DEFAULT_QWEN_TOP_P = DEFAULT_QWEN_REPAIR_CONFIG.top_p
DEFAULT_QWEN_REPEAT_PENALTY = DEFAULT_QWEN_REPAIR_CONFIG.repeat_penalty
DEFAULT_QWEN_REPAIR_MAX_LENGTH_DELTA_RATIO = (
    DEFAULT_QWEN_REPAIR_SAFETY_CONFIG.max_length_delta_ratio
)
DEFAULT_QWEN_REPAIR_MAX_CONTENT_CHANGE_RATIO = (
    DEFAULT_QWEN_REPAIR_SAFETY_CONFIG.max_content_change_ratio
)
_PARAPHRASE_MIN_CHANGED_SPAN_LENGTH = 3
_PARAPHRASE_MIN_CONTENT_CHARS = 2
_SAFE_REPLACEMENT_MIN_CHARACTER_SIMILARITY = 0.45
_SAFE_REPLACEMENT_MIN_READING_SIMILARITY = 0.75
_SHORT_UNCHECKED_REPLACEMENT_LENGTH = 2
_BOUNDARY_DUPLICATE_MAX_GAP_SECONDS = 0.2
_BOUNDARY_DUPLICATE_MAX_OVERLAP_SECONDS = 0.2
_BOUNDARY_DUPLICATE_FUNCTION_WORDS = {
    "ください",
    "くださいね",
    "です",
    "でした",
    "でしょう",
    "ます",
    "ました",
    "ません",
}
_PUNCTUATION_FALLBACK_MARKS = frozenset("。、？！?!")
_PUNCTUATION_FALLBACK_TERMINAL_MARKS = frozenset("。？！?!")

_LOGGER = logging.getLogger(__name__)
_SAFETY_SUDACHI_TOKENIZER: Any | None = None
_SAFETY_SUDACHI_MODE: Any | None = None
_SAFETY_SUDACHI_UNAVAILABLE = False


class QwenDependencyError(RuntimeError):
    """Raised when llama-cpp-python is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "llama-cpp-python is required for Qwen repair. "
            "Install it with: python -m pip install -e '.[qwen]'"
        )


class QwenModelNotFoundError(RuntimeError):
    """Raised when the configured Qwen model file is missing."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        super().__init__(f"Qwen model file not found: {model_path}")


class QwenRepairSafetyReason(Enum):
    DISABLED = "qwen_disabled"
    ACCEPTED = "accepted"
    PUNCTUATION_FALLBACK = "punctuation_fallback"
    EMPTY_CANDIDATE = "empty_candidate"
    INVALID_EDIT_CANDIDATE = "invalid_edit_candidate"
    PROTECTED_INFORMATION_REWRITE = "protected_information_rewrite"
    LENGTH_DELTA_EXCEEDED = "length_delta_exceeded"
    CONTENT_CHANGE_EXCEEDED = "content_change_exceeded"
    MEANINGFUL_CONTENT_DELETED = "meaningful_content_deleted"
    PARAPHRASE_REWRITE = "paraphrase_rewrite"


def _normalize_ratio(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")

    try:
        ratio = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a number.") from error

    if ratio < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")

    return ratio


@dataclass(frozen=True, slots=True)
class QwenRepairSafetyDecision:
    """Decision for accepting or rejecting one candidate repair."""

    original_text: str
    candidate_text: str
    accepted: bool
    reason: QwenRepairSafetyReason
    length_delta_ratio: float
    content_change_ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.original_text, str):
            raise TypeError("original_text must be a string.")

        if not isinstance(self.candidate_text, str):
            raise TypeError("candidate_text must be a string.")

        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool.")

        if not isinstance(self.reason, QwenRepairSafetyReason):
            raise TypeError("reason must be a QwenRepairSafetyReason.")

        object.__setattr__(
            self,
            "length_delta_ratio",
            _normalize_ratio(self.length_delta_ratio, "length_delta_ratio"),
        )
        object.__setattr__(
            self,
            "content_change_ratio",
            _normalize_ratio(self.content_change_ratio, "content_change_ratio"),
        )

    @property
    def selected_text(self) -> str:
        if self.accepted:
            return self.candidate_text

        return self.original_text


@dataclass(frozen=True, slots=True)
class _SafetyChange:
    original_unit: str
    candidate_unit: str
    original_start: int
    original_end: int
    candidate_start: int
    candidate_end: int


@dataclass(frozen=True, slots=True)
class _StructuredReplacementOption:
    replacement_text: str
    reason: str
    confidence: float | None
    source_reading: str
    replacement_reading: str


@dataclass(frozen=True, slots=True)
class _StructuredReplacementCandidate:
    edit_type: str
    source_text: str
    source_reading: str
    options: tuple[_StructuredReplacementOption, ...]


@dataclass(frozen=True, slots=True)
class _StructuredPunctuationCandidate:
    after_text: str
    mark: str


@dataclass(frozen=True, slots=True)
class _StructuredRepairCandidate:
    replacements: tuple[_StructuredReplacementCandidate, ...]
    punctuations: tuple[_StructuredPunctuationCandidate, ...]
    operation_count: int


@dataclass(frozen=True, slots=True)
class _StructuredRepairApplication:
    candidate_text: str
    applied_count: int
    rejected_count: int
    last_rejected_decision: QwenRepairSafetyDecision | None


@dataclass(frozen=True, slots=True)
class QwenRepairSafetyPolicy:
    """Reject Qwen repairs that likely add or remove spoken content."""

    max_length_delta_ratio: float = DEFAULT_QWEN_REPAIR_MAX_LENGTH_DELTA_RATIO
    max_content_change_ratio: float = DEFAULT_QWEN_REPAIR_MAX_CONTENT_CHANGE_RATIO

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_length_delta_ratio",
            _normalize_ratio(
                self.max_length_delta_ratio,
                "max_length_delta_ratio",
            ),
        )
        object.__setattr__(
            self,
            "max_content_change_ratio",
            _normalize_ratio(
                self.max_content_change_ratio,
                "max_content_change_ratio",
            ),
        )

    def decide(
        self,
        original_text: str,
        candidate_text: str,
    ) -> QwenRepairSafetyDecision:
        if not isinstance(original_text, str):
            raise TypeError("original_text must be a string.")

        if not isinstance(candidate_text, str):
            raise TypeError("candidate_text must be a string.")

        original = original_text.strip()
        candidate = candidate_text.strip()
        if not candidate:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.EMPTY_CANDIDATE,
                length_delta_ratio=1.0,
                content_change_ratio=1.0,
            )

        original_core = _normalize_text_for_safety(original)
        candidate_core = _normalize_text_for_safety(candidate)
        if original_core == candidate_core:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=True,
                reason=QwenRepairSafetyReason.ACCEPTED,
                length_delta_ratio=0.0,
                content_change_ratio=0.0,
            )

        length_delta_ratio = _length_delta_ratio(original_core, candidate_core)
        content_change_ratio = _content_change_ratio(original_core, candidate_core)
        if length_delta_ratio > self.max_length_delta_ratio:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.LENGTH_DELTA_EXCEEDED,
                length_delta_ratio=length_delta_ratio,
                content_change_ratio=content_change_ratio,
            )

        if content_change_ratio > self.max_content_change_ratio:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.CONTENT_CHANGE_EXCEEDED,
                length_delta_ratio=length_delta_ratio,
                content_change_ratio=content_change_ratio,
            )

        if _has_protected_information_rewrite(original_core, candidate_core):
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.PROTECTED_INFORMATION_REWRITE,
                length_delta_ratio=length_delta_ratio,
                content_change_ratio=content_change_ratio,
            )

        if _has_meaningful_deleted_content(original_core, candidate_core):
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.MEANINGFUL_CONTENT_DELETED,
                length_delta_ratio=length_delta_ratio,
                content_change_ratio=content_change_ratio,
            )

        if _has_unsafe_paraphrase_rewrite(original_core, candidate_core):
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=candidate,
                accepted=False,
                reason=QwenRepairSafetyReason.PARAPHRASE_REWRITE,
                length_delta_ratio=length_delta_ratio,
                content_change_ratio=content_change_ratio,
            )

        return QwenRepairSafetyDecision(
            original_text=original,
            candidate_text=candidate,
            accepted=True,
            reason=QwenRepairSafetyReason.ACCEPTED,
            length_delta_ratio=length_delta_ratio,
            content_change_ratio=content_change_ratio,
        )


def _normalize_text_for_safety(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    characters: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if character.isspace() or category.startswith("P"):
            continue

        characters.append(character.lower())

    return "".join(characters)


def _length_delta_ratio(original: str, candidate: str) -> float:
    baseline = max(len(original), 1)
    return abs(len(candidate) - len(original)) / baseline


def _content_change_ratio(original: str, candidate: str) -> float:
    baseline = max(len(original), len(candidate), 1)
    changed_units = 0
    for tag, original_start, original_end, candidate_start, candidate_end in (
        SequenceMatcher(None, original, candidate).get_opcodes()
    ):
        if tag == "equal":
            continue

        original_length = original_end - original_start
        candidate_length = candidate_end - candidate_start
        changed_units += max(original_length, candidate_length)

    return changed_units / baseline


def _has_meaningful_deleted_content(original: str, candidate: str) -> bool:
    for tag, original_start, original_end, _, _ in (
        SequenceMatcher(None, original, candidate).get_opcodes()
    ):
        if tag != "delete":
            continue

        if _is_meaningful_deleted_unit(original[original_start:original_end]):
            return True

    return False


def _has_protected_information_rewrite(original: str, candidate: str) -> bool:
    for change in _changed_safety_spans(original, candidate):
        if not _contains_ascii_alnum(change.original_unit):
            continue

        if change.original_unit == change.candidate_unit:
            continue

        if _ascii_alnum_sequence(change.original_unit) == _ascii_alnum_sequence(
            change.candidate_unit
        ):
            continue

        return True

    return False


def _contains_ascii_alnum(text: str) -> bool:
    return any(character.isascii() and character.isalnum() for character in text)


def _ascii_alnum_sequence(text: str) -> str:
    return "".join(
        character.lower()
        for character in text
        if character.isascii() and character.isalnum()
    )


def _has_unsafe_paraphrase_rewrite(original: str, candidate: str) -> bool:
    for change in _changed_safety_spans(original, candidate):
        if not _is_substantive_replacement(
            change.original_unit,
            change.candidate_unit,
        ):
            continue

        if _looks_like_transcription_repair(
            change.original_unit,
            change.candidate_unit,
        ):
            continue

        if _looks_like_expanded_transcription_repair(original, candidate, change):
            continue

        return True

    return False


def _changed_safety_spans(
    original: str,
    candidate: str,
) -> tuple[_SafetyChange, ...]:
    spans: list[tuple[int, int, int, int]] = []
    active_span: tuple[int, int, int, int] | None = None
    for tag, original_start, original_end, candidate_start, candidate_end in (
        SequenceMatcher(None, original, candidate).get_opcodes()
    ):
        if tag == "equal":
            if active_span is None:
                continue

            if _is_short_equal_bridge(
                original,
                candidate,
                original_start,
                original_end,
                candidate_start,
                candidate_end,
            ):
                active_span = (
                    active_span[0],
                    original_end,
                    active_span[2],
                    candidate_end,
                )
                continue

            spans.append(active_span)
            active_span = None
            continue

        if active_span is None:
            active_span = (
                original_start,
                original_end,
                candidate_start,
                candidate_end,
            )
            continue

        active_span = (
            active_span[0],
            original_end,
            active_span[2],
            candidate_end,
        )

    if active_span is not None:
        spans.append(active_span)

    return tuple(
        _SafetyChange(
            original_unit=original[original_start:original_end],
            candidate_unit=candidate[candidate_start:candidate_end],
            original_start=original_start,
            original_end=original_end,
            candidate_start=candidate_start,
            candidate_end=candidate_end,
        )
        for original_start, original_end, candidate_start, candidate_end in spans
    )


def _is_short_equal_bridge(
    original: str,
    candidate: str,
    original_start: int,
    original_end: int,
    candidate_start: int,
    candidate_end: int,
) -> bool:
    original_bridge = original[original_start:original_end]
    candidate_bridge = candidate[candidate_start:candidate_end]
    return (
        original_bridge == candidate_bridge
        and len(original_bridge) <= 1
        and len(candidate_bridge) <= 1
    )


def _is_substantive_replacement(original_unit: str, candidate_unit: str) -> bool:
    if not original_unit or not candidate_unit:
        return False

    return (
        _content_character_count(original_unit) >= _minimum_replacement_content_chars(
            original_unit,
            candidate_unit,
        )
        and _content_character_count(candidate_unit)
        >= _minimum_replacement_content_chars(original_unit, candidate_unit)
    )


def _minimum_replacement_content_chars(original_unit: str, candidate_unit: str) -> int:
    if (
        max(len(original_unit), len(candidate_unit))
        <= _SHORT_UNCHECKED_REPLACEMENT_LENGTH
    ):
        return 1

    if (
        max(len(original_unit), len(candidate_unit))
        < _PARAPHRASE_MIN_CHANGED_SPAN_LENGTH
    ):
        return 1

    return _PARAPHRASE_MIN_CONTENT_CHARS


def _looks_like_transcription_repair(
    original_unit: str,
    candidate_unit: str,
) -> bool:
    if _text_similarity(original_unit, candidate_unit) >= (
        _SAFE_REPLACEMENT_MIN_CHARACTER_SIMILARITY
    ):
        return True

    reading_similarity = _reading_similarity(original_unit, candidate_unit)
    return reading_similarity >= _SAFE_REPLACEMENT_MIN_READING_SIMILARITY


def _looks_like_expanded_transcription_repair(
    original: str,
    candidate: str,
    change: _SafetyChange,
) -> bool:
    for original_unit, candidate_unit in _expanded_replacement_units(
        original,
        candidate,
        change,
    ):
        if not _looks_like_transcription_repair(original_unit, candidate_unit):
            continue

        return True

    return False


def _expanded_replacement_units(
    original: str,
    candidate: str,
    change: _SafetyChange,
) -> tuple[tuple[str, str], ...]:
    units: list[tuple[str, str]] = []
    if (
        change.original_end < len(original)
        and change.candidate_end < len(candidate)
        and original[change.original_end] == candidate[change.candidate_end]
        and _is_strong_replacement_context_character(original[change.original_end])
    ):
        units.append(
            (
                original[change.original_start : change.original_end + 1],
                candidate[change.candidate_start : change.candidate_end + 1],
            )
        )

    if (
        change.original_start > 0
        and change.candidate_start > 0
        and original[change.original_start - 1] == candidate[change.candidate_start - 1]
        and _is_strong_replacement_context_character(
            original[change.original_start - 1]
        )
    ):
        units.append(
            (
                original[change.original_start - 1 : change.original_end],
                candidate[change.candidate_start - 1 : change.candidate_end],
            )
        )

    return tuple(units)


def _is_strong_replacement_context_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x30A0 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _text_similarity(original: str, candidate: str) -> float:
    return SequenceMatcher(None, original, candidate).ratio()


def _reading_similarity(original: str, candidate: str) -> float:
    original_reading = _japanese_reading(original)
    candidate_reading = _japanese_reading(candidate)
    if original_reading is None or candidate_reading is None:
        return 0.0

    if not original_reading or not candidate_reading:
        return 0.0

    return SequenceMatcher(None, original_reading, candidate_reading).ratio()


def _japanese_reading(text: str) -> str | None:
    tokenizer, mode = _load_safety_sudachi()
    if tokenizer is None or mode is None:
        return None

    readings: list[str] = []
    try:
        morphemes = tokenizer.tokenize(text, mode)
    except Exception:
        return None

    for morpheme in morphemes:
        reading = str(morpheme.reading_form())
        if not reading or reading == "*":
            reading = str(morpheme.surface())
        readings.append(reading)

    return _normalize_reading("".join(readings))


def _load_safety_sudachi() -> tuple[Any | None, Any | None]:
    global _SAFETY_SUDACHI_MODE
    global _SAFETY_SUDACHI_TOKENIZER
    global _SAFETY_SUDACHI_UNAVAILABLE

    if _SAFETY_SUDACHI_UNAVAILABLE:
        return None, None

    if _SAFETY_SUDACHI_TOKENIZER is not None and _SAFETY_SUDACHI_MODE is not None:
        return _SAFETY_SUDACHI_TOKENIZER, _SAFETY_SUDACHI_MODE

    try:
        from sudachipy import dictionary
        from sudachipy import tokenizer

        _SAFETY_SUDACHI_TOKENIZER = dictionary.Dictionary().create()
        _SAFETY_SUDACHI_MODE = tokenizer.Tokenizer.SplitMode.C
    except Exception:
        _SAFETY_SUDACHI_UNAVAILABLE = True
        return None, None

    return _SAFETY_SUDACHI_TOKENIZER, _SAFETY_SUDACHI_MODE


def _normalize_reading(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        _hiragana_to_katakana(character)
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _hiragana_to_katakana(character: str) -> str:
    codepoint = ord(character)
    if 0x3041 <= codepoint <= 0x3096:
        return chr(codepoint + 0x60)

    return character


def _is_meaningful_deleted_unit(text: str) -> bool:
    if any(character.isascii() and character.isalnum() for character in text):
        return True

    return _content_character_count(text) >= 2


def _content_character_count(text: str) -> int:
    return sum(1 for character in text if _is_japanese_content_character(character))


def _remove_cross_segment_leading_duplicates(
    segments: tuple[Segment, ...],
) -> tuple[Segment, ...]:
    cleaned_segments: list[Segment] = []
    previous_segment: Segment | None = None
    for segment in segments:
        cleaned_segment = _remove_leading_duplicate_from_segment(
            previous_segment,
            segment,
        )
        cleaned_segments.append(cleaned_segment)
        previous_segment = cleaned_segment

    return tuple(cleaned_segments)


def _remove_leading_duplicate_from_segment(
    previous_segment: Segment | None,
    segment: Segment,
) -> Segment:
    if previous_segment is None:
        return segment

    previous_word = _last_segment_word(previous_segment)
    first_word = _first_segment_word(segment)
    if previous_word is None or first_word is None:
        return segment

    if not _is_cross_segment_duplicate(previous_word, first_word):
        return segment

    return _drop_first_word(segment)


def _is_cross_segment_duplicate(previous_word: Word, current_word: Word) -> bool:
    previous_text = _normalize_boundary_duplicate_text(previous_word.text)
    current_text = _normalize_boundary_duplicate_text(current_word.text)
    if previous_text != current_text:
        return False

    if current_text not in _BOUNDARY_DUPLICATE_FUNCTION_WORDS:
        return False

    gap_seconds = current_word.time_range.start_seconds - previous_word.time_range.end_seconds
    if gap_seconds > _BOUNDARY_DUPLICATE_MAX_GAP_SECONDS:
        return False

    overlap_seconds = previous_word.time_range.end_seconds - current_word.time_range.start_seconds
    return overlap_seconds <= _BOUNDARY_DUPLICATE_MAX_OVERLAP_SECONDS


def _normalize_boundary_duplicate_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    ).lower()


def _drop_first_word(segment: Segment) -> Segment:
    if not segment.sentences:
        return segment

    sentences = tuple(
        sentence for sentence in (_drop_first_word_from_sentences(segment.sentences))
        if sentence is not None
    )
    if not sentences:
        return segment

    return Segment(
        position=segment.position,
        text=_remove_leading_text(segment.text, _first_segment_word(segment).text),
        time_range=_time_range_for_sentences(sentences),
        sentences=sentences,
        speaker_id=segment.speaker_id,
    )


def _drop_first_word_from_sentences(
    sentences: tuple[Sentence, ...],
) -> tuple[Sentence | None, ...]:
    dropped = False
    updated_sentences: list[Sentence | None] = []
    for sentence in sentences:
        if dropped or not sentence.words:
            updated_sentences.append(sentence)
            continue

        dropped = True
        remaining_words = sentence.words[1:]
        if not remaining_words:
            updated_sentences.append(None)
            continue

        updated_sentences.append(
            Sentence(
                text=_remove_leading_text(sentence.text, sentence.words[0].text),
                time_range=_time_range_for_words(remaining_words),
                words=remaining_words,
                speaker_id=sentence.speaker_id,
            )
        )

    return tuple(updated_sentences)


def _remove_leading_text(text: str, prefix: str) -> str:
    stripped_text = text.lstrip()
    if stripped_text.startswith(prefix):
        stripped_text = stripped_text[len(prefix) :].lstrip()
        if stripped_text:
            return stripped_text

    compact_prefix = _normalize_boundary_duplicate_text(prefix)
    index = 0
    for character in stripped_text:
        if unicodedata.category(character).startswith("P") or character.isspace():
            index += 1
            continue

        break

    if _normalize_boundary_duplicate_text(stripped_text[index:]).startswith(
        compact_prefix
    ):
        stripped_text = stripped_text[index + len(prefix) :].lstrip()
        if stripped_text:
            return stripped_text

    return text


def _time_range_for_sentences(sentences: tuple[Sentence, ...]) -> TimeRange:
    return TimeRange(
        min(sentence.time_range.start_seconds for sentence in sentences),
        max(sentence.time_range.end_seconds for sentence in sentences),
    )


def _time_range_for_words(words: tuple[Word, ...]) -> TimeRange:
    return TimeRange(
        min(word.time_range.start_seconds for word in words),
        max(word.time_range.end_seconds for word in words),
    )


def _first_segment_word(segment: Segment) -> Word | None:
    for sentence in segment.sentences:
        if sentence.words:
            return sentence.words[0]

    return None


def _last_segment_word(segment: Segment) -> Word | None:
    for sentence in reversed(segment.sentences):
        if sentence.words:
            return sentence.words[-1]

    return None


def _is_japanese_content_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


@dataclass(frozen=True, slots=True)
class DisabledQwenRepairer:
    """Keep aligned segments unchanged when Qwen repair is disabled."""

    def repair(self, request: QwenRepairRequest) -> QwenRepair:
        if not isinstance(request, QwenRepairRequest):
            raise TypeError("request must be a QwenRepairRequest.")

        return QwenRepair(
            source_path=request.source_path,
            segments=request.segments,
            decisions=tuple(
                QwenRepairDecision(
                    segment_position=segment.position,
                    original_text=segment.text,
                    raw_text="",
                    candidate_text=segment.text,
                    selected_text=segment.text,
                    accepted=True,
                    reason=QwenRepairSafetyReason.DISABLED.value,
                    length_delta_ratio=0.0,
                    content_change_ratio=0.0,
                )
                for segment in request.segments
            ),
        )


@dataclass(frozen=True, slots=True)
class PassthroughQwenRepairer:
    """Keep aligned segments mostly unchanged while cleaning boundary duplicates."""

    def repair(self, request: QwenRepairRequest) -> QwenRepair:
        if not isinstance(request, QwenRepairRequest):
            raise TypeError("request must be a QwenRepairRequest.")

        return QwenRepair(
            source_path=request.source_path,
            segments=_remove_cross_segment_leading_duplicates(request.segments),
        )


@dataclass(slots=True)
class LlamaCppQwenRepairer:
    """Repair Japanese transcript text with a local Qwen GGUF model."""

    model_path: Path
    context_size: int = DEFAULT_QWEN_CONTEXT
    threads: int = DEFAULT_QWEN_THREADS
    gpu_layers: int = DEFAULT_QWEN_GPU_LAYERS
    max_tokens: int = DEFAULT_QWEN_MAX_TOKENS
    temperature: float = DEFAULT_QWEN_TEMPERATURE
    top_p: float = DEFAULT_QWEN_TOP_P
    repeat_penalty: float = DEFAULT_QWEN_REPEAT_PENALTY
    safety_policy: QwenRepairSafetyPolicy = QwenRepairSafetyPolicy()
    _model: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if not isinstance(self.safety_policy, QwenRepairSafetyPolicy):
            raise TypeError("safety_policy must be a QwenRepairSafetyPolicy.")

    def repair(self, request: QwenRepairRequest) -> QwenRepair:
        if not isinstance(request, QwenRepairRequest):
            raise TypeError("request must be a QwenRepairRequest.")

        cleaned_segments = _remove_cross_segment_leading_duplicates(request.segments)
        repaired_segments: list[Segment] = []
        decisions: list[QwenRepairDecision] = []
        for segment in cleaned_segments:
            repaired_segment, decision = self._repair_segment(segment)
            repaired_segments.append(repaired_segment)
            decisions.append(decision)

        return QwenRepair(
            source_path=request.source_path,
            segments=tuple(repaired_segments),
            decisions=tuple(decisions),
        )

    def _repair_segment(
        self,
        segment: Segment,
    ) -> tuple[Segment, QwenRepairDecision]:
        repaired_text, decision = self._repair_text(
            segment.position,
            segment.text,
            (),
        )
        speaker_id = _segment_speaker_id(segment)
        words = tuple(
            word
            for sentence in segment.sentences
            for word in sentence.words
        )
        sentence = Sentence(
            text=repaired_text,
            time_range=segment.time_range,
            words=words,
            speaker_id=speaker_id,
        )
        return (
            Segment(
                position=segment.position,
                text=repaired_text,
                time_range=segment.time_range,
                sentences=(sentence,),
                speaker_id=speaker_id,
            ),
            decision,
        )

    def _repair_text(
        self,
        segment_position: int,
        current_text: str,
        context_hints: tuple[str, ...],
    ) -> tuple[str, QwenRepairDecision]:
        normalized_text = current_text.strip()
        if not normalized_text:
            return current_text, QwenRepairDecision(
                segment_position=segment_position,
                original_text=current_text,
                raw_text="",
                candidate_text="",
                selected_text=current_text,
                accepted=True,
                reason=QwenRepairSafetyReason.ACCEPTED.value,
                length_delta_ratio=0.0,
                content_change_ratio=0.0,
            )

        repaired_text, decision = self._repair_with_prompt(
            prompt=self._build_prompt(normalized_text, context_hints),
            original_text=normalized_text,
        )

        if not decision.accepted:
            _LOGGER.info(
                "Rejected unsafe Qwen repair: reason=%s length_delta_ratio=%.3f "
                "content_change_ratio=%.3f",
                decision.reason.value,
                decision.length_delta_ratio,
                decision.content_change_ratio,
            )
            retry_text, retry_decision = self._repair_with_prompt(
                prompt=self._build_semantic_retry_prompt(normalized_text),
                original_text=normalized_text,
            )
            if (
                retry_decision.accepted
                and retry_decision.selected_text != normalized_text
            ):
                return retry_decision.selected_text, _qwen_repair_decision(
                    segment_position=segment_position,
                    raw_text=retry_text,
                    decision=retry_decision,
                )

        return decision.selected_text, _qwen_repair_decision(
            segment_position=segment_position,
            raw_text=repaired_text,
            decision=decision,
        )

    def _repair_with_prompt(
        self,
        *,
        prompt: str,
        original_text: str,
    ) -> tuple[str, QwenRepairSafetyDecision]:
        response = self._load_model()(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            stop=("<|im_end|>",),
        )
        choices = response.get("choices", ())
        if not choices:
            return "", self.safety_policy.decide(original_text, "")

        repaired_text = str(choices[0].get("text", "")).strip()
        if not repaired_text:
            return repaired_text, self.safety_policy.decide(original_text, "")

        structured_candidate = _parse_structured_repair_output(repaired_text)
        if structured_candidate is None:
            return repaired_text, _invalid_edit_candidate_decision(
                original_text,
                self._clean_output(repaired_text),
            )

        return repaired_text, _structured_repair_decision(
            original_text,
            structured_candidate,
            self.safety_policy,
        )

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise QwenModelNotFoundError(self.model_path)

        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise QwenDependencyError() from error

            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                n_threads=self.threads,
                n_gpu_layers=self.gpu_layers,
                verbose=False,
            )

        return self._model

    def _build_prompt(self, current_text: str, context_hints: tuple[str, ...]) -> str:
        metadata_lines = "\n".join(
            f"- {line}" for line in _base_prompt_metadata_lines(context_hints)
        )
        return f"""<|im_start|>system
あなたは日本語字幕修正AIです。
Whisper音声認識の誤認識だけを修正してください。
音声にない語を追加したり、音声にある語を削除したりしないでください。
言い換え、要約、説明、補足は禁止です。
意味を変えない範囲で、日本語の句読点（。、？！）は補って構いません。
不自然な語や誤変換らしい語を削除してはいけません。
現在の文でASR誤認識らしい箇所だけを、同音または非常に近い発音の候補へ置換してください。
意味だけから自然な語を推測してはいけません。
数字、級名、試験名、固有名詞、見出し語などの情報単位は保持してください。
修正済み全文は出力しないでください。
必ずJSONオブジェクトだけを出力してください。
JSON形式:
{{"edits":[{{"type":"replace","from":"<修正対象テキスト内の連続文字列>","from_reading":"<カタカナ読み>","candidates":[{{"to":"<同音または近音候補>","to_reading":"<カタカナ読み>","reason":"<短い理由>","confidence":0.0}}]}}],"punctuation":[{{"after":"<修正対象テキスト内の連続文字列>","mark":"。"}}]}}
edits は type="replace" のみ使用できます。delete、insert、rewrite は禁止です。
各 edit の candidates は最大3件まで、発音が近い順に出してください。
to_reading が from_reading と大きく異なる候補は出さないでください。
同義語への言い換えや自然化ではなく、ASR誤認識として説明できる近い発音の置換だけを提案してください。
英数字や級名を普通語へ置換してはいけません。
不確かな場合は {{"edits":[],"punctuation":[]}} を出力してください。
<|im_end|>
<|im_start|>user
文脈メタデータ:
{metadata_lines}

修正対象テキスト:
{current_text}
<|im_end|>
<|im_start|>assistant
"""

    def _build_semantic_retry_prompt(self, current_text: str) -> str:
        return f"""<|im_start|>system
あなたは日本語ASR修正レビューAIです。
現在の字幕テキストだけを見て、意味的に明らかに不自然な語を探してください。
不自然な語がある場合だけ、同音または非常に近い発音で、文脈に合う置換候補を提案してください。
候補を思いつかない場合は修正しないでください。
削除、追加、言い換え、要約、自然化は禁止です。
数字、英字、級名、固有名詞などの情報単位は保持してください。
修正済み全文は出力しないでください。
必ずJSONオブジェクトだけを出力してください。
JSON形式:
{{"edits":[{{"type":"replace","from":"<修正対象テキスト内の連続文字列>","from_reading":"<カタカナ読み>","candidates":[{{"to":"<同音または近音候補>","to_reading":"<カタカナ読み>","reason":"<短い理由>","confidence":0.0}}]}}],"punctuation":[]}}
<|im_end|>
<|im_start|>user
修正対象テキスト:
{current_text}
<|im_end|>
<|im_start|>assistant
"""

    def _clean_output(self, text: str) -> str:
        return " ".join(text.strip().split())


def _base_prompt_metadata_lines(context_hints: tuple[str, ...]) -> tuple[str, ...]:
    return (
        "入力単位: current_segment_only",
        "前後セグメント原文: omitted",
        "目的: subtitle_text_repair",
        *context_hints,
    )


def _parse_structured_repair_output(
    text: str,
) -> _StructuredRepairCandidate | None:
    decoded = _decode_json_object(text)
    if not isinstance(decoded, dict):
        return None

    replacements: list[_StructuredReplacementCandidate] = []
    punctuations: list[_StructuredPunctuationCandidate] = []
    operation_count = 0

    raw_edits = decoded.get("edits", ())
    if isinstance(raw_edits, list):
        operation_count += len(raw_edits)
        for raw_edit in raw_edits:
            replacement = _parse_structured_replacement_candidate(raw_edit)
            if replacement is not None:
                replacements.append(replacement)
    elif raw_edits:
        operation_count += 1

    raw_punctuations = decoded.get("punctuation", ())
    if isinstance(raw_punctuations, list):
        operation_count += len(raw_punctuations)
        for raw_punctuation in raw_punctuations:
            punctuation = _parse_structured_punctuation_candidate(raw_punctuation)
            if punctuation is not None:
                punctuations.append(punctuation)
    elif raw_punctuations:
        operation_count += 1

    return _StructuredRepairCandidate(
        replacements=tuple(replacements),
        punctuations=tuple(punctuations),
        operation_count=operation_count,
    )


def _decode_json_object(text: str) -> Any | None:
    stripped = _strip_markdown_json_fence(text.strip())
    if not stripped:
        return None

    start_index = stripped.find("{")
    if start_index < 0:
        return None

    try:
        decoded, _ = json.JSONDecoder().raw_decode(stripped[start_index:])
    except json.JSONDecodeError:
        return None

    return decoded


def _strip_markdown_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if not lines:
        return text

    if lines[-1].strip() == "```":
        lines = lines[1:-1]
    else:
        lines = lines[1:]

    return "\n".join(lines).strip()


def _parse_structured_replacement_candidate(
    raw_edit: object,
) -> _StructuredReplacementCandidate | None:
    if not isinstance(raw_edit, dict):
        return None

    source_reading = _string_field(raw_edit.get("from_reading"))
    return _StructuredReplacementCandidate(
        edit_type=_string_field(raw_edit.get("type")).lower(),
        source_text=_string_field(raw_edit.get("from")),
        source_reading=source_reading,
        options=_parse_structured_replacement_options(
            raw_edit,
            source_reading,
        ),
    )


def _parse_structured_replacement_options(
    raw_edit: dict[object, object],
    source_reading: str,
) -> tuple[_StructuredReplacementOption, ...]:
    options: list[_StructuredReplacementOption] = []
    raw_candidates = raw_edit.get("candidates")
    if isinstance(raw_candidates, list):
        for raw_candidate in raw_candidates:
            option = _parse_structured_replacement_option(
                raw_candidate,
                fallback_reason=_string_field(raw_edit.get("reason")),
                fallback_confidence=_optional_float_field(raw_edit.get("confidence")),
                fallback_source_reading=source_reading,
            )
            if option is not None:
                options.append(option)

    direct_option = _parse_structured_replacement_option(
        raw_edit,
        fallback_reason="",
        fallback_confidence=None,
        fallback_source_reading=source_reading,
    )
    if direct_option is not None and not options:
        options.append(direct_option)

    return tuple(options)


def _parse_structured_replacement_option(
    raw_candidate: object,
    *,
    fallback_reason: str,
    fallback_confidence: float | None,
    fallback_source_reading: str,
) -> _StructuredReplacementOption | None:
    if not isinstance(raw_candidate, dict):
        return None

    replacement_text = _string_field(raw_candidate.get("to"))
    if not replacement_text:
        return None

    reason = _string_field(raw_candidate.get("reason")) or fallback_reason
    confidence = _optional_float_field(raw_candidate.get("confidence"))
    if confidence is None:
        confidence = fallback_confidence

    return _StructuredReplacementOption(
        replacement_text=replacement_text,
        reason=reason,
        confidence=confidence,
        source_reading=_string_field(raw_candidate.get("from_reading"))
        or fallback_source_reading,
        replacement_reading=_string_field(raw_candidate.get("to_reading")),
    )


def _parse_structured_punctuation_candidate(
    raw_punctuation: object,
) -> _StructuredPunctuationCandidate | None:
    if not isinstance(raw_punctuation, dict):
        return None

    return _StructuredPunctuationCandidate(
        after_text=_string_field(raw_punctuation.get("after")),
        mark=_string_field(raw_punctuation.get("mark")),
    )


def _string_field(value: object) -> str:
    if not isinstance(value, str):
        return ""

    return value.strip()


def _optional_float_field(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _structured_repair_decision(
    original_text: str,
    candidate: _StructuredRepairCandidate,
    safety_policy: QwenRepairSafetyPolicy,
) -> QwenRepairSafetyDecision:
    original = original_text.strip()
    application = _apply_structured_repair_candidate(
        original,
        candidate,
        safety_policy,
    )
    if application.applied_count == 0:
        if candidate.operation_count == 0:
            return QwenRepairSafetyDecision(
                original_text=original,
                candidate_text=original,
                accepted=True,
                reason=QwenRepairSafetyReason.ACCEPTED,
                length_delta_ratio=0.0,
                content_change_ratio=0.0,
            )

        if application.last_rejected_decision is not None:
            return application.last_rejected_decision

        return _invalid_edit_candidate_decision(
            original,
            application.candidate_text,
        )

    decision = safety_policy.decide(original, application.candidate_text)
    if (
        application.rejected_count > 0
        and decision.accepted
        and _normalize_text_for_safety(original)
        == _normalize_text_for_safety(application.candidate_text)
    ):
        return QwenRepairSafetyDecision(
            original_text=decision.original_text,
            candidate_text=decision.candidate_text,
            accepted=True,
            reason=QwenRepairSafetyReason.PUNCTUATION_FALLBACK,
            length_delta_ratio=decision.length_delta_ratio,
            content_change_ratio=decision.content_change_ratio,
        )

    return decision


def _apply_structured_repair_candidate(
    original_text: str,
    candidate: _StructuredRepairCandidate,
    safety_policy: QwenRepairSafetyPolicy,
) -> _StructuredRepairApplication:
    working_text = original_text
    applied_count = 0
    rejected_count = 0
    last_rejected_decision: QwenRepairSafetyDecision | None = None

    for replacement in candidate.replacements:
        replacement_text, rejected_decision = _apply_replacement_candidate(
            working_text,
            replacement,
            safety_policy,
        )
        if replacement_text is None:
            rejected_count += 1
            if rejected_decision is not None:
                last_rejected_decision = rejected_decision
            continue

        working_text = replacement_text
        applied_count += 1

    for punctuation in candidate.punctuations:
        punctuation_text = _apply_punctuation_candidate(
            working_text,
            punctuation,
        )
        if punctuation_text is None:
            rejected_count += 1
            continue

        working_text = punctuation_text
        applied_count += 1

    return _StructuredRepairApplication(
        candidate_text=working_text,
        applied_count=applied_count,
        rejected_count=rejected_count,
        last_rejected_decision=last_rejected_decision,
    )


def _apply_replacement_candidate(
    text: str,
    replacement: _StructuredReplacementCandidate,
    safety_policy: QwenRepairSafetyPolicy,
) -> tuple[str | None, QwenRepairSafetyDecision | None]:
    if replacement.edit_type != "replace":
        return None, None

    source_text = replacement.source_text
    if not source_text or text.count(source_text) != 1:
        return None, None

    last_rejected_decision: QwenRepairSafetyDecision | None = None
    for option in replacement.options:
        if not _declared_readings_are_compatible(
            replacement.source_reading or option.source_reading,
            option.replacement_reading,
        ):
            continue

        replacement_text = option.replacement_text
        if not replacement_text or source_text == replacement_text:
            continue

        candidate_text = text.replace(source_text, replacement_text, 1)
        decision = safety_policy.decide(text, candidate_text)
        if decision.accepted:
            return decision.selected_text, None

        last_rejected_decision = decision

    return None, last_rejected_decision


def _declared_readings_are_compatible(
    source_reading: str,
    replacement_reading: str,
) -> bool:
    if not source_reading or not replacement_reading:
        return True

    normalized_source = _normalize_reading(source_reading)
    normalized_replacement = _normalize_reading(replacement_reading)
    if not normalized_source or not normalized_replacement:
        return True

    return (
        SequenceMatcher(None, normalized_source, normalized_replacement).ratio()
        >= _SAFE_REPLACEMENT_MIN_READING_SIMILARITY
    )


def _apply_punctuation_candidate(
    text: str,
    punctuation: _StructuredPunctuationCandidate,
) -> str | None:
    after_text = punctuation.after_text
    mark = punctuation.mark
    if not after_text or len(mark) != 1 or mark not in _PUNCTUATION_FALLBACK_MARKS:
        return None

    if text.count(after_text) != 1:
        return None

    insertion_index = text.index(after_text) + len(after_text)
    next_content_index = insertion_index
    while next_content_index < len(text) and text[next_content_index].isspace():
        next_content_index += 1

    if (
        next_content_index < len(text)
        and text[next_content_index] in _PUNCTUATION_FALLBACK_MARKS
    ):
        return None

    if mark in _PUNCTUATION_FALLBACK_TERMINAL_MARKS and not (
        text_ends_with_complete_predicate(text[:insertion_index])
    ):
        return None

    return text[:insertion_index] + mark + text[next_content_index:]


def _invalid_edit_candidate_decision(
    original_text: str,
    candidate_text: str,
) -> QwenRepairSafetyDecision:
    original = original_text.strip()
    candidate = candidate_text.strip()
    original_core = _normalize_text_for_safety(original)
    candidate_core = _normalize_text_for_safety(candidate)
    return QwenRepairSafetyDecision(
        original_text=original,
        candidate_text=candidate,
        accepted=False,
        reason=QwenRepairSafetyReason.INVALID_EDIT_CANDIDATE,
        length_delta_ratio=_length_delta_ratio(original_core, candidate_core),
        content_change_ratio=_content_change_ratio(original_core, candidate_core),
    )


def _segment_speaker_id(segment: Segment) -> str | None:
    if segment.speaker_id is not None:
        return segment.speaker_id

    for sentence in segment.sentences:
        if sentence.speaker_id is not None:
            return sentence.speaker_id

    return None


def _qwen_repair_decision(
    segment_position: int,
    raw_text: str,
    decision: QwenRepairSafetyDecision,
) -> QwenRepairDecision:
    return QwenRepairDecision(
        segment_position=segment_position,
        original_text=decision.original_text,
        raw_text=raw_text,
        candidate_text=decision.candidate_text,
        selected_text=decision.selected_text,
        accepted=decision.accepted,
        reason=decision.reason.value,
        length_delta_ratio=decision.length_delta_ratio,
        content_change_ratio=decision.content_change_ratio,
    )


__all__ = [
    "DEFAULT_QWEN_CONTEXT",
    "DEFAULT_QWEN_GPU_LAYERS",
    "DEFAULT_QWEN_MAX_TOKENS",
    "DEFAULT_QWEN_REPEAT_PENALTY",
    "DEFAULT_QWEN_REPAIR_MAX_CONTENT_CHANGE_RATIO",
    "DEFAULT_QWEN_REPAIR_MAX_LENGTH_DELTA_RATIO",
    "DEFAULT_QWEN_TEMPERATURE",
    "DEFAULT_QWEN_TOP_P",
    "DEFAULT_QWEN_THREADS",
    "DisabledQwenRepairer",
    "LlamaCppQwenRepairer",
    "PassthroughQwenRepairer",
    "QwenDependencyError",
    "QwenModelNotFoundError",
    "QwenRepairSafetyDecision",
    "QwenRepairSafetyPolicy",
    "QwenRepairSafetyReason",
]

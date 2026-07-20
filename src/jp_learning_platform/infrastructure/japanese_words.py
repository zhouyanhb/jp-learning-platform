"""Japanese word timing normalization adapters."""

from __future__ import annotations

from difflib import SequenceMatcher
from dataclasses import dataclass, field
import unicodedata
from typing import Any, Callable, Protocol

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.workflow.japanese_word_stage import (
    JapaneseWordNormalization,
    JapaneseWordNormalizationRequest,
)

DEFAULT_SUDACHI_SPLIT_MODE = "C"
MAX_RECONCILIATION_LENGTH_DELTA_RATIO = 0.25
_VERB_CHAIN_CONNECTIVE_PARTICLES = {"て", "で", "ば"}
_VERB_CHAIN_POST_TE_PARTICLES = {"も"}
_VERB_CHAIN_AUXILIARY_BASES = {
    "ある",
    "いく",
    "いる",
    "おく",
    "くる",
    "させる",
    "しまう",
    "せる",
    "たい",
    "た",
    "だ",
    "ない",
    "ぬ",
    "ます",
    "みる",
    "られる",
    "れる",
}
_VERB_CHAIN_HELPER_VERB_BASES = {
    "ある",
    "いく",
    "いる",
    "おく",
    "くる",
    "しまう",
    "みる",
}
_VERB_CHAIN_STOP_VERB_BASES = {"くださる"}
_PARTICLE_EXPRESSION_PATTERNS = (
    ("に", "つい", "て", "は"),
    ("に", "つい", "て", "も"),
    ("に", "よっ", "て", "は"),
    ("に", "よっ", "て", "も"),
    ("に", "とっ", "て", "は"),
    ("に", "とっ", "て", "も"),
    ("に", "対し", "て", "は"),
    ("に", "対し", "て", "も"),
    ("に", "関し", "て", "は"),
    ("に", "関し", "て", "も"),
    ("と", "し", "て", "は"),
    ("と", "し", "て", "も"),
    ("だけ", "で", "は"),
    ("だけ", "で", "も"),
    ("から", "に", "は"),
    ("から", "で", "も"),
    ("まで", "で", "も"),
    ("に", "つい", "て"),
    ("に", "よっ", "て"),
    ("に", "とっ", "て"),
    ("に", "対し", "て"),
    ("に", "関し", "て"),
    ("に", "比べ", "て"),
    ("に", "おい", "て"),
    ("と", "し", "て"),
    ("まで", "に", "は"),
    ("まで", "に", "も"),
    ("まで", "に"),
    ("で", "は"),
    ("で", "も"),
    ("に", "は"),
    ("に", "も"),
    ("に", "で", "も"),
    ("と", "は"),
    ("と", "も"),
    ("へ", "は"),
    ("へ", "も"),
    ("を", "は"),
    ("を", "も"),
    ("から", "は"),
    ("から", "も"),
    ("まで", "は"),
    ("まで", "も"),
    ("だけ", "で"),
    ("だけ", "は"),
    ("だけ", "も"),
    ("より", "は"),
    ("より", "も"),
    ("ほど", "は"),
    ("ほど", "も"),
    ("ばかり", "で"),
    ("ばかり", "は"),
    ("ばかり", "も"),
)

_HIRAGANA_WORDS = tuple(
    sorted(
        {
            "くださいね",
            "ください",
            "ましょう",
            "ました",
            "ません",
            "でしょう",
            "しましょう",
            "しました",
            "しません",
            "でした",
            "します",
            "ます",
            "です",
            "から",
            "なら",
            "ので",
            "けど",
            "では",
            "には",
            "とは",
            "まで",
            "より",
            "でも",
            "だけ",
            "ほど",
            "いい",
            "ない",
            "して",
            "した",
            "する",
            "いる",
            "ある",
            "なる",
            "まず",
            "つぎ",
            "とき",
            "これ",
            "それ",
            "あれ",
            "どれ",
            "ここ",
            "そこ",
            "どこ",
            "が",
            "を",
            "に",
            "へ",
            "で",
            "と",
            "は",
            "も",
            "の",
            "か",
            "ね",
            "よ",
        },
        key=len,
        reverse=True,
    )
)


class JapaneseTokenizerDependencyError(RuntimeError):
    """Raised when the requested Japanese tokenizer is unavailable."""

    def __init__(self) -> None:
        super().__init__(
            "sudachipy and sudachidict-core are required for Japanese word "
            "normalization. Install them with: python -m pip install -r requirements.txt"
        )


class JapaneseTextTokenizer(Protocol):
    """Japanese tokenizer contract used by word timing normalization."""

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Return Japanese word surfaces for text."""


@dataclass(frozen=True, slots=True)
class _JapaneseMorpheme:
    surface: str
    part_of_speech: tuple[str, ...]
    dictionary_form: str


@dataclass(slots=True)
class SudachiJapaneseTokenizer:
    """Tokenize Japanese text with SudachiPy."""

    split_mode: str = DEFAULT_SUDACHI_SPLIT_MODE
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _mode: Any | None = field(default=None, init=False, repr=False)

    def tokenize(self, text: str) -> tuple[str, ...]:
        morphemes = self._morphemes(text)
        return _learning_word_tokens_from_morphemes(
            morphemes,
            is_stable_compound=self._is_stable_compound,
        )

    def _morphemes(self, text: str) -> tuple[_JapaneseMorpheme, ...]:
        tokenizer, mode = self._load_tokenizer()
        return tuple(
            _JapaneseMorpheme(
                surface=surface,
                part_of_speech=tuple(morpheme.part_of_speech()),
                dictionary_form=str(morpheme.dictionary_form()),
            )
            for morpheme in tokenizer.tokenize(text, mode)
            if (surface := str(morpheme.surface()).strip())
        )

    def _is_stable_compound(self, text: str) -> bool:
        if not _compact_text(text):
            return False

        tokenizer, mode = self._load_tokenizer()
        morphemes = tuple(tokenizer.tokenize(text, mode))
        return len(morphemes) == 1 and str(morphemes[0].surface()) == text

    def _load_tokenizer(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._mode is None:
            try:
                from sudachipy import dictionary
                from sudachipy import tokenizer
            except ImportError as error:
                raise JapaneseTokenizerDependencyError() from error

            mode_name = self.split_mode.strip().upper()
            try:
                self._mode = getattr(tokenizer.Tokenizer.SplitMode, mode_name)
            except AttributeError as error:
                raise ValueError(f"Unknown Sudachi split mode: {self.split_mode}") from error

            try:
                self._tokenizer = dictionary.Dictionary().create()
            except Exception as error:
                raise JapaneseTokenizerDependencyError() from error

        return self._tokenizer, self._mode


@dataclass(frozen=True, slots=True)
class HeuristicJapaneseTokenizer:
    """Small dependency-free fallback for common Japanese word boundaries."""

    def tokenize(self, text: str) -> tuple[str, ...]:
        tokens: list[str] = []
        for phrase in str(text).split():
            tokens.extend(_heuristic_phrase_tokens(phrase))
        return tuple(token for token in tokens if _compact_text(token))


@dataclass(slots=True)
class DefaultJapaneseTokenizer:
    """Use SudachiPy when available, otherwise fall back conservatively."""

    sudachi_tokenizer: SudachiJapaneseTokenizer = field(
        default_factory=SudachiJapaneseTokenizer
    )
    fallback_tokenizer: HeuristicJapaneseTokenizer = field(
        default_factory=HeuristicJapaneseTokenizer
    )
    _use_fallback: bool = field(default=False, init=False, repr=False)

    def tokenize(self, text: str) -> tuple[str, ...]:
        if not self._use_fallback:
            try:
                return self.sudachi_tokenizer.tokenize(text)
            except JapaneseTokenizerDependencyError:
                self._use_fallback = True

        return self.fallback_tokenizer.tokenize(text)


@dataclass(slots=True)
class JapaneseWordTimingNormalizer:
    """Map final Japanese text back onto ASR/aligner timing units."""

    tokenizer: JapaneseTextTokenizer | None = None

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = DefaultJapaneseTokenizer()

    def normalize(
        self,
        request: JapaneseWordNormalizationRequest,
    ) -> JapaneseWordNormalization:
        if not isinstance(request, JapaneseWordNormalizationRequest):
            raise TypeError("request must be a JapaneseWordNormalizationRequest.")

        return JapaneseWordNormalization(
            source_path=request.source_path,
            segments=tuple(
                self._normalize_segment(segment) for segment in request.segments
            ),
        )

    def _normalize_segment(self, segment: Segment) -> Segment:
        sentences = segment.sentences or (
            Sentence(
                text=segment.text,
                time_range=segment.time_range,
                words=(),
                speaker_id=segment.speaker_id,
            ),
        )
        normalized_sentences = tuple(
            self._normalize_sentence(sentence) for sentence in sentences
        )
        start_seconds = min(
            segment.time_range.start_seconds,
            *(
                sentence.time_range.start_seconds
                for sentence in normalized_sentences
            ),
        )
        end_seconds = max(
            segment.time_range.end_seconds,
            *(
                sentence.time_range.end_seconds
                for sentence in normalized_sentences
            ),
        )
        return Segment(
            position=segment.position,
            text=segment.text,
            time_range=TimeRange(start_seconds, end_seconds),
            sentences=normalized_sentences,
            speaker_id=segment.speaker_id,
        )

    def _normalize_sentence(self, sentence: Sentence) -> Sentence:
        words = _retime_words_by_tokens(
            tokens=self.tokenizer.tokenize(sentence.text) if self.tokenizer else (),
            words=sentence.words,
        )
        if not words:
            return sentence

        time_range = TimeRange(
            min(
                sentence.time_range.start_seconds,
                *(word.time_range.start_seconds for word in words),
            ),
            max(
                sentence.time_range.end_seconds,
                *(word.time_range.end_seconds for word in words),
            ),
        )
        return Sentence(
            text=sentence.text,
            time_range=time_range,
            words=words,
            speaker_id=sentence.speaker_id,
        )


def _learning_word_tokens_from_morphemes(
    morphemes: tuple[_JapaneseMorpheme, ...],
    is_stable_compound: Callable[[str], bool],
) -> tuple[str, ...]:
    tokens: list[str] = []
    index = 0
    while index < len(morphemes):
        verb_chain_end = _verb_chain_end(morphemes, index)
        if verb_chain_end > index + 1:
            tokens.append(_morphemes_text(morphemes[index:verb_chain_end]))
            index = verb_chain_end
            continue

        particle_expression_end = _particle_expression_end(morphemes, index)
        if particle_expression_end > index + 1:
            tokens.append(_morphemes_text(morphemes[index:particle_expression_end]))
            index = particle_expression_end
            continue

        numeric_chain_end = _numeric_counter_chain_end(morphemes, index)
        if numeric_chain_end > index + 1:
            tokens.append(_morphemes_text(morphemes[index:numeric_chain_end]))
            index = numeric_chain_end
            continue

        if (
            index + 1 < len(morphemes)
            and _can_merge_stable_nominal_pair(
                morphemes[index],
                morphemes[index + 1],
                is_stable_compound,
            )
        ):
            tokens.append(_morphemes_text(morphemes[index : index + 2]))
            index += 2
            continue

        tokens.append(morphemes[index].surface)
        index += 1

    return tuple(token for token in tokens if _compact_text(token))


def _verb_chain_end(
    morphemes: tuple[_JapaneseMorpheme, ...],
    index: int,
) -> int:
    if _starts_sahen_verb(morphemes, index):
        return _consume_verb_chain_tail(morphemes, index + 2)

    if _is_verb_morpheme(morphemes[index]):
        return _consume_verb_chain_tail(morphemes, index + 1)

    return index + 1


def _starts_sahen_verb(
    morphemes: tuple[_JapaneseMorpheme, ...],
    index: int,
) -> bool:
    return (
        index + 1 < len(morphemes)
        and _is_sahen_nominal(morphemes[index])
        and _is_suru_verb(morphemes[index + 1])
    )


def _consume_verb_chain_tail(
    morphemes: tuple[_JapaneseMorpheme, ...],
    index: int,
) -> int:
    end = index
    while end < len(morphemes):
        morpheme = morphemes[end]
        if _is_verb_chain_auxiliary(morpheme):
            end += 1
            continue

        if _is_verb_chain_connective_particle(morpheme):
            end += 1
            if end < len(morphemes) and _is_post_te_particle(morphemes[end]):
                end += 1
            continue

        if _is_helper_verb_after_connective(morphemes, end):
            end += 1
            continue

        break

    return end


def _particle_expression_end(
    morphemes: tuple[_JapaneseMorpheme, ...],
    index: int,
) -> int:
    if not _can_start_particle_expression(morphemes[index]):
        return index + 1

    for pattern in _PARTICLE_EXPRESSION_PATTERNS:
        end = index + len(pattern)
        if end > len(morphemes):
            continue

        if _morpheme_surfaces(morphemes[index:end]) != pattern:
            continue

        if _valid_particle_expression(morphemes[index:end]):
            return end

    return index + 1


def _can_start_particle_expression(morpheme: _JapaneseMorpheme) -> bool:
    return _pos(morpheme, 0) == "助詞"


def _valid_particle_expression(
    morphemes: tuple[_JapaneseMorpheme, ...],
) -> bool:
    return (
        _pos(morphemes[0], 0) == "助詞"
        and any(_pos(morpheme, 0) == "助詞" for morpheme in morphemes[1:])
        and not any(_is_boundary_symbol(morpheme) for morpheme in morphemes)
    )


def _morpheme_surfaces(
    morphemes: tuple[_JapaneseMorpheme, ...],
) -> tuple[str, ...]:
    return tuple(morpheme.surface for morpheme in morphemes)


def _is_sahen_nominal(morpheme: _JapaneseMorpheme) -> bool:
    return _pos(morpheme, 0) == "名詞" and _pos(morpheme, 2) == "サ変可能"


def _is_suru_verb(morpheme: _JapaneseMorpheme) -> bool:
    return _is_verb_morpheme(morpheme) and morpheme.dictionary_form == "する"


def _is_verb_morpheme(morpheme: _JapaneseMorpheme) -> bool:
    return _pos(morpheme, 0) == "動詞"


def _is_verb_chain_auxiliary(morpheme: _JapaneseMorpheme) -> bool:
    return (
        _pos(morpheme, 0) == "助動詞"
        and morpheme.dictionary_form in _VERB_CHAIN_AUXILIARY_BASES
    )


def _is_verb_chain_connective_particle(morpheme: _JapaneseMorpheme) -> bool:
    return (
        _pos(morpheme, 0) == "助詞"
        and _pos(morpheme, 1) == "接続助詞"
        and morpheme.surface in _VERB_CHAIN_CONNECTIVE_PARTICLES
    )


def _is_post_te_particle(morpheme: _JapaneseMorpheme) -> bool:
    return (
        _pos(morpheme, 0) == "助詞"
        and morpheme.surface in _VERB_CHAIN_POST_TE_PARTICLES
    )


def _is_helper_verb_after_connective(
    morphemes: tuple[_JapaneseMorpheme, ...],
    index: int,
) -> bool:
    morpheme = morphemes[index]
    if not _is_verb_morpheme(morpheme):
        return False

    if morpheme.dictionary_form in _VERB_CHAIN_STOP_VERB_BASES:
        return False

    if morpheme.dictionary_form not in _VERB_CHAIN_HELPER_VERB_BASES:
        return False

    previous = morphemes[index - 1] if index > 0 else None
    return previous is not None and previous.surface in {"て", "で"}


def _numeric_counter_chain_end(
    morphemes: tuple[_JapaneseMorpheme, ...],
    index: int,
) -> int:
    if morphemes[index].surface == "第" and index + 1 < len(morphemes):
        if not _is_number_morpheme(morphemes[index + 1]):
            return index + 1

        end = index + 2
        if end < len(morphemes) and _is_counter_morpheme(morphemes[end]):
            end += 1
        return end

    if (
        _is_number_morpheme(morphemes[index])
        and index + 1 < len(morphemes)
        and _is_counter_morpheme(morphemes[index + 1])
    ):
        return index + 2

    return index + 1


def _is_number_morpheme(morpheme: _JapaneseMorpheme) -> bool:
    return _pos(morpheme, 0) == "名詞" and _pos(morpheme, 1) == "数詞"


def _is_counter_morpheme(morpheme: _JapaneseMorpheme) -> bool:
    return _pos(morpheme, 0) == "名詞" and _pos(morpheme, 2) == "助数詞可能"


def _can_merge_stable_nominal_pair(
    left: _JapaneseMorpheme,
    right: _JapaneseMorpheme,
    is_stable_compound: Callable[[str], bool],
) -> bool:
    if not _is_nominal_piece(left) or not _is_nominal_piece(right):
        return False

    if not (_is_nominal_suffix(left) or _is_nominal_suffix(right)):
        return False

    compound = f"{left.surface}{right.surface}"
    if not _compact_text(compound):
        return False

    return bool(is_stable_compound(compound))


def _is_nominal_piece(morpheme: _JapaneseMorpheme) -> bool:
    return _pos(morpheme, 0) in {"名詞", "接尾辞"}


def _is_nominal_suffix(morpheme: _JapaneseMorpheme) -> bool:
    return _pos(morpheme, 0) == "接尾辞"


def _is_boundary_symbol(morpheme: _JapaneseMorpheme) -> bool:
    return _pos(morpheme, 0) == "補助記号"


def _morphemes_text(morphemes: tuple[_JapaneseMorpheme, ...]) -> str:
    return "".join(morpheme.surface for morpheme in morphemes)


def _pos(morpheme: _JapaneseMorpheme, index: int) -> str:
    if index >= len(morpheme.part_of_speech):
        return ""

    return morpheme.part_of_speech[index]


def _retime_words_by_tokens(
    tokens: tuple[str, ...],
    words: tuple[Word, ...],
) -> tuple[Word, ...]:
    if not tokens or not words:
        return words

    source_units = _timed_source_characters(words)
    target_characters = tuple(
        character
        for token in tokens
        for character in _compact_text(token)
        if _is_timed_character(character)
    )
    if not source_units or not target_characters:
        return words

    if not _safe_to_reconcile(len(source_units), len(target_characters)):
        return words

    source_text = "".join(unit.text for unit in source_units)
    target_text = "".join(target_characters)
    source_indexes_by_target = _align_target_to_source(source_text, target_text)
    if len(source_indexes_by_target) != len(target_characters):
        return words

    normalized_words: list[Word] = []
    target_character_index = 0
    previous_time_seconds = words[0].time_range.start_seconds
    for token in tokens:
        target_text = _compact_text(token)
        if not target_text:
            continue

        source_indexes: list[int] = []
        for character in target_text:
            if not _is_timed_character(character):
                continue

            source_index = source_indexes_by_target[target_character_index]
            if source_index is not None:
                source_indexes.append(source_index)
            target_character_index += 1

        source_indexes_tuple = tuple(dict.fromkeys(source_indexes))
        if source_indexes_tuple:
            source_word_indexes = tuple(
                dict.fromkeys(
                    source_units[source_index].word_index
                    for source_index in source_indexes_tuple
                )
            )
            source_words = tuple(words[index] for index in source_word_indexes)
            time_range = TimeRange(
                min(
                    source_units[source_index].time_range.start_seconds
                    for source_index in source_indexes_tuple
                ),
                max(
                    source_units[source_index].time_range.end_seconds
                    for source_index in source_indexes_tuple
                ),
            )
            previous_time_seconds = time_range.end_seconds
        else:
            source_words = ()
            time_range = TimeRange(previous_time_seconds, previous_time_seconds)

        normalized_words.append(
            Word(
                text=token,
                time_range=time_range,
                confidence=_mean_confidence(source_words),
                speaker_id=_common_speaker_id(source_words),
            )
        )

    if target_character_index != len(target_characters):
        return words

    return tuple(normalized_words)


@dataclass(frozen=True, slots=True)
class _SourceCharacter:
    text: str
    time_range: TimeRange
    word_index: int


def _timed_source_characters(words: tuple[Word, ...]) -> tuple[_SourceCharacter, ...]:
    characters: list[_SourceCharacter] = []
    for word_index, word in enumerate(words):
        timed_text = tuple(
            character
            for character in _compact_text(word.text)
            if _is_timed_character(character)
        )
        if not timed_text:
            continue

        duration_seconds = word.time_range.duration_seconds
        character_count = len(timed_text)
        for character_index, character in enumerate(timed_text):
            start_seconds = (
                word.time_range.start_seconds
                + duration_seconds * character_index / character_count
            )
            end_seconds = (
                word.time_range.start_seconds
                + duration_seconds * (character_index + 1) / character_count
            )
            characters.append(
                _SourceCharacter(
                    text=character,
                    time_range=TimeRange(start_seconds, end_seconds),
                    word_index=word_index,
                )
            )

    return tuple(characters)


def _safe_to_reconcile(source_length: int, target_length: int) -> bool:
    if source_length <= 0 or target_length <= 0:
        return False

    delta_ratio = abs(source_length - target_length) / max(source_length, target_length)
    return delta_ratio <= MAX_RECONCILIATION_LENGTH_DELTA_RATIO


def _align_target_to_source(
    source_text: str,
    target_text: str,
) -> tuple[int | None, ...]:
    if len(source_text) == len(target_text):
        return tuple(range(len(target_text)))

    source_indexes_by_target: list[int | None] = [None] * len(target_text)
    matcher = SequenceMatcher(a=source_text, b=target_text, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(target_end - target_start):
                source_indexes_by_target[target_start + offset] = source_start + offset
            continue

        if tag == "replace":
            for offset in range(target_end - target_start):
                source_indexes_by_target[target_start + offset] = _proportional_index(
                    source_start,
                    source_end,
                    offset,
                    target_end - target_start,
                )
            continue

        if tag == "insert":
            anchor_index = _insertion_anchor(source_start, len(source_text))
            for target_index in range(target_start, target_end):
                source_indexes_by_target[target_index] = anchor_index

    return tuple(source_indexes_by_target)


def _proportional_index(
    source_start: int,
    source_end: int,
    target_offset: int,
    target_length: int,
) -> int | None:
    source_length = source_end - source_start
    if source_length <= 0:
        return None

    if target_length <= 1:
        return source_start + source_length // 2

    return source_start + min(
        source_length - 1,
        round(target_offset * (source_length - 1) / (target_length - 1)),
    )


def _insertion_anchor(source_index: int, source_length: int) -> int | None:
    if source_length <= 0:
        return None

    if source_index <= 0:
        return 0

    return min(source_index - 1, source_length - 1)


def _heuristic_phrase_tokens(phrase: str) -> tuple[str, ...]:
    tokens: list[str] = []
    index = 0
    while index < len(phrase):
        char = phrase[index]
        if _is_hiragana(char):
            token, index = _consume_hiragana_token(phrase, index)
            tokens.append(token)
            continue

        char_type = _char_type(char)
        end = index + 1
        while end < len(phrase) and _char_type(phrase[end]) == char_type:
            end += 1

        tokens.append(phrase[index:end])
        index = end

    return tuple(tokens)


def _consume_hiragana_token(text: str, start: int) -> tuple[str, int]:
    remaining = text[start:]
    for word in _HIRAGANA_WORDS:
        if remaining.startswith(word):
            return word, start + len(word)

    end = start + 1
    while end < len(text) and _is_hiragana(text[end]):
        if any(text[end:].startswith(word) for word in _HIRAGANA_WORDS):
            break
        end += 1

    return text[start:end], end


def _char_type(char: str) -> str:
    if _is_kanji(char):
        return "kanji"
    if _is_hiragana(char):
        return "hiragana"
    if _is_katakana(char):
        return "katakana"
    if char.isascii() and char.isalnum():
        return "latin-number"
    if unicodedata.category(char).startswith("N"):
        return "number"
    return "symbol"


def _is_timed_character(char: str) -> bool:
    if not str(char).strip():
        return False

    return unicodedata.category(char)[0] not in {"P", "S"}


def _is_kanji(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"


def _is_hiragana(char: str) -> bool:
    return "\u3040" <= char <= "\u309f"


def _is_katakana(char: str) -> bool:
    return "\u30a0" <= char <= "\u30ff" or "\uff66" <= char <= "\uff9f"


def _compact_text(text: str) -> str:
    return "".join(str(text).split())


def _mean_confidence(words: tuple[Word, ...]) -> float | None:
    confidences = tuple(word.confidence for word in words if word.confidence is not None)
    if not confidences:
        return None

    return sum(confidences) / len(confidences)


def _common_speaker_id(words: tuple[Word, ...]) -> str | None:
    speaker_ids = tuple(
        dict.fromkeys(word.speaker_id for word in words if word.speaker_id is not None)
    )
    if len(speaker_ids) == 1:
        return speaker_ids[0]

    return None


__all__ = [
    "DEFAULT_SUDACHI_SPLIT_MODE",
    "DefaultJapaneseTokenizer",
    "HeuristicJapaneseTokenizer",
    "JapaneseTextTokenizer",
    "JapaneseTokenizerDependencyError",
    "JapaneseWordTimingNormalizer",
    "SudachiJapaneseTokenizer",
]

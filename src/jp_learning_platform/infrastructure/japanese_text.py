"""Japanese text heuristics shared by subtitle infrastructure adapters."""

from __future__ import annotations

from typing import Any
import unicodedata

JAPANESE_TERMINAL_MARKS = ("。", "？", "！", "?", "!")

_BOUNDARY_COMPLETE_PARTICLE_TYPES = ("終助詞",)
_BOUNDARY_CONTINUING_PARTICLE_TYPES = (
    "格助詞",
    "係助詞",
    "副助詞",
    "接続助詞",
    "準体助詞",
)
_PREDICATE_POS = ("動詞", "形容詞", "助動詞")
_TRAILING_CLOSING_MARKS = "」』）)]"
_TOKENIZER: Any | None = None
_TOKENIZER_MODE: Any | None = None
_SUDACHI_UNAVAILABLE = False


def text_ends_with_terminal_sentence_mark(text: str) -> bool:
    """Return whether text already ends with explicit sentence punctuation."""

    return _strip_trailing_closing_marks(text).endswith(JAPANESE_TERMINAL_MARKS)


def text_ends_with_complete_predicate(text: str) -> bool:
    """Return whether text ends with a complete Japanese predicate expression."""

    stripped_text = _strip_trailing_closing_marks(text)
    if not stripped_text:
        return False

    morphemes = _tokenize(stripped_text)
    if not morphemes:
        return False

    significant = tuple(
        (surface, pos) for surface, pos in morphemes if _has_content(surface)
    )
    if not significant:
        return False

    _, pos = significant[-1]
    primary_pos = pos[0] if len(pos) > 0 else ""
    sub_pos = pos[1] if len(pos) > 1 else ""
    if primary_pos == "助詞":
        return sub_pos in _BOUNDARY_COMPLETE_PARTICLE_TYPES

    if primary_pos in _PREDICATE_POS:
        return not _morpheme_has_connective_form(pos)

    if sub_pos in _BOUNDARY_CONTINUING_PARTICLE_TYPES:
        return False

    return False


def _strip_trailing_closing_marks(text: str) -> str:
    return text.strip().rstrip(_TRAILING_CLOSING_MARKS)


def _tokenize(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    tokenizer, mode = _load_tokenizer()
    if tokenizer is None or mode is None:
        return ()

    try:
        return tuple(
            (str(morpheme.surface()), tuple(morpheme.part_of_speech()))
            for morpheme in tokenizer.tokenize(text, mode)
        )
    except Exception:
        return ()


def _load_tokenizer() -> tuple[Any | None, Any | None]:
    global _SUDACHI_UNAVAILABLE
    global _TOKENIZER
    global _TOKENIZER_MODE

    if _SUDACHI_UNAVAILABLE:
        return None, None

    if _TOKENIZER is not None and _TOKENIZER_MODE is not None:
        return _TOKENIZER, _TOKENIZER_MODE

    try:
        from sudachipy import dictionary
        from sudachipy import tokenizer

        _TOKENIZER = dictionary.Dictionary().create()
        _TOKENIZER_MODE = tokenizer.Tokenizer.SplitMode.C
    except Exception:
        _SUDACHI_UNAVAILABLE = True
        return None, None

    return _TOKENIZER, _TOKENIZER_MODE


def _morpheme_has_connective_form(pos: tuple[str, ...]) -> bool:
    conjugation_form = pos[5] if len(pos) > 5 else ""
    return any(
        marker in conjugation_form
        for marker in ("仮定形", "連用形", "接続")
    )


def _has_content(text: str) -> bool:
    return any(
        not character.isspace()
        and not unicodedata.category(character).startswith("P")
        for character in text
    )


__all__ = [
    "JAPANESE_TERMINAL_MARKS",
    "text_ends_with_complete_predicate",
    "text_ends_with_terminal_sentence_mark",
]

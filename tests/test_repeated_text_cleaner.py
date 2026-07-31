from pathlib import Path

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import AuditableRepeatedTextCleaner
from jp_learning_platform.infrastructure import JapaneseMorpheme
from jp_learning_platform.workflow import RepeatedTextCleanupRequest


def _request(words: tuple[Word, ...]) -> RepeatedTextCleanupRequest:
    sentence = Sentence(
        text="".join(word.text for word in words),
        time_range=TimeRange(words[0].time_range.start_seconds, words[-1].time_range.end_seconds),
        words=words,
    )
    return RepeatedTextCleanupRequest(
        source_path=Path("input.mp3"),
        segments=(Segment(0, sentence.text, sentence.time_range, (sentence,)),),
    )


def _word(text: str, start: float, end: float) -> Word:
    return Word(text, TimeRange(start, end))


class _LexicalAnalyzer:
    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        return (JapaneseMorpheme(text, ("副詞", "*", "*")),)


def test_repeated_cleaner_removes_adjacent_aligned_sequence_with_audit() -> None:
    request = _request(
        (
            _word("問", 0.0, 0.1),
            _word("題", 0.1, 0.2),
            _word("1", 0.2, 0.3),
            _word("問", 0.3, 0.4),
            _word("題", 0.4, 0.5),
            _word("1", 0.5, 0.6),
            _word("で", 0.6, 0.7),
            _word("は", 0.7, 0.8),
        )
    )

    result = AuditableRepeatedTextCleaner().clean(request)

    assert result.segments[0].text == "問題1では"
    decision = result.decisions[0]
    assert decision.deleted_text == "問題1"
    assert (decision.deletion_start, decision.deletion_end) == (3, 6)
    assert decision.deleted_word_indexes == (3, 4, 5)
    assert decision.repetition_gap_seconds == 0.0


def test_repeated_cleaner_preserves_delayed_repetition() -> None:
    request = _request(
        (
            _word("天気", 0.0, 0.4),
            _word("です", 0.4, 0.8),
            _word("天気", 1.5, 1.9),
            _word("です", 1.9, 2.3),
        )
    )

    result = AuditableRepeatedTextCleaner().clean(request)

    assert result.segments == request.segments
    assert not result.decisions


def test_repeated_cleaner_preserves_single_character_repetition() -> None:
    request = _request(
        (_word("で", 0.0, 0.1), _word("で", 0.1, 0.2), _word("次", 0.2, 0.3))
    )

    result = AuditableRepeatedTextCleaner().clean(request)

    assert result.segments == request.segments
    assert not result.decisions


def test_repeated_cleaner_preserves_lexicalized_reduplication() -> None:
    request = _request(
        (
            _word("い", 0.0, 0.1),
            _word("よ", 0.1, 0.2),
            _word("い", 0.2, 0.3),
            _word("よ", 0.3, 0.4),
        )
    )

    result = AuditableRepeatedTextCleaner(
        morphological_analyzer=_LexicalAnalyzer()
    ).clean(request)

    assert result.segments == request.segments
    assert not result.decisions

from pathlib import Path

import pytest

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import (
    AuditableOverlapTextCleaner,
    JapaneseMorpheme,
    SudachiMorphologicalAnalyzer,
)
from jp_learning_platform.workflow import OverlapTextCleanupRequest


def _segment(
    position: int,
    words: tuple[Word, ...],
) -> Segment:
    sentence = Sentence(
        text="".join(word.text for word in words),
        time_range=TimeRange(
            words[0].time_range.start_seconds,
            words[-1].time_range.end_seconds,
        ),
        words=words,
    )
    return Segment(
        position=position,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, time_range=TimeRange(start, end))


def _request(*segments: Segment) -> OverlapTextCleanupRequest:
    return OverlapTextCleanupRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=segments,
    )


class _BoundaryAnalyzer:
    def __init__(self, surfaces: tuple[tuple[str, str], ...]) -> None:
        self._surfaces = surfaces

    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        return tuple(
            JapaneseMorpheme(surface, (part_of_speech, "*", "*"))
            for surface, part_of_speech in self._surfaces
        )


def test_cleaner_removes_exact_leading_words_with_time_overlap_and_audit() -> None:
    first = _segment(0, (_word("授業を", 0.0, 1.0), _word("休んだ", 1.0, 2.0)))
    second = _segment(
        1,
        (_word("休", 1.7, 1.8), _word("んだ", 1.8, 1.95), _word("とき", 1.95, 2.4)),
    )

    result = AuditableOverlapTextCleaner().clean(_request(first, second))

    assert result.segments[1].text == "とき"
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.original_text == "休んだとき"
    assert decision.deleted_text == "休んだ"
    assert (decision.deletion_start, decision.deletion_end) == (0, 3)
    assert decision.deleted_time_range == TimeRange(1.7, 1.95)
    assert decision.time_overlap_seconds == 0.25
    assert decision.boundary_gap_seconds == 0.0
    assert decision.reason == "exact_boundary_text_with_aligned_time_overlap"


def test_cleaner_preserves_same_text_without_time_overlap() -> None:
    first = _segment(0, (_word("授業を休んだ", 0.0, 2.0),))
    second = _segment(1, (_word("休んだ", 2.2, 2.6), _word("とき", 2.6, 2.8)))

    result = AuditableOverlapTextCleaner().clean(_request(first, second))

    assert result.segments == (first, second)
    assert not result.decisions


def test_cleaner_removes_exact_contiguous_words_without_alignment_anomaly() -> None:
    first = _segment(0, (_word("どのように宿題", 0.0, 2.0),))
    second = _segment(
        1,
        (_word("宿", 2.04, 2.12), _word("題", 2.12, 2.2), _word("を確認", 2.2, 2.8)),
    )

    result = AuditableOverlapTextCleaner().clean(_request(first, second))

    assert result.segments[1].text == "を確認"
    decision = result.decisions[0]
    assert decision.deleted_text == "宿題"
    assert decision.boundary_gap_seconds == pytest.approx(0.04)
    assert decision.reason == "exact_boundary_words_with_contiguous_timing"


def test_cleaner_transfers_leading_punctuation_after_deleted_duplicate() -> None:
    first = _segment(0, (_word("販売はどう", 0.0, 2.0),))
    second = _segment(
        1,
        (
            _word("ど", 2.02, 2.2),
            _word("う", 2.2, 2.4),
            _word("?", 2.4, 2.5),
            _word("写真", 2.5, 2.8),
        ),
    )

    result = AuditableOverlapTextCleaner().clean(_request(first, second))

    assert result.segments[0].text == "販売はどう?"
    assert tuple(
        word.text for word in result.segments[0].sentences[0].words
    ) == ("販売はどう", "?")
    assert result.segments[0].sentences[0].is_question
    assert result.segments[1].text == "写真"
    decision = result.decisions[0]
    assert decision.deleted_text == "どう"
    assert decision.transferred_punctuation_text == "?"
    assert decision.transferred_word_indexes == (2,)
    assert decision.transferred_time_range == TimeRange(2.4, 2.5)


def test_cleaner_preserves_punctuation_when_transfer_would_empty_segment() -> None:
    first = _segment(0, (_word("販売はどう", 0.0, 2.0),))
    second = _segment(
        1,
        (_word("どう", 2.0, 2.4), _word("?", 2.4, 2.5)),
    )

    result = AuditableOverlapTextCleaner().clean(_request(first, second))

    assert result.segments == (first, second)
    assert not result.decisions


def test_cleaner_removes_contiguous_duplicate_with_anomalous_alignment() -> None:
    first = _segment(0, (_word("いつでもいいです", 65.346, 66.18),))
    second = _segment(
        1,
        (
            _word("で", 66.18, 66.26),
            _word("す", 66.26, 73.829),
            _word("問題1", 73.829, 75.812),
        ),
    )

    result = AuditableOverlapTextCleaner().clean(_request(first, second))

    assert result.segments[1].text == "問題1"
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.deleted_text == "です"
    assert decision.time_overlap_seconds == 0.0
    assert decision.boundary_gap_seconds == 0.0
    assert decision.reason == (
        "exact_boundary_text_with_contiguous_anomalous_alignment"
    )


def test_cleaner_never_deletes_an_entire_segment() -> None:
    first = _segment(0, (_word("休んだ", 0.0, 2.0),))
    second = _segment(1, (_word("休んだ", 1.8, 2.0),))

    result = AuditableOverlapTextCleaner().clean(_request(first, second))

    assert result.segments == (first, second)
    assert not result.decisions


def test_cleaner_removes_single_character_echo_with_boundary_morphology() -> None:
    first = _segment(0, (_word("感想を話して", 0.0, 2.0),))
    second = _segment(
        1,
        (_word("て", 2.04, 2.12), _word("います", 2.12, 2.7)),
    )
    analyzer = _BoundaryAnalyzer(
        (
            ("感想", "名詞"),
            ("を", "助詞"),
            ("話し", "動詞"),
            ("て", "助詞"),
            ("て", "助詞"),
            ("い", "動詞"),
            ("ます", "助動詞"),
        )
    )

    result = AuditableOverlapTextCleaner(
        morphological_analyzer=analyzer
    ).clean(_request(first, second))

    assert result.segments[1].text == "います"
    decision = result.decisions[0]
    assert decision.deleted_text == "て"
    assert decision.reason == (
        "single_character_boundary_echo_with_morphological_evidence"
    )
    assert decision.evidence == (
        "exact_single_aligned_word_boundary",
        "contiguous_timing",
        "matching_function_morphemes_at_boundary",
        "right_character_is_independent_morpheme",
    )


def test_cleaner_preserves_character_that_begins_a_longer_morpheme() -> None:
    first = _segment(0, (_word("増えますね", 0.0, 2.0),))
    second = _segment(
        1,
        (_word("ね", 2.04, 2.12), _word("えポイント", 2.12, 2.7)),
    )
    analyzer = _BoundaryAnalyzer(
        (
            ("増え", "動詞"),
            ("ます", "助動詞"),
            ("ね", "助詞"),
            ("ねえ", "感動詞"),
            ("ポイント", "名詞"),
        )
    )

    result = AuditableOverlapTextCleaner(
        morphological_analyzer=analyzer
    ).clean(_request(first, second))

    assert result.segments == (first, second)
    assert not result.decisions


def test_cleaner_removes_single_character_when_deletion_repairs_morphology() -> None:
    first = _segment(0, (_word("通路側ですね", 0.0, 2.0),))
    second = _segment(
        1,
        (_word("ね", 2.04, 2.2), _word("うーん", 2.2, 2.5), _word("仕方ない", 2.5, 3.0)),
    )

    result = AuditableOverlapTextCleaner(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).clean(_request(first, second))

    assert result.segments[1].text == "うーん仕方ない"
    assert result.decisions[0].reason == (
        "single_character_boundary_echo_with_repaired_morphology"
    )


def test_cleaner_ignores_left_terminal_punctuation_for_single_character_echo() -> None:
    first = _segment(0, (_word("通路側ですね。", 0.0, 2.0),))
    second = _segment(
        1,
        (_word("ね", 2.04, 2.2), _word("うーん", 2.2, 2.5), _word("仕方ない", 2.5, 3.0)),
    )

    result = AuditableOverlapTextCleaner(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).clean(_request(first, second))

    assert result.segments[0].text == "通路側ですね。"
    assert result.segments[1].text == "うーん仕方ない"
    assert result.decisions[0].original_text == "ねうーん仕方ない"
    assert result.decisions[0].deletion_start == 0
    assert result.decisions[0].deleted_text == "ね"


def test_cleaner_preserves_single_character_repeat_without_morphology() -> None:
    first = _segment(0, (_word("感想を話して", 0.0, 2.0),))
    second = _segment(
        1,
        (_word("て", 2.04, 2.12), _word("います", 2.12, 2.7)),
    )

    result = AuditableOverlapTextCleaner().clean(_request(first, second))

    assert result.segments == (first, second)
    assert not result.decisions


def test_cleaner_removes_single_character_echo_with_wider_high_evidence_window() -> None:
    first = _segment(
        0,
        (_word("思う", 0.0, 0.8), _word("ん", 0.8, 1.0)),
    )
    second = _segment(
        1,
        (
            _word("ん", 1.16, 1.24),
            _word("で", 1.24, 1.34),
            _word("す", 1.34, 1.44),
            _word("品物", 1.44, 2.0),
        ),
    )
    analyzer = _BoundaryAnalyzer(
        (
            ("思う", "動詞"),
            ("ん", "助動詞"),
            ("ん", "助詞"),
            ("です", "助動詞"),
            ("品物", "名詞"),
        )
    )

    result = AuditableOverlapTextCleaner(
        morphological_analyzer=analyzer
    ).clean(_request(first, second))

    assert result.segments[0].text == "思うん"
    assert result.segments[1].text == "です品物"
    assert result.decisions[0].deleted_text == "ん"
    assert result.decisions[0].boundary_gap_seconds == pytest.approx(0.16)

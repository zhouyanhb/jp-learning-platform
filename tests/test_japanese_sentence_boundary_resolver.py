from __future__ import annotations

from pathlib import Path

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import JapaneseSentenceBoundaryResolver
from jp_learning_platform.workflow import SentenceBoundaryResolutionRequest


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, time_range=TimeRange(start, end))


def _segment(words: tuple[Word, ...], text: str | None = None) -> Segment:
    sentence_text = text or "".join(word.text for word in words)
    sentence = Sentence(
        text=sentence_text,
        time_range=TimeRange(words[0].time_range.start_seconds, words[-1].time_range.end_seconds),
        words=words,
    )
    return Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )


def _request(segment: Segment) -> SentenceBoundaryResolutionRequest:
    return SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(segment,),
    )


def _sentence_segment(
    position: int,
    text: str,
    start: float,
    end: float,
    *,
    speaker_id: str | None = None,
) -> Segment:
    word = Word(
        text=text,
        time_range=TimeRange(start, end),
        speaker_id=speaker_id,
    )
    sentence = Sentence(
        text=text,
        time_range=word.time_range,
        words=(word,),
        speaker_id=speaker_id,
    )
    return Segment(
        position=position,
        text=text,
        time_range=sentence.time_range,
        sentences=(sentence,),
        speaker_id=speaker_id,
    )


def test_sentence_boundary_resolver_splits_pause_after_sentence_final_expression() -> None:
    words = (
        _word("これ", 4.2, 4.46),
        _word("から", 4.46, 5.1),
        _word("音", 5.1, 5.4),
        _word("を", 5.4, 5.58),
        _word("聞い", 5.58, 5.81),
        _word("て", 5.81, 5.92),
        _word("ください", 5.92, 6.16),
        _word("音", 6.81, 7.67),
        _word("が", 7.93, 8.09),
        _word("よく", 8.09, 8.45),
        _word("聞こえ", 8.45, 8.75),
        _word("ない", 8.75, 8.99),
        _word("とき", 8.99, 9.07),
        _word("は", 9.07, 9.23),
        _word("手", 9.95, 10.13),
        _word("を", 10.13, 10.23),
        _word("挙げ", 10.23, 10.47),
        _word("て", 10.47, 10.59),
        _word("ください", 10.59, 10.79),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(
        _request(
            _segment(
                words,
                text="これから音を聞いてください 音がよく聞こえないときは手を挙げてください",
            )
        )
    )

    sentences = result.segments[0].sentences
    assert tuple(sentence.text for sentence in sentences) == (
        "これから音を聞いてください",
        "音がよく聞こえないときは手を挙げてください",
    )
    assert len(result.decisions) == 1
    assert result.decisions[0].reason == "pause_after_sentence_final"


def test_sentence_boundary_resolver_splits_repeated_sentence_after_mashou() -> None:
    words = (
        _word("天気", 13.65, 14.33),
        _word("が", 14.33, 14.45),
        _word("いい", 14.45, 14.75),
        _word("から", 14.75, 15.2),
        _word("散歩", 15.2, 15.62),
        _word("し", 15.62, 15.74),
        _word("ましょう", 15.74, 16.02),
        _word("天気", 16.67, 17.35),
        _word("が", 17.35, 17.47),
        _word("いい", 17.47, 17.77),
        _word("から", 17.77, 18.22),
        _word("散歩", 18.22, 18.65),
        _word("し", 18.65, 18.77),
        _word("ましょう", 18.77, 19.49),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "天気がいいから散歩しましょう",
        "天気がいいから散歩しましょう",
    )


def test_sentence_boundary_resolver_keeps_conditional_clause_with_following_main_clause() -> None:
    words = (
        _word("音", 6.81, 7.67),
        _word("が", 7.93, 8.09),
        _word("よく", 8.09, 8.45),
        _word("聞こえ", 8.45, 8.75),
        _word("ない", 8.75, 8.99),
        _word("とき", 8.99, 9.07),
        _word("は", 9.07, 9.23),
        _word("手", 9.95, 10.13),
        _word("を", 10.13, 10.23),
        _word("挙げ", 10.23, 10.47),
        _word("て", 10.47, 10.59),
        _word("ください", 10.59, 10.79),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "音がよく聞こえないときは手を挙げてください",
    )
    assert result.decisions == ()


def test_sentence_boundary_resolver_keeps_connection_expression_together() -> None:
    words = (
        _word("それ", 20.0, 20.2),
        _word("から", 20.9, 21.1),
        _word("話", 21.1, 21.4),
        _word("を", 21.4, 21.52),
        _word("聞い", 21.52, 21.8),
        _word("て", 21.8, 22.0),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "それから話を聞いて",
    )
    assert result.decisions == ()


def test_sentence_boundary_resolver_splits_terminal_punctuation_without_pause() -> None:
    words = (
        _word("聞いてください。", 0.0, 1.0),
        _word("音", 1.0, 1.2),
        _word("を", 1.2, 1.3),
        _word("聞いてください", 1.3, 2.0),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "聞いてください。",
        "音を聞いてください",
    )


def test_sentence_boundary_resolver_splits_real_alignment_just_below_pause_threshold() -> None:
    words = (
        _word("天気がいいから散歩しましょう", 13.65, 16.19),
        _word("天気がいいから散歩しましょう", 16.67, 19.49),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "天気がいいから散歩しましょう",
        "天気がいいから散歩しましょう",
    )


def test_sentence_boundary_resolver_splits_final_expression_before_new_clause_without_pause() -> None:
    words = (
        _word("問題がよく見えないときも手を挙げてください", 60.49, 63.99),
        _word("いつでもいいです", 63.99, 66.18),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "問題がよく見えないときも手を挙げてください",
        "いつでもいいです",
    )


def test_sentence_boundary_resolver_reindexes_after_cross_segment_overlap_merge() -> None:
    first = _segment(
        (
            _word("学生は授業を", 104.62, 107.5),
            _word("休んだ", 107.5, 107.94),
        )
    )
    second_sentence = Sentence(
        text="休んだときどのように確認しますか",
        time_range=TimeRange(107.94, 111.08),
        words=(
            _word("休", 107.94, 107.94),
            _word("んだ", 107.94, 107.94),
            _word("ときどのように確認しますか", 107.94, 111.08),
        ),
    )
    second = Segment(
        position=1,
        text=second_sentence.text,
        time_range=second_sentence.time_range,
        sentences=(second_sentence,),
    )
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(first, second),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(request)

    assert tuple(segment.position for segment in result.segments) == (0,)
    assert result.segments[0].text == "学生は授業を休んだときどのように確認しますか"


def test_sentence_boundary_resolver_keeps_final_particle_with_predicate() -> None:
    words = (
        _word("とき、どのように宿題を確認します", 107.94, 110.72),
        _word("か?", 110.72, 111.08),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "とき、どのように宿題を確認しますか?",
    )


def test_sentence_boundary_resolver_adds_comma_to_connective_te_at_segment_boundary() -> None:
    first = _segment((_word("それから話を聞いて", 79.14, 82.35),))
    second_sentence = Sentence(
        text="問題用紙の1から4の中から最も良いものを選んでください",
        time_range=TimeRange(82.35, 89.07),
        words=(
            _word("問題用紙の1から4の中から最も良いものを選んでください", 82.35, 89.07),
        ),
    )
    second = Segment(
        position=1,
        text=second_sentence.text,
        time_range=second_sentence.time_range,
        sentences=(second_sentence,),
    )
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(first, second),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(request)

    assert len(result.segments) == 2
    assert result.segments[0].sentences[0].text == "それから話を聞いて、"
    assert result.segments[0].time_range == TimeRange(79.14, 82.35)


def test_sentence_boundary_resolver_merges_contiguous_dependent_continuation() -> None:
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "学生は授業を休んだ", 104.62, 107.94),
            _sentence_segment(
                1,
                "とき、どのように宿題を確認しますか?",
                107.94,
                111.08,
            ),
        ),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(request)

    assert len(result.segments) == 1
    assert result.segments[0].position == 0
    assert result.segments[0].sentences[0].text == (
        "学生は授業を休んだとき、"
        "どのように宿題を確認しますか?"
    )
    assert result.segments[0].time_range == TimeRange(104.62, 111.08)


def test_sentence_boundary_resolver_keeps_guarded_dependent_segments_separate() -> None:
    guarded_pairs = (
        (
            _sentence_segment(0, "学生は授業を休んだ。", 0.0, 1.0),
            _sentence_segment(1, "ときには休むことも必要です", 1.0, 2.0),
        ),
        (
            _sentence_segment(0, "学生は授業を休んだ", 0.0, 1.0, speaker_id="a"),
            _sentence_segment(1, "ときどうしますか", 1.0, 2.0, speaker_id="b"),
        ),
        (
            _sentence_segment(0, "学生は授業を休んだ", 0.0, 1.0),
            _sentence_segment(1, "ときどうしますか", 2.0, 3.0),
        ),
    )

    for segments in guarded_pairs:
        request = SentenceBoundaryResolutionRequest(
            source_path=Path("input.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=segments,
        )
        result = JapaneseSentenceBoundaryResolver().resolve(request)
        assert len(result.segments) == 2

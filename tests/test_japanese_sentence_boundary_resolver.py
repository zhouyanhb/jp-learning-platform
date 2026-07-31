from __future__ import annotations

from pathlib import Path

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import (
    JapaneseMorpheme,
    JapaneseSentenceBoundaryResolver,
    SudachiMorphologicalAnalyzer,
)
from jp_learning_platform.infrastructure.japanese_sentence_boundary_resolver import (
    _has_morpheme_boundary_at,
    _is_complete_clause,
    _is_valid_prefix_partition,
)
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
) -> Segment:
    word = Word(
        text=text,
        time_range=TimeRange(start, end),
    )
    sentence = Sentence(
        text=text,
        time_range=word.time_range,
        words=(word,),
    )
    return Segment(
        position=position,
        text=text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )


class _MorphologicalAnalyzer:
    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        if text.endswith("そうしよう"):
            return (
                JapaneseMorpheme("そう", ("副詞", "*", "*")),
                JapaneseMorpheme(
                    "しよう",
                    ("動詞", "非自立可能", "*"),
                    dictionary_form="する",
                    conjugation_form="意志推量形",
                ),
            )
        if text.endswith("そう"):
            return (JapaneseMorpheme("そう", ("副詞", "*", "*")),)
        if text == "しよう":
            return (
                JapaneseMorpheme(
                    "しよう",
                    ("動詞", "非自立可能", "*"),
                    dictionary_form="する",
                    conjugation_form="意志推量形",
                ),
            )
        return (JapaneseMorpheme(text, ("名詞", "普通名詞", "一般")),)


class _CaseContinuationAnalyzer:
    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        if text == "宿題":
            return (JapaneseMorpheme("宿題", ("名詞", "普通名詞", "一般")),)
        if text == "を確認しますか":
            return (
                JapaneseMorpheme("を", ("助詞", "格助詞", "*")),
                JapaneseMorpheme("確認します", ("動詞", "一般", "*")),
                JapaneseMorpheme("か", ("助詞", "終助詞", "*")),
            )
        if text == "宿題を確認しますか":
            return (
                JapaneseMorpheme("宿題", ("名詞", "普通名詞", "一般")),
                JapaneseMorpheme("を", ("助詞", "格助詞", "*")),
                JapaneseMorpheme("確認します", ("動詞", "一般", "*")),
                JapaneseMorpheme("か", ("助詞", "終助詞", "*")),
            )
        return (JapaneseMorpheme(text, ("名詞", "普通名詞", "一般")),)


class _AuxiliaryContinuationAnalyzer:
    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        if text == "思うん":
            return (
                JapaneseMorpheme("思う", ("動詞", "一般", "*")),
                JapaneseMorpheme(
                    "ん",
                    ("助動詞", "*", "*"),
                    conjugation_form="終止形-撥音便",
                ),
            )
        if text == "思うんです":
            return (
                JapaneseMorpheme("思う", ("動詞", "一般", "*")),
                JapaneseMorpheme("ん", ("助動詞", "*", "*")),
                JapaneseMorpheme("です", ("助動詞", "*", "*")),
            )
        return (JapaneseMorpheme(text, ("助動詞", "*", "*")),)


class _IndependentInterjectionAnalyzer:
    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        if text.startswith("そう"):
            return (
                JapaneseMorpheme("そう", ("副詞", "*", "*")),
                JapaneseMorpheme("です", ("助動詞", "*", "*")),
                JapaneseMorpheme("ね", ("助詞", "終助詞", "*")),
            )
        if text.startswith("ねえ"):
            return (
                JapaneseMorpheme("ねえ", ("感動詞", "一般", "*")),
                JapaneseMorpheme("次", ("名詞", "普通名詞", "一般")),
            )
        if text.endswith("ねねえ"):
            return (
                JapaneseMorpheme("終わります", ("動詞", "一般", "*")),
                JapaneseMorpheme("ね", ("助詞", "終助詞", "*")),
                JapaneseMorpheme("ね", ("助詞", "終助詞", "*")),
                JapaneseMorpheme("え", ("感動詞", "一般", "*")),
            )
        return (JapaneseMorpheme(text, ("動詞", "一般", "*")),)
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


def test_sentence_boundary_resolver_marks_existing_question_punctuation() -> None:
    words = (
        _word("そうですか?", 0.0, 1.0),
        _word("」", 1.0, 1.1),
        _word("次です", 1.1, 2.0),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert result.segments[0].sentences[0].text == "そうですか?"
    assert result.segments[0].sentences[0].is_question
    assert result.segments[0].sentences[0].words[0].text == "そうですか?"


def test_sentence_boundary_resolver_keeps_pause_below_threshold() -> None:
    words = (
        _word("天気がいいから散歩しましょう", 13.65, 16.19),
        _word("天気がいいから散歩しましょう", 16.67, 19.49),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "天気がいいから散歩しましょう天気がいいから散歩しましょう",
    )
    assert not result.decisions


def test_sentence_boundary_resolver_does_not_split_final_expression_without_pause() -> None:
    words = (
        _word("問題がよく見えないときも手を挙げてください", 60.49, 63.99),
        _word("いつでもいいです", 63.99, 66.18),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "問題がよく見えないときも手を挙げてくださいいつでもいいです",
    )
    assert not result.decisions


def test_sentence_boundary_resolver_does_not_delete_cross_segment_overlap() -> None:
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

    assert tuple(segment.position for segment in result.segments) == (0, 1)
    assert tuple(segment.text for segment in result.segments) == (
        "学生は授業を休んだ",
        "休んだときどのように確認しますか",
    )


def test_sentence_boundary_resolver_preserves_all_source_characters() -> None:
    words = (
        _word("いつでもいいです", 0.0, 1.0),
        _word("質問22人は", 1.0, 2.0),
        _word("大丈夫だったただテキスト", 2.0, 3.0),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(_request(_segment(words)))

    assert result.segments[0].text == (
        "いつでもいいです質問22人は大丈夫だったただテキスト"
    )


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
        time_range=TimeRange(83.15, 89.07),
        words=(
            _word("問題用紙の1から4の中から最も良いものを選んでください", 83.15, 89.07),
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

    assert len(result.segments) == 1
    assert result.segments[0].sentences[0].text == (
        "それから話を聞いて、"
        "問題用紙の1から4の中から最も良いものを選んでください"
    )
    assert result.segments[0].time_range == TimeRange(79.14, 89.07)


def test_sentence_boundary_resolver_splits_extended_question_particle() -> None:
    words = (
        _word("男の社員はこの後まず何をします", 202.131, 202.8),
        _word("か", 202.8, 208.8),
        _word("あの", 208.8, 209.0),
        _word("ちょっと頼みたいことがあるんだけど", 209.0, 209.76),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(
        _request(_segment(words))
    )

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "男の社員はこの後まず何をしますか",
        "あのちょっと頼みたいことがあるんだけど",
    )
    assert result.segments[0].sentences[0].is_question
    assert result.segments[0].sentences[0].words[-1].text == "か"
    assert result.segments[0].sentences[0].time_range == TimeRange(202.131, 203.3)
    assert result.decisions[0].reason == "sentence_final_question_particle"


def test_sentence_boundary_resolver_splits_polite_question_with_held_pause() -> None:
    words = (
        _word("男の社員はこの後まず何をし", 0.0, 0.8),
        _word("ます", 0.8, 1.1),
        _word("か", 1.1, 2.3),
        _word("あの", 2.3, 2.5),
        _word("お願いがあります", 2.5, 3.2),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(
        _request(_segment(words))
    )

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "男の社員はこの後まず何をしますか",
        "あのお願いがあります",
    )
    assert result.segments[0].sentences[0].is_question
    assert result.decisions[0].reason == "sentence_final_question_particle"


def test_sentence_boundary_resolver_requires_timing_evidence_for_question_particle() -> None:
    continuous_constructions = (
        ("誰", "か", "伝えておいてください"),
        ("何", "か", "野菜を作りたい"),
        ("家具", "と", "か", "電気製品"),
        ("買い取ってくれない", "か", "も", "しれない"),
        ("就職できる", "か", "どう", "か", "不安です"),
        ("自分の考えな", "の", "か", "資料の引用な", "の", "か", "が曖昧です"),
    )

    for texts in continuous_constructions:
        words = tuple(
            _word(text, index * 0.2, (index + 1) * 0.2)
            for index, text in enumerate(texts)
        )
        result = JapaneseSentenceBoundaryResolver().resolve(
            _request(_segment(words))
        )

        assert tuple(
            sentence.text for sentence in result.segments[0].sentences
        ) == ("".join(texts),)
        assert not result.decisions


def test_sentence_boundary_resolver_uses_effective_pause_not_extended_word_reason() -> None:
    words = (
        _word("次回", 0.0, 0.4),
        _word("の", 0.4, 0.5),
        _word("応募", 0.5, 0.9),
        _word("方法", 0.9, 13.9),
        _word("問題", 13.9, 14.4),
        _word("4", 14.4, 14.6),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(
        _request(_segment(words))
    )

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "次回の応募方法",
        "問題4",
    )
    assert result.segments[0].sentences[0].time_range == TimeRange(0.0, 1.9)
    assert result.decisions[0].reason == "strong_pause"


def test_sentence_boundary_resolver_reattaches_morphological_prefix_before_pause() -> None:
    first = _sentence_segment(0, "じゃあそう", 640.0, 646.499)
    second_words = (
        _word("し", 646.9, 646.92),
        _word("よ", 646.92, 646.94),
        _word("う", 646.94, 653.425),
        _word("男", 653.425, 653.545),
        _word("の人はどの席を選びますか", 653.545, 657.0),
    )
    second_sentence = Sentence(
        text="しよう男の人はどの席を選びますか",
        time_range=TimeRange(646.9, 657.0),
        words=second_words,
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

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(segment.text for segment in result.segments) == (
        "じゃあそうしよう",
        "男の人はどの席を選びますか",
    )


def test_sentence_boundary_resolver_joins_tight_connective_without_comma() -> None:
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "話して", 0.0, 0.5),
            _sentence_segment(1, "います", 0.5, 1.0),
        ),
    )

    result = JapaneseSentenceBoundaryResolver().resolve(request)

    assert len(result.segments) == 1
    assert result.segments[0].sentences[0].text == "話しています"


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


def test_sentence_boundary_resolver_uses_grammar_beyond_close_gap_threshold() -> None:
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "宿題", 0.0, 1.0),
            _sentence_segment(1, "を確認しますか", 1.203, 2.5),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=_CaseContinuationAnalyzer()
    ).resolve(request)

    assert len(result.segments) == 1
    assert result.segments[0].sentences[0].text == "宿題を確認しますか"


def test_sentence_boundary_resolver_scores_cross_segment_auxiliary_chain() -> None:
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "思うん", 0.0, 1.0),
            _sentence_segment(1, "です", 1.241, 1.8),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=_AuxiliaryContinuationAnalyzer()
    ).resolve(request)

    assert len(result.segments) == 1
    assert result.segments[0].sentences[0].text == "思うんです"


def test_sentence_boundary_resolver_moves_only_minimal_dependent_prefix() -> None:
    right_words = (
        _word("です", 1.05, 1.25),
        _word("品物", 1.25, 1.55),
        _word("を", 1.55, 1.65),
        _word("変えます", 1.65, 2.2),
    )
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "思うん", 0.0, 1.0),
            _segment(right_words),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(segment.text for segment in result.segments) == (
        "思うんです",
        "品物を変えます",
    )


def test_minimal_prefix_migration_preserves_internal_asr_word_boundary() -> None:
    right_words = (
        _word("です", 1.05, 1.2),
        _word("品物", 1.2, 1.4),
        _word("を", 1.4, 1.5),
        _word("変えて", 1.5, 1.7),
        _word("みて", 1.7, 1.9),
        _word("は", 1.9, 2.0),
        _word("どう", 2.0, 2.15),
        _word("でしょう", 2.15, 2.3),
        _word("か", 2.3, 2.4),
        _word("そう", 2.4, 2.55),
        _word("です", 2.55, 2.7),
    )
    right_sentence = Sentence(
        text="です 品物を変えてみてはどうでしょうか そうです",
        time_range=TimeRange(1.05, 2.7),
        words=right_words,
        asr_boundary_word_indexes=(1, 9),
    )
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "思うん", 0.0, 1.0),
            Segment(1, right_sentence.text, right_sentence.time_range, (right_sentence,)),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(
        sentence.text
        for segment in result.segments
        for sentence in segment.sentences
    ) == ("思うんです", "品物を変えてみてはどうでしょうか", "そうです")


def test_sentence_boundary_resolver_scores_question_answer_relative_pause() -> None:
    words = (
        _word("どう", 0.0, 0.2),
        _word("です", 0.2, 0.4),
        _word("か", 0.4, 0.5),
        _word("そう", 0.8, 1.0),
        _word("です", 1.0, 1.2),
    )
    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "どうですか",
        "そうです",
    )
    assert result.segments[0].sentences[0].is_question


def test_sentence_boundary_resolver_joins_adverb_to_independent_predicate() -> None:
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "じゃあそう", 0.0, 1.0),
            _sentence_segment(1, "しよう", 1.15, 1.6),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=_MorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(segment.text for segment in result.segments) == ("じゃあそうしよう",)


def test_minimal_prefix_keeps_single_character_inflection_extension() -> None:
    right_words = (
        _word("し", 1.15, 1.17),
        _word("よ", 1.17, 1.19),
        _word("う", 1.19, 4.0),
        _word("次", 4.0, 4.3),
        _word("です", 4.3, 4.6),
    )
    right = _segment(right_words)
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "じゃあそう", 0.0, 1.0),
            Segment(1, right.text, right.time_range, right.sentences),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(
        sentence.text
        for segment in result.segments
        for sentence in segment.sentences
    ) == ("じゃあそうしよう", "次です")


def test_comma_auxiliary_symbol_does_not_complete_clause() -> None:
    morphemes = SudachiMorphologicalAnalyzer().analyze("確認して、")

    assert morphemes[-1].part_of_speech[:2] == ("補助記号", "読点")
    assert not _is_complete_clause(morphemes)


def test_period_auxiliary_symbol_completes_clause() -> None:
    morphemes = SudachiMorphologicalAnalyzer().analyze("確認します。")

    assert morphemes[-1].part_of_speech[:2] == ("補助記号", "句点")
    assert _is_complete_clause(morphemes)


def test_shape_word_can_start_independent_answer_without_asr_boundary() -> None:
    words = (
        _word("どうですか", 0.0, 0.5),
        _word("静かです", 0.7, 1.1),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "どうですか",
        "静かです",
    )


def test_connective_form_before_independent_response_changes_turn() -> None:
    words = (
        _word("確認し", 0.0, 0.4),
        _word("はい", 0.85, 1.1),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "確認し",
        "はい",
    )
    assert result.decisions[0].reason == "connective_response_transition"


def test_asr_boundary_recognizes_generic_numbered_restart() -> None:
    words = (
        _word("以上です", 0.0, 0.5),
        _word("課題", 0.5, 0.7),
        _word("2", 0.7, 0.8),
        _word("次です", 0.8, 1.2),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words, text="以上です 課題2次です")))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "以上です",
        "課題2次です",
    )
    assert result.decisions[0].reason == "asr_structural_restart"


def test_sentence_boundary_resolver_scores_retained_asr_text_boundary() -> None:
    words = (
        _word("どう", 0.0, 0.2),
        _word("です", 0.2, 0.4),
        _word("か", 0.4, 0.5),
        _word("そう", 0.5, 0.7),
        _word("です", 0.7, 0.9),
    )
    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words, text="どうですか そうです")))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "どうですか",
        "そうです",
    )
    assert result.decisions[0].reason == "asr_question_answer_transition"


def test_sentence_boundary_resolver_does_not_reattach_independent_interjection() -> None:
    first = _sentence_segment(0, "終わりますね", 0.0, 1.0)
    right_words = (
        _word("ね", 1.05, 1.1),
        _word("え", 1.1, 1.2),
        _word("次", 1.8, 2.0),
    )
    second = _segment(right_words, text="ねえ次")
    second = Segment(
        position=1,
        text=second.text,
        time_range=second.time_range,
        sentences=second.sentences,
    )
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(first, second),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=_IndependentInterjectionAnalyzer()
    ).resolve(request)

    assert tuple(segment.text for segment in result.segments) == (
        "終わりますね",
        "ねえ次",
    )


def test_sentence_boundary_resolver_keeps_close_question_and_answer_separate() -> None:
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "どうですか", 0.0, 1.0),
            _sentence_segment(1, "そうですね", 1.05, 1.8),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=_IndependentInterjectionAnalyzer()
    ).resolve(request)

    assert tuple(segment.text for segment in result.segments) == (
        "どうですか",
        "そうですね",
    )


def test_connective_merge_preserves_independent_cross_segment_responses() -> None:
    pairs = (
        ("聞かれて", "はい"),
        ("早退するので", "わかりました"),
        ("秘密にして", "どういうことですか"),
        ("ようにして", "なるほど"),
    )
    for left, right in pairs:
        request = SentenceBoundaryResolutionRequest(
            source_path=Path("input.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(
                _sentence_segment(0, left, 0.0, 1.0),
                _sentence_segment(1, right, 1.1, 1.8),
            ),
        )

        result = JapaneseSentenceBoundaryResolver(
            morphological_analyzer=SudachiMorphologicalAnalyzer()
        ).resolve(request)

        assert tuple(segment.text for segment in result.segments) == (left, right)


def test_connective_scoring_keeps_true_cross_segment_continuations() -> None:
    pairs = (
        ("確認して", "ください", "確認してください"),
        ("話し", "ています", "話しています"),
        ("そう", "しよう", "そうしよう"),
    )
    for left, right, expected in pairs:
        request = SentenceBoundaryResolutionRequest(
            source_path=Path("input.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(
                _sentence_segment(0, left, 0.0, 1.0),
                _sentence_segment(1, right, 1.1, 1.8),
            ),
        )

        result = JapaneseSentenceBoundaryResolver(
            morphological_analyzer=SudachiMorphologicalAnalyzer()
        ).resolve(request)

        assert tuple(segment.text for segment in result.segments) == (expected,)


def test_combined_morphology_vetoes_false_intra_segment_boundaries() -> None:
    pairs = (
        ("旅行に行った", "時も旅行先で楽しんでいます"),
        ("映画を見た", "後で感想を話しています"),
        ("経費がかかる", "からこの際やめてもいいと思います"),
        ("毎日1時間ほど歩く", "ことにしてみたんです"),
        ("トレーニングをしても思う", "ように筋肉がつかない"),
        ("目に浮かぶように鮮やかに", "描かれた作品です"),
    )
    for left, right in pairs:
        words = (
            _word(left, 0.0, 1.0),
            _word(right, 1.6, 2.6),
        )

        result = JapaneseSentenceBoundaryResolver(
            morphological_analyzer=SudachiMorphologicalAnalyzer()
        ).resolve(_request(_segment(words)))

        assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
            left + right,
        )


def test_cross_segment_adverbial_tail_looks_ahead_to_right_predicate() -> None:
    right_words = (
        _word("話をする", 1.12, 1.4),
        _word("ようになって", 1.4, 1.65),
        _word("それが", 1.65, 1.8),
        _word("楽しいです", 1.8, 2.1),
    )
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "体を動かしながらいろいろ", 0.0, 1.0),
            _segment(right_words),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(segment.text for segment in result.segments) == (
        "体を動かしながらいろいろ話をするようになってそれが楽しいです",
    )


def test_cross_segment_adverbial_form_joins_nonfinite_predicate() -> None:
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(
            _sentence_segment(0, "目に浮かぶように鮮やかに", 0.0, 1.0),
            _sentence_segment(1, "描かれ高く評価されました", 1.12, 2.0),
        ),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(segment.text for segment in result.segments) == (
        "目に浮かぶように鮮やかに描かれ高く評価されました",
    )


def test_prefix_partition_rejects_functional_and_adnominal_fragments() -> None:
    analyzer = SudachiMorphologicalAnalyzer()
    invalid_partitions = (("あ、奨学金はまだなんです", "けど"),)

    for attached_text, remainder_text in invalid_partitions:
        assert not _is_valid_prefix_partition(
            analyzer.analyze(attached_text),
            analyzer.analyze(remainder_text),
        )


def test_prefix_partition_accepts_complete_independent_remainder() -> None:
    analyzer = SudachiMorphologicalAnalyzer()

    assert _is_valid_prefix_partition(
        analyzer.analyze("思うんです"),
        analyzer.analyze("品物を変えます"),
    )


def test_prefix_partition_rejects_cut_inside_morpheme() -> None:
    analyzer = SudachiMorphologicalAnalyzer()
    attached_text = "あ、奨学金はまだなんですけ"
    remainder_text = "ど"

    assert not _is_valid_prefix_partition(
        analyzer.analyze(attached_text),
        analyzer.analyze(remainder_text),
        has_morpheme_boundary=_has_morpheme_boundary_at(
            f"{attached_text}{remainder_text}",
            len(attached_text),
            analyzer,
        ),
    )


def test_local_numbering_region_splits_incrementing_items() -> None:
    words = (
        _word("どれを選びますか", 0.0, 0.8),
        _word("1", 1.0, 1.1),
        _word("資料を読む", 1.1, 2.0),
        _word("2", 2.4, 2.5),
        _word("担当者に聞く", 2.5, 3.4),
        _word("3", 3.8, 3.9),
        _word("後で確認する", 3.9, 4.8),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "どれを選びますか",
        "1資料を読む",
        "2担当者に聞く",
        "3後で確認する",
    )
    assert all(
        decision.reason == "structured_numbering_sequence"
        for decision in result.decisions
    )


def test_local_numbering_region_ignores_ordinary_quantities() -> None:
    words = (
        _word("会議は", 0.0, 0.4),
        _word("1", 0.4, 0.5),
        _word("時間で", 0.5, 1.0),
        _word("資料を", 1.0, 1.4),
        _word("2", 1.4, 1.5),
        _word("部用意して", 1.5, 2.0),
        _word("3", 2.0, 2.1),
        _word("人で確認します", 2.1, 3.0),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "会議は1時間で資料を2部用意して3人で確認します",
    )
    assert not result.decisions


def test_local_numbering_region_requires_multiple_incrementing_items() -> None:
    words = (
        _word("案", 0.0, 0.3),
        _word("1", 0.3, 0.4),
        _word("を検討して案", 0.4, 1.2),
        _word("2", 1.2, 1.3),
        _word("を保留します", 1.3, 2.0),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words)))

    assert not any(
        decision.reason == "structured_numbering_sequence"
        for decision in result.decisions
    )


def test_local_numbering_region_restarts_after_confirmed_sequence() -> None:
    words = (
        _word("1", 0.0, 0.1),
        _word("第一案", 0.1, 0.8),
        _word("2", 1.0, 1.1),
        _word("第二案", 1.1, 1.8),
        _word("3", 2.0, 2.1),
        _word("第三案", 2.1, 2.8),
        _word("1", 3.6, 3.7),
        _word("番", 3.7, 3.8),
        _word("新しい内容です", 3.8, 4.8),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(_request(_segment(words)))

    assert tuple(sentence.text for sentence in result.segments[0].sentences) == (
        "1第一案",
        "2第二案",
        "3第三案",
        "1番新しい内容です",
    )


def test_numbering_sequence_attaches_body_from_next_asr_segment() -> None:
    first = _segment(
        (
            _word("1", 0.0, 0.1),
            _word("第一案", 0.1, 0.8),
            _word("2", 1.0, 1.1),
            _word("第二案", 1.1, 1.8),
            _word("3", 2.0, 2.1),
        )
    )
    second = _sentence_segment(1, "最後の案です", 2.5, 3.5)
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(first, second),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(
        sentence.text
        for segment in result.segments
        for sentence in segment.sentences
    ) == ("1第一案", "2第二案", "3最後の案です")
    assert any(
        decision.reason == "cross_asr_numbering_body"
        for decision in result.decisions
    )


def test_numbering_sequence_moves_minimal_prefix_before_next_number() -> None:
    first = _segment(
        (
            _word("1", 0.0, 0.1),
            _word("開催目的", 0.1, 0.8),
            _word("2", 1.0, 1.1),
            _word("受賞コメント", 1.1, 1.8),
            _word("3", 2.0, 2.1),
            _word("最優秀賞に選ばれた", 2.1, 3.0),
        )
    )
    second = _segment(
        (
            _word("作品", 3.0, 3.3),
            _word("4", 3.8, 3.9),
            _word("次回の応募方法", 3.9, 4.8),
        )
    )
    request = SentenceBoundaryResolutionRequest(
        source_path=Path("input.mp3"),
        working_directory=Path("work"),
        run_id="run-001",
        segments=(first, second),
    )

    result = JapaneseSentenceBoundaryResolver(
        morphological_analyzer=SudachiMorphologicalAnalyzer()
    ).resolve(request)

    assert tuple(
        sentence.text
        for segment in result.segments
        for sentence in segment.sentences
    ) == (
        "1開催目的",
        "2受賞コメント",
        "3最優秀賞に選ばれた作品",
        "4次回の応募方法",
    )
    assert any(
        decision.reason == "cross_asr_numbering_prefix"
        for decision in result.decisions
    )


def test_sentence_boundary_resolver_keeps_guarded_dependent_segments_separate() -> None:
    guarded_pairs = (
        (
            _sentence_segment(0, "学生は授業を休んだ。", 0.0, 1.0),
            _sentence_segment(1, "ときには休むことも必要です", 1.0, 2.0),
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

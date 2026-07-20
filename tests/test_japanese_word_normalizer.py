from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.japanese_words import (
    HeuristicJapaneseTokenizer,
    JapaneseWordTimingNormalizer,
    SudachiJapaneseTokenizer,
)
from jp_learning_platform.workflow import JapaneseWordNormalizationRequest


@dataclass(frozen=True, slots=True)
class StaticTokenizer:
    tokens: tuple[str, ...]

    def tokenize(self, text: str) -> tuple[str, ...]:
        return self.tokens


def _sudachi_tokenizer() -> SudachiJapaneseTokenizer:
    pytest.importorskip("sudachipy")
    pytest.importorskip("sudachidict_core")
    return SudachiJapaneseTokenizer()


def _segment_with_piece_words() -> Segment:
    words = (
        Word(text="天", time_range=TimeRange(13.69, 14.13), confidence=0.93),
        Word(text="気", time_range=TimeRange(14.13, 14.33), confidence=0.99),
        Word(text="が", time_range=TimeRange(14.33, 14.53), confidence=0.99),
        Word(text="いい", time_range=TimeRange(14.53, 14.69), confidence=0.98),
        Word(text="から", time_range=TimeRange(14.69, 14.97), confidence=0.99),
        Word(text="散", time_range=TimeRange(14.97, 15.51), confidence=0.99),
        Word(text="歩", time_range=TimeRange(15.51, 15.69), confidence=0.99),
        Word(text="し", time_range=TimeRange(15.69, 15.79), confidence=0.99),
        Word(text="ましょう", time_range=TimeRange(15.79, 16.19), confidence=0.99),
    )
    sentence = Sentence(
        text="天気がいいから散歩しましょう",
        time_range=TimeRange(13.69, 16.19),
        words=words,
    )
    return Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )


def test_japanese_word_timing_normalizer_merges_piece_words() -> None:
    normalizer = JapaneseWordTimingNormalizer(
        tokenizer=StaticTokenizer(
            tokens=("天気", "が", "いい", "から", "散歩", "しましょう")
        )
    )
    source_path = Path("lesson.mp3")

    result = normalizer.normalize(
        JapaneseWordNormalizationRequest(
            source_path=source_path,
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment_with_piece_words(),),
        )
    )

    words = result.segments[0].sentences[0].words
    assert tuple(word.text for word in words) == (
        "天気",
        "が",
        "いい",
        "から",
        "散歩",
        "しましょう",
    )
    assert words[0].time_range == TimeRange(13.69, 14.33)
    assert words[4].time_range == TimeRange(14.97, 15.69)
    assert round(words[0].confidence or 0.0, 3) == 0.96


def test_japanese_word_timing_normalizer_keeps_words_when_tokens_do_not_match() -> None:
    segment = _segment_with_piece_words()
    normalizer = JapaneseWordTimingNormalizer(
        tokenizer=StaticTokenizer(tokens=("天気", "が"))
    )

    result = normalizer.normalize(
        JapaneseWordNormalizationRequest(
            source_path=Path("lesson.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    assert result.segments == (segment,)


def test_japanese_word_timing_normalizer_maps_repaired_text_to_source_timing() -> None:
    source_words = (
        Word(text="電", time_range=TimeRange(13.69, 14.13), confidence=0.93),
        Word(text="気", time_range=TimeRange(14.13, 14.33), confidence=0.99),
        Word(text="が", time_range=TimeRange(14.33, 14.53), confidence=0.99),
        Word(text="いい", time_range=TimeRange(14.53, 14.69), confidence=0.98),
        Word(text="から", time_range=TimeRange(14.69, 14.97), confidence=0.99),
        Word(text="散", time_range=TimeRange(14.97, 15.51), confidence=0.99),
        Word(text="歩", time_range=TimeRange(15.51, 15.69), confidence=0.99),
        Word(text="し", time_range=TimeRange(15.69, 15.79), confidence=0.99),
        Word(text="ましょう", time_range=TimeRange(15.79, 16.19), confidence=0.99),
    )
    repaired_sentence = Sentence(
        text="天気がいいから散歩しましょう。",
        time_range=TimeRange(13.69, 16.19),
        words=source_words,
    )
    repaired_segment = Segment(
        position=0,
        text=repaired_sentence.text,
        time_range=repaired_sentence.time_range,
        sentences=(repaired_sentence,),
    )
    normalizer = JapaneseWordTimingNormalizer(
        tokenizer=StaticTokenizer(
            tokens=("天気", "が", "いい", "から", "散歩", "しましょう", "。")
        )
    )

    result = normalizer.normalize(
        JapaneseWordNormalizationRequest(
            source_path=Path("lesson.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(repaired_segment,),
        )
    )

    words = result.segments[0].sentences[0].words
    assert tuple(word.text for word in words) == (
        "天気",
        "が",
        "いい",
        "から",
        "散歩",
        "しましょう",
        "。",
    )
    assert words[0].time_range == TimeRange(13.69, 14.33)
    assert words[4].time_range == TimeRange(14.97, 15.69)
    assert words[-1].time_range == TimeRange(16.19, 16.19)


def test_heuristic_japanese_tokenizer_handles_common_piece_boundaries() -> None:
    tokenizer = HeuristicJapaneseTokenizer()

    assert tokenizer.tokenize("天気がいいから散歩しましょう") == (
        "天気",
        "が",
        "いい",
        "から",
        "散歩",
        "しましょう",
    )


def test_sudachi_japanese_tokenizer_merges_learning_verb_units() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("これから音を聞いてください。") == (
        "これ",
        "から",
        "音",
        "を",
        "聞いて",
        "ください",
        "。",
    )
    assert tokenizer.tokenize("手を挙げてください。") == (
        "手",
        "を",
        "挙げて",
        "ください",
        "。",
    )
    assert tokenizer.tokenize("天気がいいから散歩しましょう") == (
        "天気",
        "が",
        "いい",
        "から",
        "散歩しましょう",
    )


def test_sudachi_japanese_tokenizer_merges_sahen_and_auxiliary_chains() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("説明してください") == (
        "説明して",
        "ください",
    )
    assert tokenizer.tokenize("読んでいます") == ("読んでいます",)
    assert tokenizer.tokenize("メモを取っても構いません") == (
        "メモ",
        "を",
        "取っても",
        "構いません",
    )


def test_sudachi_japanese_tokenizer_keeps_kudasai_separate() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("確認しておいてください") == (
        "確認しておいて",
        "ください",
    )


def test_sudachi_japanese_tokenizer_merges_stable_inner_noun_units() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("問題用紙のページ") == (
        "問題",
        "用紙",
        "の",
        "ページ",
    )


def test_sudachi_japanese_tokenizer_merges_generic_number_counters() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("2021年第2回") == ("2021年", "第2回")


def test_sudachi_japanese_tokenizer_merges_compound_particle_units() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("いつでも構いません。") == (
        "いつ",
        "でも",
        "構いません",
        "。",
    )
    assert tokenizer.tokenize("それでは始めます。") == (
        "それ",
        "では",
        "始めます",
        "。",
    )
    assert tokenizer.tokenize("私にはわかりません。") == (
        "私",
        "には",
        "わかりません",
        "。",
    )
    assert tokenizer.tokenize("先生とは話しました。") == (
        "先生",
        "とは",
        "話しました",
        "。",
    )


def test_sudachi_japanese_tokenizer_merges_particle_chain_units() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("明日からは大丈夫です。") == (
        "明日",
        "からは",
        "大丈夫",
        "です",
        "。",
    )
    assert tokenizer.tokenize("ここからも見えます。") == (
        "ここ",
        "からも",
        "見えます",
        "。",
    )
    assert tokenizer.tokenize("今日までは休みです。") == (
        "今日",
        "までは",
        "休み",
        "です",
        "。",
    )
    assert tokenizer.tokenize("これだけでも大丈夫です。") == (
        "これ",
        "だけでも",
        "大丈夫",
        "です",
        "。",
    )
    assert tokenizer.tokenize("一人だけでは難しいです。") == (
        "一人",
        "だけでは",
        "難しい",
        "です",
        "。",
    )


def test_sudachi_japanese_tokenizer_merges_formal_particle_expressions() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("これについて説明します。") == (
        "これ",
        "について",
        "説明します",
        "。",
    )
    assert tokenizer.tokenize("試験によって違います。") == (
        "試験",
        "によって",
        "違います",
        "。",
    )
    assert tokenizer.tokenize("場所によっては違います。") == (
        "場所",
        "によっては",
        "違います",
        "。",
    )
    assert tokenizer.tokenize("友達として参加します。") == (
        "友達",
        "として",
        "参加します",
        "。",
    )


def test_sudachi_japanese_tokenizer_keeps_left_word_before_particle_unit() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("日曜日でも構いません。") == (
        "日曜日",
        "でも",
        "構いません",
        "。",
    )
    assert tokenizer.tokenize("誰にでもできます。") == (
        "誰",
        "にでも",
        "できます",
        "。",
    )
    assert tokenizer.tokenize("いつからでも始められます。") == (
        "いつ",
        "からでも",
        "始められます",
        "。",
    )
    assert tokenizer.tokenize("どこまででも行けます。") == (
        "どこ",
        "まででも",
        "行けます",
        "。",
    )


def test_sudachi_japanese_tokenizer_keeps_verb_temoi_as_verb_chain() -> None:
    tokenizer = _sudachi_tokenizer()

    assert tokenizer.tokenize("行っても構いません。") == (
        "行っても",
        "構いません",
        "。",
    )
    assert tokenizer.tokenize("読んでもわかりません。") == (
        "読んでも",
        "わかりません",
        "。",
    )


def test_japanese_word_timing_normalizer_merges_sudachi_learning_words() -> None:
    normalizer = JapaneseWordTimingNormalizer(tokenizer=_sudachi_tokenizer())
    source_words = (
        Word(text="音", time_range=TimeRange(5.102, 5.402), confidence=1.0),
        Word(text="を", time_range=TimeRange(5.402, 5.582), confidence=1.0),
        Word(text="聞い", time_range=TimeRange(5.582, 5.813), confidence=0.9995),
        Word(text="て", time_range=TimeRange(5.813, 5.923), confidence=0.999),
        Word(text="ください", time_range=TimeRange(5.923, 6.2), confidence=1.0),
    )
    sentence = Sentence(
        text="音を聞いてください。",
        time_range=TimeRange(5.102, 6.2),
        words=source_words,
    )
    segment = Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )

    result = normalizer.normalize(
        JapaneseWordNormalizationRequest(
            source_path=Path("lesson.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    words = result.segments[0].sentences[0].words
    assert tuple(word.text for word in words) == (
        "音",
        "を",
        "聞いて",
        "ください",
        "。",
    )
    assert words[2].time_range == TimeRange(5.582, 5.923)


def test_japanese_word_timing_normalizer_merges_sahen_learning_words() -> None:
    normalizer = JapaneseWordTimingNormalizer(tokenizer=_sudachi_tokenizer())
    source_words = (
        Word(text="散歩", time_range=TimeRange(18.647, 19.069), confidence=1.0),
        Word(text="し", time_range=TimeRange(19.069, 19.189), confidence=0.866),
        Word(text="ましょう", time_range=TimeRange(19.189, 19.47), confidence=0.5),
    )
    sentence = Sentence(
        text="散歩しましょう",
        time_range=TimeRange(18.647, 19.47),
        words=source_words,
    )
    segment = Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )

    result = normalizer.normalize(
        JapaneseWordNormalizationRequest(
            source_path=Path("lesson.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    words = result.segments[0].sentences[0].words
    assert tuple(word.text for word in words) == ("散歩しましょう",)
    assert words[0].time_range == TimeRange(18.647, 19.47)

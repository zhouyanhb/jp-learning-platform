from pathlib import Path

import pytest

from jp_learning_platform.domain import LearningWord, Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseLearningWordNormalizer,
    JapaneseMorpheme,
    SudachiMorphologicalAnalyzer,
)
from jp_learning_platform.workflow.word_normalization_stage import WordNormalizationRequest


class _Analyzer:
    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        structural_fixtures = {
            "N2": (
                JapaneseMorpheme("N", ("名詞", "普通名詞", "助数詞可能")),
                JapaneseMorpheme("2", ("名詞", "数詞", "*")),
            ),
            "2番": (
                JapaneseMorpheme("2", ("名詞", "数詞", "*")),
                JapaneseMorpheme("番", ("名詞", "普通名詞", "助数詞可能")),
            ),
            "ポイントカード": (
                JapaneseMorpheme("ポイント", ("名詞", "普通名詞", "助数詞可能")),
                JapaneseMorpheme("カード", ("名詞", "普通名詞", "一般")),
            ),
            "休みましょうでは": (
                JapaneseMorpheme("休み", ("動詞", "一般", "*"), "休む"),
                JapaneseMorpheme("ましょう", ("助動詞", "*", "*"), "ます"),
                JapaneseMorpheme(
                    "で",
                    ("助動詞", "*", "*"),
                    "だ",
                    conjugation_type="助動詞-ダ",
                ),
                JapaneseMorpheme("は", ("助詞", "係助詞", "*")),
            ),
            "休んだ": (
                JapaneseMorpheme(
                    "休ん",
                    ("動詞", "一般", "*"),
                    "休む",
                    conjugation_type="五段-マ行",
                    conjugation_form="連用形-撥音便",
                ),
                JapaneseMorpheme(
                    "だ",
                    ("助動詞", "*", "*"),
                    "だ",
                    conjugation_type="助動詞-タ",
                    conjugation_form="終止形-一般",
                ),
            ),
            "でした": (
                JapaneseMorpheme(
                    "でし",
                    ("助動詞", "*", "*"),
                    "です",
                    conjugation_type="助動詞-デス",
                    conjugation_form="連用形-一般",
                ),
                JapaneseMorpheme(
                    "た",
                    ("助動詞", "*", "*"),
                    "た",
                    conjugation_type="助動詞-タ",
                    conjugation_form="終止形-一般",
                ),
            ),
            "だった": (
                JapaneseMorpheme(
                    "だっ",
                    ("助動詞", "*", "*"),
                    "だ",
                    conjugation_type="助動詞-ダ",
                    conjugation_form="連用形-促音便",
                ),
                JapaneseMorpheme(
                    "た",
                    ("助動詞", "*", "*"),
                    "た",
                    conjugation_type="助動詞-タ",
                    conjugation_form="終止形-一般",
                ),
            ),
            "お菓子": (
                JapaneseMorpheme("お", ("接頭辞", "*", "*")),
                JapaneseMorpheme("菓子", ("名詞", "普通名詞", "一般")),
            ),
            "おじいさん": (
                JapaneseMorpheme("お", ("接頭辞", "*", "*")),
                JapaneseMorpheme("じい", ("名詞", "普通名詞", "一般")),
                JapaneseMorpheme("さん", ("接尾辞", "名詞的", "一般")),
            ),
            "皆さん": (
                JapaneseMorpheme("皆", ("名詞", "普通名詞", "副詞可能")),
                JapaneseMorpheme("さん", ("接尾辞", "名詞的", "一般")),
            ),
            "りんさん": (
                JapaneseMorpheme(
                    "りん",
                    ("名詞", "固有名詞", "人名", "名"),
                ),
                JapaneseMorpheme("さん", ("接尾辞", "名詞的", "一般")),
            ),
            "第一会場": (
                JapaneseMorpheme("第", ("接頭辞", "*", "*")),
                JapaneseMorpheme("一", ("名詞", "数詞", "*")),
                JapaneseMorpheme("会場", ("名詞", "普通名詞", "一般")),
            ),
            "3冊": (
                JapaneseMorpheme("3", ("名詞", "数詞", "*")),
                JapaneseMorpheme("冊", ("接尾辞", "名詞的", "一般")),
            ),
            "木村先生": (
                JapaneseMorpheme("木村", ("名詞", "固有名詞", "人名", "姓")),
                JapaneseMorpheme("先生", ("名詞", "普通名詞", "一般")),
            ),
            "田中社長": (
                JapaneseMorpheme("田中", ("名詞", "固有名詞", "人名", "姓")),
                JapaneseMorpheme("社長", ("名詞", "普通名詞", "一般")),
            ),
            "森先輩": (
                JapaneseMorpheme("森", ("名詞", "固有名詞", "人名", "姓")),
                JapaneseMorpheme("先輩", ("名詞", "普通名詞", "一般")),
            ),
            "2,3冊": (
                JapaneseMorpheme("2", ("名詞", "数詞", "*")),
                JapaneseMorpheme(",", ("補助記号", "読点", "*")),
                JapaneseMorpheme("3", ("名詞", "数詞", "*")),
                JapaneseMorpheme("冊", ("接尾辞", "名詞的", "一般")),
            ),
            "、確認": (
                JapaneseMorpheme("、", ("補助記号", "読点", "*")),
                JapaneseMorpheme("確認", ("名詞", "普通名詞", "サ変可能")),
            ),
        }
        if text in structural_fixtures:
            return structural_fixtures[text]
        fixtures = {
            "聞いて": (("聞い", "動詞", "一般"), ("て", "助詞", "接続助詞")),
            "話しています": (("話し", "動詞", "一般"), ("て", "助詞", "接続助詞"), ("い", "動詞", "非自立可能"), ("ます", "助動詞", "*")),
            "メールでも": (("メール", "名詞", "普通名詞"), ("で", "助詞", "格助詞"), ("も", "助詞", "係助詞")),
            "いつでも": (("いつ", "代名詞", "*"), ("で", "助詞", "格助詞"), ("も", "助詞", "係助詞")),
            "学生": (("学生", "名詞", "普通名詞"),),
            "聞こえない": (("聞こえ", "動詞", "一般"), ("ない", "助動詞", "*")),
            "散歩しましょう": (("散歩", "名詞", "サ変可能"), ("し", "動詞", "非自立可能"), ("ましょう", "助動詞", "*")),
            "行きました": (("行き", "動詞", "一般"), ("まし", "助動詞", "*"), ("た", "助動詞", "*")),
            "高くない": (("高く", "形容詞", "一般"), ("ない", "形容詞", "非自立可能")),
            "問題がない": (("問題", "名詞", "普通名詞"), ("が", "助詞", "格助詞"), ("ない", "形容詞", "非自立可能")),
            "回答用紙": (("回答", "名詞", "サ変可能"), ("用", "接尾辞", "名詞的"), ("紙", "接尾辞", "名詞的")),
            "回答用": (("回答", "名詞", "サ変可能"), ("用", "接尾辞", "名詞的")),
            "用紙": (("用紙", "名詞", "普通名詞"),),
        }
        return tuple(
            JapaneseMorpheme(
                surface,
                (major, minor, detail),
                dictionary_form="する" if surface == "し" else surface,
            )
            for surface, major, detail in fixtures[text]
            for minor in (detail if major != "名詞" else "普通名詞",)
        )


def _normalize(
    text: str,
    token_texts: tuple[str, ...],
) -> tuple[tuple[Word, ...], tuple[LearningWord, ...]]:
    duration = 1 / len(token_texts)
    words = tuple(
        Word(token, TimeRange(index * duration, (index + 1) * duration), 0.9)
        for index, token in enumerate(token_texts)
    )
    sentence = Sentence(text, TimeRange(0, 1), words)
    segment = Segment(0, text, TimeRange(0, 1), (sentence,))
    result = JapaneseLearningWordNormalizer(_Analyzer()).normalize(
        WordNormalizationRequest(Path("audio.mp3"), (segment,))
    )
    normalized_sentence = result.segments[0].sentences[0]
    return normalized_sentence.words, normalized_sentence.learning_words


@pytest.mark.parametrize(
    ("text", "source", "expected"),
    [
        ("聞いて", ("聞", "いて"), ("聞いて",)),
        ("話しています", ("話しています",), ("話して", "います")),
        ("メールでも", ("メールでも",), ("メール", "でも")),
        ("いつでも", ("いつでも",), ("いつ", "でも")),
        ("学生", ("学", "生"), ("学生",)),
        ("聞こえない", ("聞こえ", "ない"), ("聞こえない",)),
        ("散歩しましょう", ("散歩", "しましょう"), ("散歩しましょう",)),
        ("行きました", ("行き", "まし", "た"), ("行きました",)),
        ("高くない", ("高く", "ない"), ("高くない",)),
        ("問題がない", ("問題", "が", "ない"), ("問題", "が", "ない")),
        ("回答用紙", ("回", "答", "用", "紙"), ("回答", "用紙")),
        ("N2", ("N", "2"), ("N2",)),
        ("2番", ("2", "番"), ("2番",)),
        ("ポイントカード", ("ポイント", "カード"), ("ポイントカード",)),
        ("休みましょうでは", ("休み", "ましょう", "で", "は"), ("休みましょう", "で", "は")),
        ("休んだ", ("休ん", "だ"), ("休んだ",)),
        ("でした", ("でし", "た"), ("でした",)),
        ("だった", ("だっ", "た"), ("だった",)),
        ("お菓子", ("お", "菓子"), ("お菓子",)),
        ("おじいさん", ("お", "じい", "さん"), ("おじいさん",)),
        ("皆さん", ("皆", "さん"), ("皆さん",)),
        ("りんさん", ("りん", "さん"), ("りんさん",)),
        ("第一会場", ("第", "一", "会場"), ("第一", "会場")),
        ("3冊", ("3", "冊"), ("3冊",)),
        ("木村先生", ("木村", "先生"), ("木村先生",)),
        ("田中社長", ("田中", "社長"), ("田中社長",)),
        ("森先輩", ("森", "先輩"), ("森先輩",)),
        ("2,3冊", ("2", ",", "3", "冊"), ("2", "3冊")),
    ],
)
def test_normalizes_learning_units_without_sentence_specific_replacements(
    text: str, source: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    aligned_words, learning_words = _normalize(text, source)
    assert tuple(word.text for word in aligned_words) == source
    assert tuple(word.text for word in learning_words) == expected
    assert learning_words[0].time_range.start_seconds == 0
    assert learning_words[-1].time_range.end_seconds == 1


def test_learning_words_exclude_standalone_punctuation() -> None:
    _aligned_words, learning_words = _normalize("、確認", ("、確認",))

    assert tuple(word.text for word in learning_words) == ("確認",)


def test_combines_non_independent_verb_auxiliary_stem_and_inflected_auxiliary() -> None:
    text = "入っていそうな"
    words = (Word(text, TimeRange(0.0, 1.0), 0.9),)
    sentence = Sentence(text, TimeRange(0.0, 1.0), words)
    segment = Segment(0, text, sentence.time_range, (sentence,))

    result = JapaneseLearningWordNormalizer(SudachiMorphologicalAnalyzer()).normalize(
        WordNormalizationRequest(Path("audio.mp3"), (segment,))
    )

    assert tuple(
        word.text for word in result.segments[0].sentences[0].learning_words
    ) == ("入って", "いそうな")


def test_interpolates_boundaries_when_one_asr_token_is_split() -> None:
    aligned_words, learning_words = _normalize("話しています", ("話しています",))
    assert tuple(word.text for word in aligned_words) == ("話しています",)
    assert learning_words[0].time_range.end_seconds == pytest.approx(0.5)
    assert learning_words[1].time_range.start_seconds == pytest.approx(0.5)
    assert all(word.timing_estimated for word in learning_words)


def test_local_nominal_reanalysis_preserves_merged_word_metadata() -> None:
    text = "回答用紙"
    words = (
        Word("回答", TimeRange(170.602, 171.103), 0.999),
        Word("用", TimeRange(171.103, 171.344), 0.999),
        Word("紙", TimeRange(171.344, 171.504), 0.998),
    )
    sentence = Sentence(text, TimeRange(170.602, 171.504), words)
    segment = Segment(0, text, sentence.time_range, (sentence,))

    result = JapaneseLearningWordNormalizer(_Analyzer()).normalize(
        WordNormalizationRequest(Path("audio.mp3"), (segment,))
    )

    normalized_sentence = result.segments[0].sentences[0]
    assert normalized_sentence.words == words
    normalized = normalized_sentence.learning_words
    assert tuple(word.text for word in normalized) == ("回答", "用紙")
    assert normalized[1] == LearningWord(
        text="用紙",
        start_char=2,
        end_char=4,
        aligned_word_indexes=(1, 2),
        time_range=TimeRange(171.103, 171.504),
        timing_estimated=False,
    )


def test_learning_word_normalization_is_idempotent() -> None:
    text = "話しています"
    source_words = (Word(text, TimeRange(0.0, 1.0), 0.9),)
    sentence = Sentence(text, TimeRange(0.0, 1.0), source_words)
    segment = Segment(0, text, sentence.time_range, (sentence,))
    normalizer = JapaneseLearningWordNormalizer(_Analyzer())
    request = WordNormalizationRequest(Path("audio.mp3"), (segment,))

    first = normalizer.normalize(request)
    second = normalizer.normalize(
        WordNormalizationRequest(Path("audio.mp3"), first.segments)
    )

    assert second == first

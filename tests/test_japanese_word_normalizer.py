from pathlib import Path

import pytest

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseLearningWordNormalizer,
    JapaneseMorpheme,
)
from jp_learning_platform.workflow.word_normalization_stage import WordNormalizationRequest


class _Analyzer:
    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        fixtures = {
            "聞いて": (("聞い", "動詞", "一般"), ("て", "助詞", "接続助詞")),
            "話しています": (("話し", "動詞", "一般"), ("て", "助詞", "接続助詞"), ("い", "動詞", "非自立可能"), ("ます", "助動詞", "*")),
            "メールでも": (("メール", "名詞", "普通名詞"), ("で", "助詞", "格助詞"), ("も", "助詞", "係助詞")),
            "いつでも": (("いつ", "代名詞", "*"), ("で", "助詞", "格助詞"), ("も", "助詞", "係助詞")),
            "学生": (("学生", "名詞", "普通名詞"),),
        }
        return tuple(JapaneseMorpheme(surface, (major, minor)) for surface, major, minor in fixtures[text])


def _normalize(text: str, token_texts: tuple[str, ...]) -> tuple[Word, ...]:
    duration = 1 / len(token_texts)
    words = tuple(
        Word(token, TimeRange(index * duration, (index + 1) * duration), 0.9, "A")
        for index, token in enumerate(token_texts)
    )
    sentence = Sentence(text, TimeRange(0, 1), words, "A")
    segment = Segment(0, text, TimeRange(0, 1), (sentence,), "A")
    result = JapaneseLearningWordNormalizer(_Analyzer()).normalize(
        WordNormalizationRequest(Path("audio.mp3"), (segment,))
    )
    return result.segments[0].sentences[0].words


@pytest.mark.parametrize(
    ("text", "source", "expected"),
    [
        ("聞いて", ("聞", "いて"), ("聞いて",)),
        ("話しています", ("話しています",), ("話して", "います")),
        ("メールでも", ("メールでも",), ("メール", "でも")),
        ("いつでも", ("いつでも",), ("いつ", "でも")),
        ("学生", ("学", "生"), ("学生",)),
    ],
)
def test_normalizes_learning_units_without_sentence_specific_replacements(
    text: str, source: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    words = _normalize(text, source)
    assert tuple(word.text for word in words) == expected
    assert words[0].time_range.start_seconds == 0
    assert words[-1].time_range.end_seconds == 1
    assert all(word.confidence == 0.9 and word.speaker_id == "A" for word in words)


def test_interpolates_boundaries_when_one_asr_token_is_split() -> None:
    words = _normalize("話しています", ("話しています",))
    assert words[0].time_range.end_seconds == pytest.approx(0.5)
    assert words[1].time_range.start_seconds == pytest.approx(0.5)

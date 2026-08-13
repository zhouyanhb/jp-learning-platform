from pathlib import Path

from jp_learning_platform.domain import Document, PipelineContext, Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.japanese_punctuation_restorer import (
    PauseAwareJapaneseCommaRestorer,
)
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    SudachiMorphologicalAnalyzer,
)
from jp_learning_platform.workflow.punctuation_attribution_stage import (
    PunctuationAttributionStage,
)


def _sentence(text: str, extended_text: str, duration: float) -> Sentence:
    words = []
    cursor = 0.0
    for character in text:
        word_duration = duration if character == extended_text else 0.1
        words.append(
            Word(character, TimeRange(cursor, cursor + word_duration), 0.9)
        )
        cursor += word_duration
    return Sentence(text, TimeRange(0.0, cursor), tuple(words))


def test_restores_pause_supported_comma_before_independent_predicate() -> None:
    sentence = _sentence(
        "電車がだんだんゆっくりになって止まりました。",
        "て",
        3.82,
    )
    segment = Segment(0, sentence.text, sentence.time_range, (sentence,))
    context = PipelineContext(
        "run-001",
        Document(Path("audio.mp3"), (segment,)),
        Path("work"),
    )

    result = PunctuationAttributionStage(
        PauseAwareJapaneseCommaRestorer(SudachiMorphologicalAnalyzer())
    ).run(context)

    restored = result.context.document.segments[0].sentences[0]
    assert restored.text == "電車がだんだんゆっくりになって、止まりました。"
    assert "".join(word.text for word in restored.words) == restored.text
    assert result.data["decisions"][0].reason == "pause_supported_connective_comma"


def test_does_not_restore_comma_without_pause_evidence() -> None:
    sentence = _sentence("買って帰ります。", "て", 0.1)
    restorer = PauseAwareJapaneseCommaRestorer(SudachiMorphologicalAnalyzer())

    assert restorer.restore(sentence) is None


def test_does_not_split_connective_auxiliary_verb_chain() -> None:
    sentence = _sentence("話しています。", "て", 3.82)
    restorer = PauseAwareJapaneseCommaRestorer(SudachiMorphologicalAnalyzer())

    assert restorer.restore(sentence) is None

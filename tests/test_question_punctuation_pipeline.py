from pathlib import Path

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    Subtitle,
    TimeRange,
    Word,
)
from jp_learning_platform.workflow import (
    PunctuationAttributionStage,
    SubtitleDisplayNormalizationStage,
)


def _context(question: str, punctuation: str) -> PipelineContext:
    question_word = Word(question, TimeRange(0.0, 1.0))
    punctuation_word = Word(punctuation, TimeRange(1.0, 1.1))
    sentences = (
        Sentence(
            question,
            question_word.time_range,
            words=(question_word,),
            is_question=True,
        ),
        Sentence(punctuation, punctuation_word.time_range, words=(punctuation_word,)),
    )
    return PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=Path("input.mp3"),
            segments=(
                Segment(0, f"{question}{punctuation}", TimeRange(0.0, 1.1), sentences),
            ),
        ),
        working_directory=Path("work"),
    )


def test_punctuation_attribution_preserves_raw_words() -> None:
    result = PunctuationAttributionStage().run(_context("そうですか", '。」'))

    sentence = result.context.document.segments[0].sentences[0]
    assert sentence.text == 'そうですか。」'
    assert tuple(word.text for word in sentence.words) == ("そうですか", '。」')
    assert sentence.is_question
    assert result.data["decisions"][0].attributed_text == '。」'


def test_display_normalization_replaces_period_before_closing_quote() -> None:
    attributed = PunctuationAttributionStage().run(
        _context("そうですか", '。」')
    ).context
    sentence = attributed.document.segments[0].sentences[0]
    context = PipelineContext(
        run_id=attributed.run_id,
        document=Document(
            source_path=attributed.document.source_path,
            segments=attributed.document.segments,
            subtitles=(Subtitle(1, sentence.text, sentence.time_range),),
        ),
        working_directory=attributed.working_directory,
    )

    result = SubtitleDisplayNormalizationStage().run(context)

    assert result.context.document.subtitles[0].text == "そうですか？」"
    assert tuple(word.text for word in sentence.words) == ("そうですか", '。」')
    decision = result.data["decisions"][0]
    assert decision.original_text == "そうですか。」"
    assert decision.display_text == "そうですか？」"


def test_display_normalization_does_not_duplicate_existing_question_mark() -> None:
    word = Word("そうですか？", TimeRange(0.0, 1.0))
    sentence = Sentence(
        word.text,
        word.time_range,
        words=(word,),
        is_question=True,
    )
    context = PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=Path("input.mp3"),
            segments=(Segment(0, sentence.text, sentence.time_range, (sentence,)),),
            subtitles=(Subtitle(1, sentence.text, sentence.time_range),),
        ),
        working_directory=Path("work"),
    )

    result = SubtitleDisplayNormalizationStage().run(context)

    assert result.context.document.subtitles[0].text == "そうですか？"
    assert not result.data["decisions"]


def test_display_normalization_changes_only_final_cue_of_split_question() -> None:
    words = (
        Word("これは", TimeRange(0.0, 1.0)),
        Word("何ですか。", TimeRange(1.0, 2.0)),
    )
    sentence = Sentence(
        "これは 何ですか。",
        TimeRange(0.0, 2.0),
        words=words,
        is_question=True,
    )
    context = PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=Path("input.mp3"),
            segments=(Segment(0, sentence.text, sentence.time_range, (sentence,)),),
            subtitles=(
                Subtitle(1, "これは", TimeRange(0.0, 1.0)),
                Subtitle(2, "何ですか。", TimeRange(1.0, 2.0)),
            ),
        ),
        working_directory=Path("work"),
    )

    result = SubtitleDisplayNormalizationStage().run(context)

    assert tuple(item.text for item in result.context.document.subtitles) == (
        "これは",
        "何ですか？",
    )
    assert tuple(item.subtitle_index for item in result.data["decisions"]) == (2,)

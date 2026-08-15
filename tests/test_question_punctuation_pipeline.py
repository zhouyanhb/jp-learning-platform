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


def _context(
    question: str,
    punctuation: str,
    *,
    is_question: bool = True,
) -> PipelineContext:
    question_word = Word(question, TimeRange(0.0, 1.0))
    punctuation_word = Word(punctuation, TimeRange(1.0, 1.1))
    sentences = (
        Sentence(
            question,
            question_word.time_range,
            words=(question_word,),
            is_question=is_question,
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


def test_punctuation_attribution_attaches_period_to_statement() -> None:
    result = PunctuationAttributionStage().run(
        _context("みなさんこんにちは", "。", is_question=False)
    )

    sentences = result.context.document.segments[0].sentences
    assert len(sentences) == 1
    assert sentences[0].text == "みなさんこんにちは。"
    assert tuple(word.text for word in sentences[0].words) == (
        "みなさんこんにちは",
        "。",
    )
    assert not sentences[0].is_question
    decision = result.data["decisions"][0]
    assert decision.original_sentence_text == "みなさんこんにちは"
    assert decision.resulting_sentence_text == "みなさんこんにちは。"
    assert not decision.is_question


def test_punctuation_attribution_attaches_period_split_by_strong_pause() -> None:
    statement_words = (
        Word("仕事帰りに", TimeRange(7.086, 8.287)),
        Word("平凡な生活を送っていたのですが", TimeRange(14.313, 19.718)),
    )
    period_word = Word("。", TimeRange(19.718, 20.779))
    next_word = Word("そこで告げられた来世は。", TimeRange(20.779, 22.220))
    sentences = (
        Sentence(
            "仕事帰りに平凡な生活を送っていたのですが",
            TimeRange(7.086, 19.718),
            words=statement_words,
        ),
        Sentence("。", period_word.time_range, words=(period_word,)),
        Sentence(next_word.text, next_word.time_range, words=(next_word,)),
    )
    context = PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=Path("episode-2.mp4"),
            segments=(
                Segment(
                    0,
                    "".join(sentence.text for sentence in sentences),
                    TimeRange(7.086, 22.220),
                    sentences,
                ),
            ),
        ),
        working_directory=Path("work"),
    )

    result = PunctuationAttributionStage().run(context)

    output_sentences = result.context.document.segments[0].sentences
    assert tuple(sentence.text for sentence in output_sentences) == (
        "仕事帰りに平凡な生活を送っていたのですが。",
        "そこで告げられた来世は。",
    )
    assert output_sentences[0].time_range == TimeRange(7.086, 20.779)
    assert tuple(word.text for word in output_sentences[0].words) == (
        "仕事帰りに",
        "平凡な生活を送っていたのですが",
        "。",
    )


def test_punctuation_attribution_does_not_attach_opening_punctuation() -> None:
    result = PunctuationAttributionStage().run(
        _context("前の文。", "「", is_question=False)
    )

    assert tuple(
        sentence.text
        for sentence in result.context.document.segments[0].sentences
    ) == ("前の文。", "「")
    assert not result.data["decisions"]


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

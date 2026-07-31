from __future__ import annotations

from pathlib import Path

from jp_learning_platform.domain import LearningWord, Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import WordSubtitleBuilder
from jp_learning_platform.infrastructure.pipeline_config import SubtitleDisplayConfig
from jp_learning_platform.workflow import SubtitleBuildRequest


def test_word_subtitle_builder_uses_sentence_text_and_timing() -> None:
    words = (
        Word(text="日本語", time_range=TimeRange(0.0, 0.5), confidence=0.9),
        Word(text="です", time_range=TimeRange(0.6, 1.0), confidence=0.8),
    )
    sentence = Sentence(
        text="日本語です。",
        time_range=TimeRange(0.0, 1.1),
        words=words,
    )
    segment = Segment(
        position=0,
        text="日本語です。",
        time_range=TimeRange(0.0, 1.1),
        sentences=(sentence,),
    )

    result = WordSubtitleBuilder().build(
        SubtitleBuildRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    assert len(result.subtitles) == 1
    assert result.subtitles[0].index == 1
    assert result.subtitles[0].text == "日本語です。"
    assert result.subtitles[0].time_range == TimeRange(0.0, 1.1)


def test_word_subtitle_builder_falls_back_to_segment_text() -> None:
    segment = Segment(
        position=0,
        text="日本語です。",
        time_range=TimeRange(0.0, 1.1),
    )

    result = WordSubtitleBuilder().build(
        SubtitleBuildRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    assert result.subtitles[0].text == "日本語です。"
    assert result.subtitles[0].time_range == TimeRange(0.0, 1.1)


def test_word_subtitle_builder_splits_long_sentence_at_learning_words() -> None:
    text = "日本語の長い文章を字幕として読みやすく分けます"
    unit_texts = ("日本語", "の", "長い", "文章", "を", "字幕", "として", "読みやすく", "分けます")
    learning_words: list[LearningWord] = []
    cursor = 0
    for index, unit_text in enumerate(unit_texts):
        end = cursor + len(unit_text)
        learning_words.append(
            LearningWord(
                unit_text,
                cursor,
                end,
                (),
                TimeRange(index, index + 1),
                True,
            )
        )
        cursor = end
    sentence = Sentence(
        text,
        TimeRange(0, len(unit_texts)),
        learning_words=tuple(learning_words),
    )
    segment = Segment(0, text, sentence.time_range, (sentence,))

    result = WordSubtitleBuilder(
        SubtitleDisplayConfig(max_chars=10, max_duration_seconds=4)
    ).build(
        SubtitleBuildRequest(
            Path("audio.mp3"), Path("work"), "run-001", (segment,)
        )
    )

    assert len(result.subtitles) > 1
    assert "".join(item.text for item in result.subtitles) == text
    assert all(len(item.text) <= 10 for item in result.subtitles)
    assert all(item.time_range.duration_seconds <= 4 for item in result.subtitles)
    assert tuple(item.index for item in result.subtitles) == tuple(
        range(1, len(result.subtitles) + 1)
    )
    assert all(
        left.time_range.end_seconds == right.time_range.start_seconds
        for left, right in zip(result.subtitles, result.subtitles[1:])
    )


def test_word_subtitle_builder_caps_anomalous_single_unit_display_time() -> None:
    learning_word = LearningWord(
        "休みましょう",
        0,
        6,
        (0,),
        TimeRange(0, 30),
    )
    word = Word("休みましょう", TimeRange(0, 30))
    sentence = Sentence(
        word.text,
        word.time_range,
        (word,),
        learning_words=(learning_word,),
    )
    segment = Segment(0, sentence.text, sentence.time_range, (sentence,))

    result = WordSubtitleBuilder().build(
        SubtitleBuildRequest(
            Path("audio.mp3"), Path("work"), "run-001", (segment,)
        )
    )

    assert result.subtitles[0].text == "休みましょう"
    assert result.subtitles[0].time_range == TimeRange(0, 10)

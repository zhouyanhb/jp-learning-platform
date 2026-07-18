from __future__ import annotations

from pathlib import Path

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import WordSubtitleBuilder
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
        speaker_id="speaker-1",
    )
    segment = Segment(
        position=0,
        text="日本語です。",
        time_range=TimeRange(0.0, 1.1),
        sentences=(sentence,),
        speaker_id="speaker-1",
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
    assert result.subtitles[0].speaker_id == "speaker-1"


def test_word_subtitle_builder_falls_back_to_segment_text() -> None:
    segment = Segment(
        position=0,
        text="日本語です。",
        time_range=TimeRange(0.0, 1.1),
        speaker_id="speaker-2",
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
    assert result.subtitles[0].speaker_id == "speaker-2"

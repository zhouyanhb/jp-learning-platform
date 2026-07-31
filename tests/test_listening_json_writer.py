from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.domain import (
    LearningWord,
    Segment,
    Sentence,
    Subtitle,
    TimeRange,
    Word,
)
from jp_learning_platform.infrastructure import (
    CompositeSubtitleWriter,
    ListeningJsonWriter,
    SrtSubtitleWriter,
)
from jp_learning_platform.workflow import SubtitleWriteRequest


def _write_request(tmp_path: Path) -> SubtitleWriteRequest:
    source_path = tmp_path / "lesson.mp3"
    word = Word(
        text="日本語",
        time_range=TimeRange(0.0, 0.4),
        confidence=0.91,
    )
    sentence = Sentence(
        text="日本語です。",
        time_range=TimeRange(0.0, 1.0),
        words=(word,),
        learning_words=(
            LearningWord(
                "日本語",
                0,
                3,
                (0,),
                TimeRange(0.0, 0.4),
            ),
        ),
    )
    segment = Segment(
        position=0,
        text="日本語です。",
        time_range=TimeRange(0.0, 1.0),
        sentences=(sentence,),
    )
    subtitle = Subtitle(
        index=1,
        text="日本語です。",
        time_range=TimeRange(0.0, 1.0),
    )
    return SubtitleWriteRequest(
        source_path=source_path,
        working_directory=tmp_path / "work",
        run_id="run-001",
        segments=(segment,),
        subtitles=(subtitle,),
    )


def test_listening_json_writer_writes_structured_transcript_json(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "output"
    writer = ListeningJsonWriter(output_directory=output_directory)

    result = writer.write(_write_request(tmp_path))

    assert result.output_path == output_directory / "lesson.json"
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.4"
    assert payload["source_path"] == str(tmp_path / "lesson.mp3")
    assert payload["run_id"] == "run-001"
    assert payload["segments"][0]["sentences"][0]["words"][0] == {
        "text": "日本語",
        "confidence": 0.91,
        "start_seconds": 0.0,
        "end_seconds": 0.4,
        "duration_seconds": 0.4,
    }
    assert payload["segments"][0]["sentences"][0]["learning_words"][0] == {
        "text": "日本語",
        "start_char": 0,
        "end_char": 3,
        "aligned_word_indexes": [0],
        "timing_estimated": False,
        "start_seconds": 0.0,
        "end_seconds": 0.4,
        "duration_seconds": 0.4,
    }
    assert payload["subtitles"][0]["text"] == "日本語です。"
    assert "speaker_id" not in payload["segments"][0]
    assert "speaker_id" not in payload["subtitles"][0]


def test_composite_subtitle_writer_returns_json_primary_and_exports_srt(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "output"
    writer = CompositeSubtitleWriter(
        primary_writer=ListeningJsonWriter(output_directory=output_directory),
        export_writers=(SrtSubtitleWriter(output_directory=output_directory),),
    )

    result = writer.write(_write_request(tmp_path))

    assert result.output_path == output_directory / "lesson.json"
    assert (output_directory / "lesson.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,000\n日本語です。\n\n"
    )

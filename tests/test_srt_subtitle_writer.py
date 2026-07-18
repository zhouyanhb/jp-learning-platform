from __future__ import annotations

from pathlib import Path

from jp_learning_platform.domain import Subtitle, TimeRange
from jp_learning_platform.infrastructure import (
    SrtSubtitleWriter,
    format_srt_subtitle,
    format_srt_timestamp,
)
from jp_learning_platform.workflow import SubtitleWriteRequest


def test_format_srt_timestamp_uses_subrip_time_format() -> None:
    assert format_srt_timestamp(3723.456) == "01:02:03,456"


def test_format_srt_subtitle_writes_index_timing_and_text() -> None:
    subtitle = Subtitle(
        index=1,
        text="日本語です。",
        time_range=TimeRange(1.2, 3.45),
        speaker_id="speaker-1",
    )

    assert (
        format_srt_subtitle(subtitle)
        == "1\n00:00:01,200 --> 00:00:03,450\n日本語です。\n\n"
    )


def test_srt_subtitle_writer_writes_utf8_srt_file(tmp_path: Path) -> None:
    source_path = tmp_path / "audio.mp3"
    output_directory = tmp_path / "output"
    subtitle = Subtitle(
        index=1,
        text="日本語です。",
        time_range=TimeRange(0.0, 1.5),
    )
    writer = SrtSubtitleWriter(output_directory=output_directory)

    result = writer.write(
        SubtitleWriteRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(),
            subtitles=(subtitle,),
        )
    )

    assert result.output_path == output_directory / "audio.srt"
    assert result.output_path.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\n日本語です。\n\n"
    )

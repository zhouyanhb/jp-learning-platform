"""SRT subtitle writer adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jp_learning_platform.domain import Subtitle
from jp_learning_platform.workflow.subtitle_writer_stage import (
    SubtitleWrite,
    SubtitleWriteRequest,
)

MILLISECONDS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("seconds must be non-negative.")

    total_milliseconds = round(seconds * MILLISECONDS_PER_SECOND)
    hours, remainder = divmod(
        total_milliseconds,
        SECONDS_PER_HOUR * MILLISECONDS_PER_SECOND,
    )
    minutes, remainder = divmod(
        remainder,
        SECONDS_PER_MINUTE * MILLISECONDS_PER_SECOND,
    )
    whole_seconds, milliseconds = divmod(remainder, MILLISECONDS_PER_SECOND)

    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def format_srt_subtitle(subtitle: Subtitle) -> str:
    start = format_srt_timestamp(subtitle.time_range.start_seconds)
    end = format_srt_timestamp(subtitle.time_range.end_seconds)
    return f"{subtitle.index}\n{start} --> {end}\n{subtitle.text}\n\n"


@dataclass(frozen=True, slots=True)
class SrtSubtitleWriter:
    """Write validated domain subtitles to UTF-8 SRT files."""

    output_directory: Path = Path("output")

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_directory", Path(self.output_directory))

    def write(self, request: SubtitleWriteRequest) -> SubtitleWrite:
        if not isinstance(request, SubtitleWriteRequest):
            raise TypeError("request must be a SubtitleWriteRequest.")

        output_path = self.output_path_for(request.source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(format_srt_subtitle(subtitle) for subtitle in request.subtitles)
        output_path.write_text(content, encoding="utf-8")

        return SubtitleWrite(
            source_path=request.source_path,
            output_path=output_path,
        )

    def output_path_for(self, source_path: Path) -> Path:
        return self.output_directory / f"{Path(source_path).stem}.srt"


__all__ = [
    "SrtSubtitleWriter",
    "format_srt_subtitle",
    "format_srt_timestamp",
]

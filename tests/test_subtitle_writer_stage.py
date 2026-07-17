from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Subtitle,
    TimeRange,
)
from jp_learning_platform.workflow import (
    InvalidSubtitleWriteError,
    InvalidSubtitleWriterError,
    MissingSubtitlesToWriteError,
    StageResult,
    SubtitleWrite,
    SubtitleWriteRequest,
    SubtitleWriterStage,
)


@dataclass(slots=True)
class FakeWriter:
    write_result: SubtitleWrite
    requests: list[SubtitleWriteRequest]

    def write(self, request: SubtitleWriteRequest) -> SubtitleWrite:
        self.requests.append(request)
        return self.write_result


@dataclass(frozen=True, slots=True)
class InvalidWriteWriter:
    def write(self, request: SubtitleWriteRequest) -> object:
        return request


def _segment() -> Segment:
    return Segment(
        position=0,
        text="Nihongo desu.",
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
    )


def _subtitle(index: int = 1, text: str = "Nihongo desu.") -> Subtitle:
    return Subtitle(
        index=index,
        text=text,
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
    )


def _context(
    source_path: Path,
    subtitles: tuple[Subtitle, ...],
    segments: tuple[Segment, ...] = (),
) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=source_path,
            segments=segments,
            subtitles=subtitles,
        ),
        working_directory=source_path.parent / "work",
    )


def test_subtitle_writer_stage_writes_existing_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    output_path = tmp_path / "work" / "input.srt"
    segment = _segment()
    subtitle = _subtitle()
    writer = FakeWriter(
        write_result=SubtitleWrite(
            source_path=source_path,
            output_path=output_path,
        ),
        requests=[],
    )
    context = _context(source_path, (subtitle,), segments=(segment,))

    result = SubtitleWriterStage(writer=writer).run(context)

    assert isinstance(result, StageResult)
    assert result.stage_name == "subtitle-writer"
    assert result.context is context
    assert writer.requests == [
        SubtitleWriteRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(segment,),
            subtitles=(subtitle,),
        )
    ]


def test_subtitle_writer_stage_accepts_custom_stage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = _subtitle()
    writer = FakeWriter(
        write_result=SubtitleWrite(
            source_path=source_path,
            output_path=tmp_path / "work" / "input.srt",
        ),
        requests=[],
    )
    stage = SubtitleWriterStage(writer=writer, name="  writer  ")

    result = stage.run(_context(source_path, (subtitle,)))

    assert stage.name == "writer"
    assert result.stage_name == "writer"


def test_subtitle_writer_stage_rejects_missing_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    writer = FakeWriter(
        write_result=SubtitleWrite(
            source_path=source_path,
            output_path=tmp_path / "work" / "input.srt",
        ),
        requests=[],
    )
    stage = SubtitleWriterStage(writer=writer)

    with pytest.raises(MissingSubtitlesToWriteError):
        stage.run(_context(source_path, ()))

    assert writer.requests == []


def test_subtitle_writer_stage_rejects_invalid_writer() -> None:
    with pytest.raises(InvalidSubtitleWriterError):
        SubtitleWriterStage(writer=object())


def test_subtitle_writer_stage_rejects_invalid_write_return(
    tmp_path: Path,
) -> None:
    stage = SubtitleWriterStage(writer=InvalidWriteWriter())

    with pytest.raises(InvalidSubtitleWriteError, match="SubtitleWrite"):
        stage.run(_context(tmp_path / "input.wav", (_subtitle(),)))


def test_subtitle_writer_stage_rejects_mismatched_source_path(
    tmp_path: Path,
) -> None:
    writer = FakeWriter(
        write_result=SubtitleWrite(
            source_path=tmp_path / "other.wav",
            output_path=tmp_path / "work" / "input.srt",
        ),
        requests=[],
    )
    stage = SubtitleWriterStage(writer=writer)

    with pytest.raises(InvalidSubtitleWriteError, match="source path"):
        stage.run(_context(tmp_path / "input.wav", (_subtitle(),)))


def test_subtitle_write_normalizes_paths() -> None:
    write = SubtitleWrite(
        source_path="audio/input.wav",
        output_path="work/input.srt",
    )

    assert write.source_path == Path("audio/input.wav")
    assert write.output_path == Path("work/input.srt")


def test_subtitle_write_is_immutable() -> None:
    write = SubtitleWrite(
        source_path=Path("audio/input.wav"),
        output_path=Path("work/input.srt"),
    )

    with pytest.raises(FrozenInstanceError):
        write.output_path = Path("other.srt")

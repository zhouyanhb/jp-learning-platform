from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

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
    InvalidSubtitleBuildError,
    InvalidSubtitleBuilderError,
    MissingSubtitleBuildSegmentsError,
    StageResult,
    SubtitleBuild,
    SubtitleBuildRequest,
    SubtitleBuilderStage,
)


@dataclass(slots=True)
class FakeBuilder:
    build_result: SubtitleBuild
    requests: list[SubtitleBuildRequest]

    def build(self, request: SubtitleBuildRequest) -> SubtitleBuild:
        self.requests.append(request)
        return self.build_result


@dataclass(frozen=True, slots=True)
class InvalidBuildBuilder:
    def build(self, request: SubtitleBuildRequest) -> object:
        return request


def _segment() -> Segment:
    words = (
        Word(
            text="Nihongo",
            time_range=TimeRange(start_seconds=0.0, end_seconds=0.8),
            confidence=0.93,
        ),
        Word(
            text="desu.",
            time_range=TimeRange(start_seconds=0.9, end_seconds=1.4),
            confidence=0.91,
        ),
    )
    sentence = Sentence(
        text="Nihongo desu.",
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
        words=words,
    )
    return Segment(
        position=0,
        text="Nihongo desu.",
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
        sentences=(sentence,),
    )


def _subtitle(index: int = 1, text: str = "Nihongo desu.") -> Subtitle:
    return Subtitle(
        index=index,
        text=text,
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
    )


def _context(
    source_path: Path,
    segments: tuple[Segment, ...],
    subtitles: tuple[Subtitle, ...] = (),
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


def test_subtitle_builder_stage_builds_subtitles_from_segments(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input.wav"
    segment = _segment()
    subtitle = _subtitle()
    builder = FakeBuilder(
        build_result=SubtitleBuild(
            source_path=source_path,
            subtitles=(subtitle,),
        ),
        requests=[],
    )

    result = SubtitleBuilderStage(builder=builder).run(
        _context(source_path, (segment,))
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "subtitle-builder"
    assert builder.requests == [
        SubtitleBuildRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(segment,),
        )
    ]
    assert result.context.document.source_path == source_path
    assert result.context.document.segments == (segment,)
    assert result.context.document.subtitles == (subtitle,)
    assert result.context.run_id == "run-001"
    assert result.context.working_directory == tmp_path / "work"


def test_subtitle_builder_stage_replaces_existing_subtitles(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input.wav"
    segment = _segment()
    existing_subtitle = _subtitle(text="Old subtitle.")
    built_subtitle = _subtitle(text="Built subtitle.")
    builder = FakeBuilder(
        build_result=SubtitleBuild(
            source_path=source_path,
            subtitles=(built_subtitle,),
        ),
        requests=[],
    )

    result = SubtitleBuilderStage(builder=builder).run(
        _context(source_path, (segment,), subtitles=(existing_subtitle,))
    )

    assert result.context.document.segments == (segment,)
    assert result.context.document.subtitles == (built_subtitle,)


def test_subtitle_builder_stage_accepts_custom_stage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    builder = FakeBuilder(
        build_result=SubtitleBuild(
            source_path=source_path,
            subtitles=(_subtitle(),),
        ),
        requests=[],
    )
    stage = SubtitleBuilderStage(builder=builder, name="  builder  ")

    result = stage.run(_context(source_path, (_segment(),)))

    assert stage.name == "builder"
    assert result.stage_name == "builder"


def test_subtitle_builder_stage_rejects_missing_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    builder = FakeBuilder(
        build_result=SubtitleBuild(
            source_path=source_path,
            subtitles=(_subtitle(),),
        ),
        requests=[],
    )
    stage = SubtitleBuilderStage(builder=builder)

    with pytest.raises(MissingSubtitleBuildSegmentsError):
        stage.run(_context(source_path, ()))

    assert builder.requests == []


def test_subtitle_builder_stage_rejects_invalid_builder() -> None:
    with pytest.raises(InvalidSubtitleBuilderError):
        SubtitleBuilderStage(builder=object())


def test_subtitle_builder_stage_rejects_invalid_build_return(
    tmp_path: Path,
) -> None:
    stage = SubtitleBuilderStage(builder=InvalidBuildBuilder())

    with pytest.raises(InvalidSubtitleBuildError, match="SubtitleBuild"):
        stage.run(_context(tmp_path / "input.wav", (_segment(),)))


def test_subtitle_builder_stage_rejects_mismatched_source_path(
    tmp_path: Path,
) -> None:
    builder = FakeBuilder(
        build_result=SubtitleBuild(
            source_path=tmp_path / "other.wav",
            subtitles=(_subtitle(),),
        ),
        requests=[],
    )
    stage = SubtitleBuilderStage(builder=builder)

    with pytest.raises(InvalidSubtitleBuildError, match="source path"):
        stage.run(_context(tmp_path / "input.wav", (_segment(),)))


def test_subtitle_build_requires_subtitles() -> None:
    with pytest.raises(ValueError, match="subtitles"):
        SubtitleBuild(source_path=Path("input.wav"), subtitles=())


def test_subtitle_build_is_immutable() -> None:
    build = SubtitleBuild(
        source_path=Path("input.wav"),
        subtitles=(_subtitle(),),
    )

    with pytest.raises(FrozenInstanceError):
        build.subtitles = ()

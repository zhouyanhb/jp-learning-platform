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
    InvalidSubtitleMergeError,
    InvalidSubtitleMergerError,
    MissingSubtitlesToMergeError,
    StageResult,
    SubtitleMerge,
    SubtitleMergeRequest,
    SubtitleMergerStage,
)


@dataclass(slots=True)
class FakeMerger:
    merge_result: SubtitleMerge
    requests: list[SubtitleMergeRequest]

    def merge(self, request: SubtitleMergeRequest) -> SubtitleMerge:
        self.requests.append(request)
        return self.merge_result


@dataclass(frozen=True, slots=True)
class InvalidMergeMerger:
    def merge(self, request: SubtitleMergeRequest) -> object:
        return request


def _segment() -> Segment:
    return Segment(
        position=0,
        text="Nihongo desu. Benkyou shimasu.",
        time_range=TimeRange(start_seconds=0.0, end_seconds=3.0),
    )


def _subtitle(
    index: int,
    text: str,
    start_seconds: float,
    end_seconds: float,
) -> Subtitle:
    return Subtitle(
        index=index,
        text=text,
        time_range=TimeRange(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        ),
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


def test_subtitle_merger_stage_merges_existing_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    segment = _segment()
    first = _subtitle(1, "Nihongo desu.", 0.0, 1.5)
    second = _subtitle(2, "Benkyou shimasu.", 1.5, 3.0)
    merged = _subtitle(1, "Nihongo desu. Benkyou shimasu.", 0.0, 3.0)
    merger = FakeMerger(
        merge_result=SubtitleMerge(
            source_path=source_path,
            subtitles=(merged,),
        ),
        requests=[],
    )

    result = SubtitleMergerStage(merger=merger).run(
        _context(source_path, (first, second), segments=(segment,))
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "subtitle-merger"
    assert merger.requests == [
        SubtitleMergeRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(segment,),
            subtitles=(first, second),
        )
    ]
    assert result.context.document.source_path == source_path
    assert result.context.document.segments == (segment,)
    assert result.context.document.subtitles == (merged,)
    assert result.context.run_id == "run-001"
    assert result.context.working_directory == tmp_path / "work"


def test_subtitle_merger_stage_replaces_existing_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    first = _subtitle(1, "First.", 0.0, 1.0)
    second = _subtitle(2, "Second.", 1.0, 2.0)
    merged = _subtitle(1, "First. Second.", 0.0, 2.0)
    merger = FakeMerger(
        merge_result=SubtitleMerge(
            source_path=source_path,
            subtitles=(merged,),
        ),
        requests=[],
    )

    result = SubtitleMergerStage(merger=merger).run(
        _context(source_path, (first, second))
    )

    assert result.context.document.subtitles == (merged,)


def test_subtitle_merger_stage_accepts_custom_stage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = _subtitle(1, "Nihongo desu.", 0.0, 1.5)
    merger = FakeMerger(
        merge_result=SubtitleMerge(
            source_path=source_path,
            subtitles=(subtitle,),
        ),
        requests=[],
    )
    stage = SubtitleMergerStage(merger=merger, name="  merger  ")

    result = stage.run(_context(source_path, (subtitle,)))

    assert stage.name == "merger"
    assert result.stage_name == "merger"


def test_subtitle_merger_stage_rejects_missing_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = _subtitle(1, "Nihongo desu.", 0.0, 1.5)
    merger = FakeMerger(
        merge_result=SubtitleMerge(
            source_path=source_path,
            subtitles=(subtitle,),
        ),
        requests=[],
    )
    stage = SubtitleMergerStage(merger=merger)

    with pytest.raises(MissingSubtitlesToMergeError):
        stage.run(_context(source_path, ()))

    assert merger.requests == []


def test_subtitle_merger_stage_rejects_invalid_merger() -> None:
    with pytest.raises(InvalidSubtitleMergerError):
        SubtitleMergerStage(merger=object())


def test_subtitle_merger_stage_rejects_invalid_merge_return(
    tmp_path: Path,
) -> None:
    stage = SubtitleMergerStage(merger=InvalidMergeMerger())

    with pytest.raises(InvalidSubtitleMergeError, match="SubtitleMerge"):
        stage.run(
            _context(
                tmp_path / "input.wav",
                (_subtitle(1, "Nihongo desu.", 0.0, 1.5),),
            )
        )


def test_subtitle_merger_stage_rejects_mismatched_source_path(
    tmp_path: Path,
) -> None:
    merger = FakeMerger(
        merge_result=SubtitleMerge(
            source_path=tmp_path / "other.wav",
            subtitles=(_subtitle(1, "Nihongo desu.", 0.0, 1.5),),
        ),
        requests=[],
    )
    stage = SubtitleMergerStage(merger=merger)

    with pytest.raises(InvalidSubtitleMergeError, match="source path"):
        stage.run(
            _context(
                tmp_path / "input.wav",
                (_subtitle(1, "Nihongo desu.", 0.0, 1.5),),
            )
        )


def test_subtitle_merge_requires_subtitles() -> None:
    with pytest.raises(ValueError, match="subtitles"):
        SubtitleMerge(source_path=Path("input.wav"), subtitles=())


def test_subtitle_merge_is_immutable() -> None:
    merge = SubtitleMerge(
        source_path=Path("input.wav"),
        subtitles=(_subtitle(1, "Nihongo desu.", 0.0, 1.5),),
    )

    with pytest.raises(FrozenInstanceError):
        merge.subtitles = ()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import Document, PipelineContext, Segment, TimeRange
from jp_learning_platform.workflow import (
    HomophoneResolution,
    HomophoneResolutionDecision,
    HomophoneResolutionRequest,
    HomophoneResolutionStage,
    InvalidHomophoneResolutionError,
    InvalidHomophoneResolverError,
    MissingHomophoneSegmentsError,
    StageResult,
)


@dataclass(slots=True)
class FakeHomophoneResolver:
    resolution: HomophoneResolution
    requests: list[HomophoneResolutionRequest]

    def resolve(
        self,
        request: HomophoneResolutionRequest,
    ) -> HomophoneResolution:
        self.requests.append(request)
        return self.resolution


@dataclass(frozen=True, slots=True)
class InvalidHomophoneResolver:
    def resolve(self, request: HomophoneResolutionRequest) -> object:
        return request


def _segment() -> Segment:
    return Segment(
        position=0,
        text="懲戒N2",
        time_range=TimeRange(0.0, 1.0),
    )


def _context(source_path: Path, segments: tuple[Segment, ...]) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path, segments=segments),
        working_directory=source_path.parent / "work",
    )


def test_homophone_resolution_stage_replaces_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "lesson.mp3"
    resolved_segment = Segment(
        position=0,
        text="聴解N2",
        time_range=TimeRange(0.0, 1.0),
    )
    decision = HomophoneResolutionDecision(
        segment_position=0,
        sentence_index=0,
        original_text="懲戒",
        selected_text="聴解",
        reading="ちょうかい",
        accepted=True,
        reason="accepted_same_reading_context",
    )
    resolver = FakeHomophoneResolver(
        resolution=HomophoneResolution(
            source_path=source_path,
            segments=(resolved_segment,),
            decisions=(decision,),
        ),
        requests=[],
    )

    result = HomophoneResolutionStage(resolver=resolver).run(
        _context(source_path, (_segment(),))
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "homophone-resolution"
    assert result.context.document.segments == (resolved_segment,)
    assert result.data == {"decisions": (decision,)}
    assert resolver.requests == [
        HomophoneResolutionRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(_segment(),),
        )
    ]


def test_homophone_resolution_stage_rejects_missing_segments(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "lesson.mp3"
    resolver = FakeHomophoneResolver(
        resolution=HomophoneResolution(
            source_path=source_path,
            segments=(_segment(),),
        ),
        requests=[],
    )

    with pytest.raises(MissingHomophoneSegmentsError):
        HomophoneResolutionStage(resolver=resolver).run(_context(source_path, ()))

    assert resolver.requests == []


def test_homophone_resolution_stage_rejects_invalid_resolver() -> None:
    with pytest.raises(InvalidHomophoneResolverError):
        HomophoneResolutionStage(resolver=object())


def test_homophone_resolution_stage_rejects_invalid_return(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "lesson.mp3"

    with pytest.raises(InvalidHomophoneResolutionError):
        HomophoneResolutionStage(resolver=InvalidHomophoneResolver()).run(
            _context(source_path, (_segment(),))
        )

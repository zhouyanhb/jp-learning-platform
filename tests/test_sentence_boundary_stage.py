from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import Document, PipelineContext, Segment, TimeRange
from jp_learning_platform.workflow import (
    InvalidSentenceBoundaryResolutionError,
    InvalidSentenceBoundaryResolverError,
    SentenceBoundaryDecision,
    SentenceBoundaryResolution,
    SentenceBoundaryResolutionRequest,
    SentenceBoundaryResolutionStage,
)


@dataclass(slots=True)
class FakeSentenceBoundaryResolver:
    resolution: SentenceBoundaryResolution
    requests: list[SentenceBoundaryResolutionRequest]

    def resolve(
        self,
        request: SentenceBoundaryResolutionRequest,
    ) -> SentenceBoundaryResolution:
        self.requests.append(request)
        return self.resolution


class InvalidSentenceBoundaryResolver:
    def resolve(self, request: SentenceBoundaryResolutionRequest) -> object:
        return request


def _segment(text: str = "日本語です") -> Segment:
    return Segment(
        position=0,
        text=text,
        time_range=TimeRange(0.0, 1.0),
    )


def _context(source_path: Path, segments: tuple[Segment, ...]) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path, segments=segments),
        working_directory=source_path.parent / "work",
    )


def test_sentence_boundary_stage_updates_document_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "lesson.mp3"
    resolved_segment = _segment("日本語です今日は")
    decision = SentenceBoundaryDecision(
        segment_position=0,
        sentence_index=0,
        word_index=1,
        gap_seconds=0.6,
        reason="pause_after_sentence_final",
        left_text="日本語です",
        right_text="今日は",
    )
    resolver = FakeSentenceBoundaryResolver(
        resolution=SentenceBoundaryResolution(
            source_path=source_path,
            segments=(resolved_segment,),
            decisions=(decision,),
        ),
        requests=[],
    )
    stage = SentenceBoundaryResolutionStage(resolver=resolver)

    result = stage.run(_context(source_path, (_segment(),)))

    assert result.context.document.segments == (resolved_segment,)
    assert result.data == {"decisions": (decision,)}
    assert resolver.requests == [
        SentenceBoundaryResolutionRequest(
            source_path=source_path,
            working_directory=source_path.parent / "work",
            run_id="run-001",
            segments=(_segment(),),
        )
    ]


def test_sentence_boundary_stage_rejects_invalid_resolver() -> None:
    with pytest.raises(InvalidSentenceBoundaryResolverError):
        SentenceBoundaryResolutionStage(resolver=object())


def test_sentence_boundary_stage_rejects_invalid_return(tmp_path: Path) -> None:
    stage = SentenceBoundaryResolutionStage(resolver=InvalidSentenceBoundaryResolver())

    with pytest.raises(InvalidSentenceBoundaryResolutionError):
        stage.run(_context(tmp_path / "lesson.mp3", (_segment(),)))

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
    InvalidWhisperXAlignerError,
    InvalidWhisperXAlignmentError,
    MissingWhisperSegmentsError,
    StageResult,
    WhisperXAlignment,
    WhisperXAlignmentRequest,
    WhisperXAlignmentStage,
)


@dataclass(slots=True)
class FakeAligner:
    alignment: WhisperXAlignment
    requests: list[WhisperXAlignmentRequest]

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        self.requests.append(request)
        return self.alignment


@dataclass(frozen=True, slots=True)
class InvalidAlignmentAligner:
    def align(self, request: WhisperXAlignmentRequest) -> object:
        return request


def _raw_segment() -> Segment:
    return Segment(
        position=0,
        text="日本語です",
        time_range=TimeRange(start_seconds=0.0, end_seconds=2.0),
    )


def _aligned_segment() -> Segment:
    words = (
        Word(
            text="日本語",
            time_range=TimeRange(start_seconds=0.0, end_seconds=0.9),
            confidence=0.95,
        ),
        Word(
            text="です",
            time_range=TimeRange(start_seconds=1.0, end_seconds=1.8),
            confidence=0.92,
        ),
    )
    sentence = Sentence(
        text="日本語です",
        time_range=TimeRange(start_seconds=0.0, end_seconds=2.0),
        words=words,
    )
    return Segment(
        position=0,
        text="日本語です",
        time_range=TimeRange(start_seconds=0.0, end_seconds=2.0),
        sentences=(sentence,),
    )


def _context(source_path: Path, segments: tuple[Segment, ...]) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path, segments=segments),
        working_directory=source_path.parent / "work",
    )


def test_whisperx_alignment_stage_aligns_existing_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    raw_segment = _raw_segment()
    aligned_segment = _aligned_segment()
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(aligned_segment,),
        ),
        requests=[],
    )

    result = WhisperXAlignmentStage(aligner=aligner).run(
        _context(source_path, (raw_segment,))
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "whisperx-alignment"
    assert aligner.requests == [
        WhisperXAlignmentRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(raw_segment,),
        )
    ]
    assert result.context.document.source_path == source_path
    assert result.context.document.segments == (aligned_segment,)
    assert result.context.run_id == "run-001"
    assert result.context.working_directory == tmp_path / "work"


def test_whisperx_alignment_stage_preserves_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = Subtitle(
        index=1,
        text="字幕",
        time_range=TimeRange(start_seconds=0.0, end_seconds=2.0),
    )
    context = PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=source_path,
            segments=(_raw_segment(),),
            subtitles=(subtitle,),
        ),
        working_directory=tmp_path / "work",
    )
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(_aligned_segment(),),
        ),
        requests=[],
    )

    result = WhisperXAlignmentStage(aligner=aligner).run(context)

    assert result.context.document.subtitles == (subtitle,)


def test_whisperx_alignment_stage_accepts_custom_stage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(_aligned_segment(),),
        ),
        requests=[],
    )
    stage = WhisperXAlignmentStage(aligner=aligner, name="  whisperx  ")

    result = stage.run(_context(source_path, (_raw_segment(),)))

    assert stage.name == "whisperx"
    assert result.stage_name == "whisperx"


def test_whisperx_alignment_stage_rejects_missing_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(_aligned_segment(),),
        ),
        requests=[],
    )
    stage = WhisperXAlignmentStage(aligner=aligner)

    with pytest.raises(MissingWhisperSegmentsError):
        stage.run(_context(source_path, ()))

    assert aligner.requests == []


def test_whisperx_alignment_stage_rejects_invalid_aligner() -> None:
    with pytest.raises(InvalidWhisperXAlignerError):
        WhisperXAlignmentStage(aligner=object())


def test_whisperx_alignment_stage_rejects_invalid_alignment_return(
    tmp_path: Path,
) -> None:
    stage = WhisperXAlignmentStage(aligner=InvalidAlignmentAligner())

    with pytest.raises(InvalidWhisperXAlignmentError, match="WhisperXAlignment"):
        stage.run(_context(tmp_path / "input.wav", (_raw_segment(),)))


def test_whisperx_alignment_stage_rejects_mismatched_source_path(
    tmp_path: Path,
) -> None:
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=tmp_path / "other.wav",
            segments=(_aligned_segment(),),
        ),
        requests=[],
    )
    stage = WhisperXAlignmentStage(aligner=aligner)

    with pytest.raises(InvalidWhisperXAlignmentError, match="source path"):
        stage.run(_context(tmp_path / "input.wav", (_raw_segment(),)))


def test_whisperx_alignment_requires_segments() -> None:
    with pytest.raises(ValueError, match="segments"):
        WhisperXAlignment(source_path=Path("input.wav"), segments=())


def test_whisperx_alignment_is_immutable() -> None:
    alignment = WhisperXAlignment(
        source_path=Path("input.wav"),
        segments=(_aligned_segment(),),
    )

    with pytest.raises(FrozenInstanceError):
        alignment.segments = ()

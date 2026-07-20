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
    InvalidQwenRepairError,
    InvalidQwenRepairerError,
    MissingAlignedSegmentsError,
    QwenRepair,
    QwenRepairDecision,
    QwenRepairRequest,
    QwenRepairStage,
    StageResult,
)


@dataclass(slots=True)
class FakeRepairer:
    repair_result: QwenRepair
    requests: list[QwenRepairRequest]

    def repair(self, request: QwenRepairRequest) -> QwenRepair:
        self.requests.append(request)
        return self.repair_result


@dataclass(frozen=True, slots=True)
class InvalidRepairer:
    def repair(self, request: QwenRepairRequest) -> object:
        return request


def _aligned_segment() -> Segment:
    words = (
        Word(
            text="nihongo",
            time_range=TimeRange(start_seconds=0.0, end_seconds=0.8),
            confidence=0.93,
        ),
        Word(
            text="desu",
            time_range=TimeRange(start_seconds=0.9, end_seconds=1.4),
            confidence=0.91,
        ),
    )
    sentence = Sentence(
        text="nihongo desu",
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
        words=words,
    )
    return Segment(
        position=0,
        text="nihongo desu",
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
        sentences=(sentence,),
    )


def _repaired_segment() -> Segment:
    words = (
        Word(
            text="nihongo",
            time_range=TimeRange(start_seconds=0.0, end_seconds=0.8),
            confidence=0.93,
        ),
        Word(
            text="desu",
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


def _context(source_path: Path, segments: tuple[Segment, ...]) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path, segments=segments),
        working_directory=source_path.parent / "work",
    )


def test_qwen_repair_stage_repairs_existing_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    aligned_segment = _aligned_segment()
    repaired_segment = _repaired_segment()
    repairer = FakeRepairer(
        repair_result=QwenRepair(
            source_path=source_path,
            segments=(repaired_segment,),
        ),
        requests=[],
    )

    result = QwenRepairStage(repairer=repairer).run(
        _context(source_path, (aligned_segment,))
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "qwen-repair"
    assert repairer.requests == [
        QwenRepairRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(aligned_segment,),
        )
    ]
    assert result.context.document.source_path == source_path
    assert result.context.document.segments == (repaired_segment,)
    assert result.context.run_id == "run-001"
    assert result.context.working_directory == tmp_path / "work"


def test_qwen_repair_stage_exposes_decisions_as_stage_data(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    decision = QwenRepairDecision(
        segment_position=0,
        original_text="懲戒N2",
        raw_text="聴解N2",
        candidate_text="聴解N2",
        selected_text="聴解N2",
        accepted=True,
        reason="accepted",
        length_delta_ratio=0.0,
        content_change_ratio=0.1,
    )
    repairer = FakeRepairer(
        repair_result=QwenRepair(
            source_path=source_path,
            segments=(_repaired_segment(),),
            decisions=(decision,),
        ),
        requests=[],
    )

    result = QwenRepairStage(repairer=repairer).run(
        _context(source_path, (_aligned_segment(),))
    )

    assert result.data == {"decisions": (decision,)}


def test_qwen_repair_stage_preserves_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = Subtitle(
        index=1,
        text="Nihongo desu.",
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
    )
    context = PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=source_path,
            segments=(_aligned_segment(),),
            subtitles=(subtitle,),
        ),
        working_directory=tmp_path / "work",
    )
    repairer = FakeRepairer(
        repair_result=QwenRepair(
            source_path=source_path,
            segments=(_repaired_segment(),),
        ),
        requests=[],
    )

    result = QwenRepairStage(repairer=repairer).run(context)

    assert result.context.document.subtitles == (subtitle,)


def test_qwen_repair_stage_accepts_custom_stage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    repairer = FakeRepairer(
        repair_result=QwenRepair(
            source_path=source_path,
            segments=(_repaired_segment(),),
        ),
        requests=[],
    )
    stage = QwenRepairStage(repairer=repairer, name="  qwen  ")

    result = stage.run(_context(source_path, (_aligned_segment(),)))

    assert stage.name == "qwen"
    assert result.stage_name == "qwen"


def test_qwen_repair_stage_rejects_missing_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    repairer = FakeRepairer(
        repair_result=QwenRepair(
            source_path=source_path,
            segments=(_repaired_segment(),),
        ),
        requests=[],
    )
    stage = QwenRepairStage(repairer=repairer)

    with pytest.raises(MissingAlignedSegmentsError):
        stage.run(_context(source_path, ()))

    assert repairer.requests == []


def test_qwen_repair_stage_rejects_invalid_repairer() -> None:
    with pytest.raises(InvalidQwenRepairerError):
        QwenRepairStage(repairer=object())


def test_qwen_repair_stage_rejects_invalid_repair_return(tmp_path: Path) -> None:
    stage = QwenRepairStage(repairer=InvalidRepairer())

    with pytest.raises(InvalidQwenRepairError, match="QwenRepair"):
        stage.run(_context(tmp_path / "input.wav", (_aligned_segment(),)))


def test_qwen_repair_stage_rejects_mismatched_source_path(tmp_path: Path) -> None:
    repairer = FakeRepairer(
        repair_result=QwenRepair(
            source_path=tmp_path / "other.wav",
            segments=(_repaired_segment(),),
        ),
        requests=[],
    )
    stage = QwenRepairStage(repairer=repairer)

    with pytest.raises(InvalidQwenRepairError, match="source path"):
        stage.run(_context(tmp_path / "input.wav", (_aligned_segment(),)))


def test_qwen_repair_requires_segments() -> None:
    with pytest.raises(ValueError, match="segments"):
        QwenRepair(source_path=Path("input.wav"), segments=())


def test_qwen_repair_is_immutable() -> None:
    repair = QwenRepair(
        source_path=Path("input.wav"),
        segments=(_repaired_segment(),),
    )

    with pytest.raises(FrozenInstanceError):
        repair.segments = ()

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
    InvalidReadabilityOptimizationError,
    InvalidReadabilityOptimizerError,
    MissingSubtitlesToOptimizeError,
    ReadabilityOptimization,
    ReadabilityOptimizationRequest,
    ReadabilityOptimizerStage,
    StageResult,
)


@dataclass(slots=True)
class FakeOptimizer:
    optimization: ReadabilityOptimization
    requests: list[ReadabilityOptimizationRequest]

    def optimize(
        self,
        request: ReadabilityOptimizationRequest,
    ) -> ReadabilityOptimization:
        self.requests.append(request)
        return self.optimization


@dataclass(frozen=True, slots=True)
class InvalidOptimizationOptimizer:
    def optimize(self, request: ReadabilityOptimizationRequest) -> object:
        return request


def _segment() -> Segment:
    return Segment(
        position=0,
        text="This is a very long subtitle line. It needs a calmer split.",
        time_range=TimeRange(start_seconds=0.0, end_seconds=4.0),
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


def test_readability_optimizer_stage_optimizes_existing_subtitles(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input.wav"
    segment = _segment()
    original = _subtitle(
        1,
        "This is a very long subtitle line. It needs a calmer split.",
        0.0,
        4.0,
    )
    optimized_first = _subtitle(1, "This is a very long subtitle line.", 0.0, 2.0)
    optimized_second = _subtitle(2, "It needs a calmer split.", 2.0, 4.0)
    optimizer = FakeOptimizer(
        optimization=ReadabilityOptimization(
            source_path=source_path,
            subtitles=(optimized_first, optimized_second),
        ),
        requests=[],
    )

    result = ReadabilityOptimizerStage(optimizer=optimizer).run(
        _context(source_path, (original,), segments=(segment,))
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "readability-optimizer"
    assert optimizer.requests == [
        ReadabilityOptimizationRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(segment,),
            subtitles=(original,),
        )
    ]
    assert result.context.document.source_path == source_path
    assert result.context.document.segments == (segment,)
    assert result.context.document.subtitles == (optimized_first, optimized_second)
    assert result.context.run_id == "run-001"
    assert result.context.working_directory == tmp_path / "work"


def test_readability_optimizer_stage_replaces_existing_subtitles(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input.wav"
    original = _subtitle(1, "Original subtitle.", 0.0, 1.5)
    optimized = _subtitle(1, "Optimized subtitle.", 0.0, 1.5)
    optimizer = FakeOptimizer(
        optimization=ReadabilityOptimization(
            source_path=source_path,
            subtitles=(optimized,),
        ),
        requests=[],
    )

    result = ReadabilityOptimizerStage(optimizer=optimizer).run(
        _context(source_path, (original,))
    )

    assert result.context.document.subtitles == (optimized,)


def test_readability_optimizer_stage_accepts_custom_stage_name(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = _subtitle(1, "Nihongo desu.", 0.0, 1.5)
    optimizer = FakeOptimizer(
        optimization=ReadabilityOptimization(
            source_path=source_path,
            subtitles=(subtitle,),
        ),
        requests=[],
    )
    stage = ReadabilityOptimizerStage(optimizer=optimizer, name="  readability  ")

    result = stage.run(_context(source_path, (subtitle,)))

    assert stage.name == "readability"
    assert result.stage_name == "readability"


def test_readability_optimizer_stage_rejects_missing_subtitles(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = _subtitle(1, "Nihongo desu.", 0.0, 1.5)
    optimizer = FakeOptimizer(
        optimization=ReadabilityOptimization(
            source_path=source_path,
            subtitles=(subtitle,),
        ),
        requests=[],
    )
    stage = ReadabilityOptimizerStage(optimizer=optimizer)

    with pytest.raises(MissingSubtitlesToOptimizeError):
        stage.run(_context(source_path, ()))

    assert optimizer.requests == []


def test_readability_optimizer_stage_rejects_invalid_optimizer() -> None:
    with pytest.raises(InvalidReadabilityOptimizerError):
        ReadabilityOptimizerStage(optimizer=object())


def test_readability_optimizer_stage_rejects_invalid_optimization_return(
    tmp_path: Path,
) -> None:
    stage = ReadabilityOptimizerStage(optimizer=InvalidOptimizationOptimizer())

    with pytest.raises(
        InvalidReadabilityOptimizationError,
        match="ReadabilityOptimization",
    ):
        stage.run(
            _context(
                tmp_path / "input.wav",
                (_subtitle(1, "Nihongo desu.", 0.0, 1.5),),
            )
        )


def test_readability_optimizer_stage_rejects_mismatched_source_path(
    tmp_path: Path,
) -> None:
    optimizer = FakeOptimizer(
        optimization=ReadabilityOptimization(
            source_path=tmp_path / "other.wav",
            subtitles=(_subtitle(1, "Nihongo desu.", 0.0, 1.5),),
        ),
        requests=[],
    )
    stage = ReadabilityOptimizerStage(optimizer=optimizer)

    with pytest.raises(InvalidReadabilityOptimizationError, match="source path"):
        stage.run(
            _context(
                tmp_path / "input.wav",
                (_subtitle(1, "Nihongo desu.", 0.0, 1.5),),
            )
        )


def test_readability_optimization_requires_subtitles() -> None:
    with pytest.raises(ValueError, match="subtitles"):
        ReadabilityOptimization(source_path=Path("input.wav"), subtitles=())


def test_readability_optimization_is_immutable() -> None:
    optimization = ReadabilityOptimization(
        source_path=Path("input.wav"),
        subtitles=(_subtitle(1, "Nihongo desu.", 0.0, 1.5),),
    )

    with pytest.raises(FrozenInstanceError):
        optimization.subtitles = ()

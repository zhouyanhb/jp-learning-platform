from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import Document, PipelineContext
from jp_learning_platform.workflow import (
    ExecutionEngine,
    Pipeline,
    StageResult,
    Workflow,
    create_pipeline,
)


@dataclass(frozen=True, slots=True)
class RecordingStage:
    name: str
    run_id_suffix: str
    calls: list[str]

    def run(self, context: PipelineContext) -> StageResult:
        self.calls.append(context.run_id)
        next_context = PipelineContext(
            run_id=f"{context.run_id}-{self.run_id_suffix}",
            document=context.document,
            working_directory=context.working_directory,
        )
        return StageResult(stage_name=self.name, context=next_context)


@dataclass(frozen=True, slots=True)
class InvalidResultStage:
    name: str

    def run(self, context: PipelineContext) -> object:
        return context


@dataclass(frozen=True, slots=True)
class MismatchedResultStage:
    name: str

    def run(self, context: PipelineContext) -> StageResult:
        return StageResult(stage_name="other", context=context)


@dataclass(slots=True)
class RecordingObserver:
    events: list[tuple[str, str, str, str]]

    def stage_started(self, event) -> None:
        self.events.append(
            ("started", event.workflow_name, event.stage_name, event.context.run_id)
        )

    def stage_succeeded(self, event) -> None:
        self.events.append(
            ("succeeded", event.workflow_name, event.stage_name, event.context.run_id)
        )

    def stage_failed(self, event) -> None:
        self.events.append(
            ("failed", event.workflow_name, event.stage_name, event.error_message)
        )


def _context(run_id: str = "run") -> PipelineContext:
    return PipelineContext(
        run_id=run_id,
        document=Document(source_path=Path("audio/input.wav")),
        working_directory=Path("work/run"),
    )


def test_execution_engine_runs_pipeline_stages_in_order() -> None:
    calls: list[str] = []
    first = RecordingStage(name="first", run_id_suffix="a", calls=calls)
    second = RecordingStage(name="second", run_id_suffix="b", calls=calls)
    workflow = Workflow(
        name="subtitle",
        pipeline=Pipeline(name="subtitle-pipeline", stages=(first, second)),
    )

    results = ExecutionEngine().execute(workflow, _context())

    assert calls == ["run", "run-a"]
    assert tuple(result.stage_name for result in results) == ("first", "second")
    assert results[-1].context.run_id == "run-a-b"


def test_execution_engine_notifies_observer_for_stage_progress() -> None:
    calls: list[str] = []
    observer = RecordingObserver(events=[])
    first = RecordingStage(name="first", run_id_suffix="a", calls=calls)
    second = RecordingStage(name="second", run_id_suffix="b", calls=calls)
    workflow = Workflow(
        name="subtitle",
        pipeline=Pipeline(name="subtitle-pipeline", stages=(first, second)),
    )

    ExecutionEngine().execute(workflow, _context(), observer=observer)

    assert observer.events == [
        ("started", "subtitle", "first", "run"),
        ("succeeded", "subtitle", "first", "run-a"),
        ("started", "subtitle", "second", "run-a"),
        ("succeeded", "subtitle", "second", "run-a-b"),
    ]


def test_create_pipeline_accepts_stage_iterables() -> None:
    calls: list[str] = []
    stage = RecordingStage(name="stage", run_id_suffix="done", calls=calls)

    pipeline = create_pipeline("subtitle-pipeline", [stage])

    assert pipeline.stages == (stage,)


def test_pipeline_rejects_empty_stage_collection() -> None:
    with pytest.raises(ValueError, match="stages"):
        Pipeline(name="empty", stages=())


def test_pipeline_rejects_invalid_stage_contract() -> None:
    with pytest.raises(TypeError, match="stage.name"):
        Pipeline(name="invalid", stages=(object(),))


def test_stage_result_requires_pipeline_context() -> None:
    with pytest.raises(TypeError, match="PipelineContext"):
        StageResult(stage_name="stage", context=object())


def test_execution_engine_rejects_invalid_stage_result() -> None:
    workflow = Workflow(
        name="subtitle",
        pipeline=Pipeline(
            name="subtitle-pipeline",
            stages=(InvalidResultStage("bad"),),
        ),
    )

    with pytest.raises(TypeError, match="StageResult"):
        ExecutionEngine().execute(workflow, _context())


def test_execution_engine_rejects_mismatched_stage_result_name() -> None:
    workflow = Workflow(
        name="subtitle",
        pipeline=Pipeline(
            name="subtitle-pipeline",
            stages=(MismatchedResultStage("actual"),),
        ),
    )

    with pytest.raises(ValueError, match="stage result name"):
        ExecutionEngine().execute(workflow, _context())


def test_execution_engine_propagates_stage_errors() -> None:
    class FailingStage:
        name = "failing"

        def run(self, context: PipelineContext) -> StageResult:
            raise RuntimeError("stage failed")

    workflow = Workflow(
        name="subtitle",
        pipeline=Pipeline(name="subtitle-pipeline", stages=(FailingStage(),)),
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        ExecutionEngine().execute(workflow, _context())


def test_execution_engine_notifies_observer_for_stage_failure() -> None:
    class FailingStage:
        name = "failing"

        def run(self, context: PipelineContext) -> StageResult:
            raise RuntimeError("stage failed")

    observer = RecordingObserver(events=[])
    workflow = Workflow(
        name="subtitle",
        pipeline=Pipeline(name="subtitle-pipeline", stages=(FailingStage(),)),
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        ExecutionEngine().execute(workflow, _context(), observer=observer)

    assert observer.events == [
        ("started", "subtitle", "failing", "run"),
        ("failed", "subtitle", "failing", "stage failed"),
    ]


def test_execution_engine_rejects_invalid_observer() -> None:
    calls: list[str] = []
    workflow = Workflow(
        name="subtitle",
        pipeline=Pipeline(
            name="subtitle-pipeline",
            stages=(RecordingStage(name="stage", run_id_suffix="done", calls=calls),),
        ),
    )

    with pytest.raises(TypeError, match="observer.stage_started"):
        ExecutionEngine().execute(workflow, _context(), observer=object())

"""Workflow runtime primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from jp_learning_platform.domain import PipelineContext


def _normalize_name(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


class Stage(Protocol):
    """Workflow stage contract."""

    name: str

    def run(self, context: PipelineContext) -> StageResult:
        """Run a stage and return the updated pipeline context."""


@dataclass(frozen=True, slots=True)
class StageExecutionEvent:
    """Execution event emitted while a workflow stage runs."""

    workflow_name: str
    pipeline_name: str
    stage_name: str
    context: PipelineContext
    elapsed_seconds: float | None = None
    error_message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if self.elapsed_seconds is not None:
            if isinstance(self.elapsed_seconds, bool):
                raise TypeError("elapsed_seconds must be a number.")

            elapsed_seconds = float(self.elapsed_seconds)
            if elapsed_seconds < 0:
                raise ValueError("elapsed_seconds must be non-negative.")
            object.__setattr__(self, "elapsed_seconds", elapsed_seconds)

        if not isinstance(self.error_message, str):
            raise TypeError("error_message must be a string.")

        object.__setattr__(
            self,
            "workflow_name",
            _normalize_name(self.workflow_name, "workflow_name"),
        )
        object.__setattr__(
            self,
            "pipeline_name",
            _normalize_name(self.pipeline_name, "pipeline_name"),
        )
        object.__setattr__(
            self,
            "stage_name",
            _normalize_name(self.stage_name, "stage_name"),
        )


class StageExecutionObserver(Protocol):
    """Observer for workflow stage execution events."""

    def stage_started(self, event: StageExecutionEvent) -> None:
        """Handle a stage start event."""

    def stage_succeeded(self, event: StageExecutionEvent) -> None:
        """Handle a stage success event."""

    def stage_failed(self, event: StageExecutionEvent) -> None:
        """Handle a stage failure event."""


@dataclass(frozen=True, slots=True)
class StageResult:
    stage_name: str
    context: PipelineContext

    def __post_init__(self) -> None:
        if not isinstance(self.context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        object.__setattr__(
            self,
            "stage_name",
            _normalize_name(self.stage_name, "stage_name"),
        )


@dataclass(frozen=True, slots=True)
class Pipeline:
    name: str
    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("stages must not be empty.")

        for stage in stages:
            _validate_stage(stage)

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))
        object.__setattr__(self, "stages", stages)


@dataclass(frozen=True, slots=True)
class Workflow:
    name: str
    pipeline: Pipeline

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, Pipeline):
            raise TypeError("pipeline must be a Pipeline.")

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))


@dataclass(frozen=True, slots=True)
class ExecutionEngine:
    """Execute workflow stages in order."""

    def execute(
        self,
        workflow: Workflow,
        context: PipelineContext,
        observer: StageExecutionObserver | None = None,
    ) -> tuple[StageResult, ...]:
        if not isinstance(workflow, Workflow):
            raise TypeError("workflow must be a Workflow.")

        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if observer is not None:
            _validate_observer(observer)

        results: list[StageResult] = []
        current_context = context

        for stage in workflow.pipeline.stages:
            stage_name = _stage_name(stage)
            if observer is not None:
                observer.stage_started(
                    StageExecutionEvent(
                        workflow_name=workflow.name,
                        pipeline_name=workflow.pipeline.name,
                        stage_name=stage_name,
                        context=current_context,
                    )
                )

            started_at = monotonic()
            try:
                result = stage.run(current_context)
                if not isinstance(result, StageResult):
                    raise TypeError("stage run must return a StageResult.")

                if result.stage_name != stage_name:
                    raise ValueError("stage result name must match the stage name.")
            except Exception as error:
                if observer is not None:
                    observer.stage_failed(
                        StageExecutionEvent(
                            workflow_name=workflow.name,
                            pipeline_name=workflow.pipeline.name,
                            stage_name=stage_name,
                            context=current_context,
                            elapsed_seconds=monotonic() - started_at,
                            error_message=str(error),
                        )
                    )
                raise

            current_context = result.context
            results.append(result)
            if observer is not None:
                observer.stage_succeeded(
                    StageExecutionEvent(
                        workflow_name=workflow.name,
                        pipeline_name=workflow.pipeline.name,
                        stage_name=stage_name,
                        context=current_context,
                        elapsed_seconds=monotonic() - started_at,
                    )
                )

        return tuple(results)


def _validate_stage(stage: Stage) -> None:
    _stage_name(stage)

    if not callable(getattr(stage, "run", None)):
        raise TypeError("stage.run must be callable.")


def _stage_name(stage: Stage) -> str:
    return _normalize_name(getattr(stage, "name", None), "stage.name")


def _validate_observer(observer: StageExecutionObserver) -> None:
    for method_name in ("stage_started", "stage_succeeded", "stage_failed"):
        if not callable(getattr(observer, method_name, None)):
            raise TypeError(f"observer.{method_name} must be callable.")


def create_pipeline(name: str, stages: Iterable[Stage]) -> Pipeline:
    return Pipeline(name=name, stages=tuple(stages))


__all__ = [
    "ExecutionEngine",
    "Pipeline",
    "Stage",
    "StageExecutionEvent",
    "StageExecutionObserver",
    "StageResult",
    "Workflow",
    "create_pipeline",
]

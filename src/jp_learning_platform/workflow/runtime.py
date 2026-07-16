"""Workflow runtime primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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
    ) -> tuple[StageResult, ...]:
        if not isinstance(workflow, Workflow):
            raise TypeError("workflow must be a Workflow.")

        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        results: list[StageResult] = []
        current_context = context

        for stage in workflow.pipeline.stages:
            result = stage.run(current_context)
            if not isinstance(result, StageResult):
                raise TypeError("stage run must return a StageResult.")

            stage_name = _stage_name(stage)
            if result.stage_name != stage_name:
                raise ValueError("stage result name must match the stage name.")

            current_context = result.context
            results.append(result)

        return tuple(results)


def _validate_stage(stage: Stage) -> None:
    _stage_name(stage)

    if not callable(getattr(stage, "run", None)):
        raise TypeError("stage.run must be callable.")


def _stage_name(stage: Stage) -> str:
    return _normalize_name(getattr(stage, "name", None), "stage.name")


def create_pipeline(name: str, stages: Iterable[Stage]) -> Pipeline:
    return Pipeline(name=name, stages=tuple(stages))


__all__ = [
    "ExecutionEngine",
    "Pipeline",
    "Stage",
    "StageResult",
    "Workflow",
    "create_pipeline",
]

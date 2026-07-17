"""Workflow progress and artifact recording contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from jp_learning_platform.domain import PipelineContext


class PipelineProgressStatus(Enum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _normalize_stage_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("stage_name must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError("stage_name must not be empty.")

    return normalized


def _normalize_position(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value < 1:
        raise ValueError(f"{field_name} must be at least 1.")

    return value


def _normalize_elapsed_seconds(value: float | None) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise TypeError("elapsed_seconds must be a number.")

    elapsed_seconds = float(value)
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative.")

    return elapsed_seconds


@dataclass(frozen=True, slots=True)
class PipelineProgressEvent:
    """Progress event for one file and one pipeline stage."""

    source_path: Path
    output_path: Path
    file_index: int
    file_total: int
    stage_name: str
    status: PipelineProgressStatus
    elapsed_seconds: float | None = None
    artifact_path: Path | None = None
    message: str = ""

    def __post_init__(self) -> None:
        file_index = _normalize_position(self.file_index, "file_index")
        file_total = _normalize_position(self.file_total, "file_total")
        if file_index > file_total:
            raise ValueError("file_index must not exceed file_total.")

        if not isinstance(self.status, PipelineProgressStatus):
            raise TypeError("status must be a PipelineProgressStatus.")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")

        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "file_index", file_index)
        object.__setattr__(self, "file_total", file_total)
        object.__setattr__(self, "stage_name", _normalize_stage_name(self.stage_name))
        object.__setattr__(
            self,
            "elapsed_seconds",
            _normalize_elapsed_seconds(self.elapsed_seconds),
        )
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))


@dataclass(frozen=True, slots=True)
class StageArtifactRecord:
    """Serializable record for one file and one pipeline stage."""

    source_path: Path
    output_path: Path
    file_index: int
    file_total: int
    stage_name: str
    status: PipelineProgressStatus
    context: PipelineContext
    elapsed_seconds: float | None = None
    data: object | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        progress_event = PipelineProgressEvent(
            source_path=self.source_path,
            output_path=self.output_path,
            file_index=self.file_index,
            file_total=self.file_total,
            stage_name=self.stage_name,
            status=self.status,
            elapsed_seconds=self.elapsed_seconds,
            message=self.message,
        )

        object.__setattr__(self, "source_path", progress_event.source_path)
        object.__setattr__(self, "output_path", progress_event.output_path)
        object.__setattr__(self, "file_index", progress_event.file_index)
        object.__setattr__(self, "file_total", progress_event.file_total)
        object.__setattr__(self, "stage_name", progress_event.stage_name)
        object.__setattr__(self, "elapsed_seconds", progress_event.elapsed_seconds)


class ProgressReporter(Protocol):
    """Report pipeline progress events to a user-facing sink."""

    def report(self, event: PipelineProgressEvent) -> None:
        """Report one pipeline progress event."""


class StageArtifactRecorder(Protocol):
    """Persist pipeline stage artifacts."""

    def audio_directory(self, source_path: Path) -> Path:
        """Return the artifact directory for one audio source."""

    def record(self, record: StageArtifactRecord) -> Path:
        """Persist one stage artifact and return its path."""


@dataclass(frozen=True, slots=True)
class NoOpProgressReporter:
    """Progress reporter that intentionally emits no output."""

    def report(self, event: PipelineProgressEvent) -> None:
        if not isinstance(event, PipelineProgressEvent):
            raise TypeError("event must be a PipelineProgressEvent.")


__all__ = [
    "NoOpProgressReporter",
    "PipelineProgressEvent",
    "PipelineProgressStatus",
    "ProgressReporter",
    "StageArtifactRecord",
    "StageArtifactRecorder",
]

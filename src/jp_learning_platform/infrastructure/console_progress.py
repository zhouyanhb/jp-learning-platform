"""Console progress reporter for local subtitle pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import TextIO

from jp_learning_platform.workflow.progress import (
    PipelineProgressEvent,
    PipelineProgressStatus,
)


@dataclass(frozen=True, slots=True)
class ConsoleProgressReporter:
    """Write one-line pipeline progress updates to a text stream."""

    output: TextIO = sys.stderr

    def report(self, event: PipelineProgressEvent) -> None:
        if not isinstance(event, PipelineProgressEvent):
            raise TypeError("event must be a PipelineProgressEvent.")

        self.output.write(f"{_format_event(event)}\n")
        self.output.flush()


def _format_event(event: PipelineProgressEvent) -> str:
    prefix = (
        f"[{event.file_index}/{event.file_total}] "
        f"{event.source_path.name} {event.stage_name}"
    )

    if event.status is PipelineProgressStatus.STARTED:
        return f"{prefix} started"

    elapsed = ""
    if event.elapsed_seconds is not None:
        elapsed = f" {event.elapsed_seconds:.2f}s"

    if event.status is PipelineProgressStatus.SUCCEEDED:
        return _append_artifact_path(f"{prefix} done{elapsed}", event)

    message = f": {event.message}" if event.message else ""
    return _append_artifact_path(f"{prefix} failed{elapsed}{message}", event)


def _append_artifact_path(message: str, event: PipelineProgressEvent) -> str:
    if event.artifact_path is None:
        return message

    return f"{message} -> {event.artifact_path}"


__all__ = [
    "ConsoleProgressReporter",
]

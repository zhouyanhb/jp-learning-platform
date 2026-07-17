"""Subtitle pipeline runner for local audio inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jp_learning_platform.application import (
    AudioInputDiscovery,
    SubtitlePipelineItemResult,
    SubtitlePipelineRequest,
    SubtitlePipelineResult,
)
from jp_learning_platform.domain import Document, PipelineContext
from jp_learning_platform.workflow.runtime import ExecutionEngine, Workflow, create_pipeline
from jp_learning_platform.workflow.subtitle_builder_stage import (
    SubtitleBuilder,
    SubtitleBuilderStage,
)
from jp_learning_platform.workflow.subtitle_writer_stage import (
    SubtitleWriter,
    SubtitleWriterStage,
)
from jp_learning_platform.workflow.whisper_stage import WhisperStage, WhisperTranscriber


class AudioLoader(Protocol):
    """Audio loader contract required by the pipeline runner."""

    def load(self, source_path: Path) -> object:
        """Validate and load a local audio source."""


class SubtitlePipelineRunnerError(RuntimeError):
    """Base error for subtitle pipeline runner failures."""


class DuplicateSubtitleOutputError(SubtitlePipelineRunnerError):
    """Raised when multiple audio inputs would write the same subtitle path."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        super().__init__(f"Duplicate subtitle output path: {output_path}")


@dataclass(frozen=True, slots=True)
class SubtitlePipelineRunner:
    """Run the local audio-to-SRT subtitle pipeline."""

    audio_loader: AudioLoader
    transcriber: WhisperTranscriber
    builder: SubtitleBuilder
    writer: SubtitleWriter
    discovery: AudioInputDiscovery = AudioInputDiscovery()
    engine: ExecutionEngine = ExecutionEngine()

    def run(self, request: SubtitlePipelineRequest) -> SubtitlePipelineResult:
        if not isinstance(request, SubtitlePipelineRequest):
            raise TypeError("request must be a SubtitlePipelineRequest.")

        audio_paths = self.discovery.discover(request.input_path)
        output_paths = tuple(
            self.output_path_for(source_path, request.output_directory)
            for source_path in audio_paths
        )
        self._ensure_unique_outputs(output_paths)

        request.output_directory.mkdir(parents=True, exist_ok=True)

        items: list[SubtitlePipelineItemResult] = []
        for source_path, output_path in zip(audio_paths, output_paths, strict=True):
            self.audio_loader.load(source_path)
            context = PipelineContext(
                run_id=f"transcribe-{source_path.stem}",
                document=Document(source_path=source_path),
                working_directory=request.output_directory / ".work" / source_path.stem,
            )
            workflow = Workflow(
                name="audio-to-srt",
                pipeline=create_pipeline(
                    "audio-to-srt",
                    (
                        WhisperStage(self.transcriber),
                        SubtitleBuilderStage(self.builder),
                        SubtitleWriterStage(self.writer),
                    ),
                ),
            )
            self.engine.execute(workflow, context)
            items.append(
                SubtitlePipelineItemResult(
                    source_path=source_path,
                    output_path=output_path,
                )
            )

        return SubtitlePipelineResult(items=tuple(items))

    def output_path_for(self, source_path: Path, output_directory: Path) -> Path:
        return Path(output_directory) / f"{Path(source_path).stem}.srt"

    def _ensure_unique_outputs(self, output_paths: tuple[Path, ...]) -> None:
        seen: set[Path] = set()
        for output_path in output_paths:
            if output_path in seen:
                raise DuplicateSubtitleOutputError(output_path)
            seen.add(output_path)


__all__ = [
    "AudioLoader",
    "DuplicateSubtitleOutputError",
    "SubtitlePipelineRunner",
    "SubtitlePipelineRunnerError",
]

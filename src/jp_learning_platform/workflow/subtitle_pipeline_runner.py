"""Subtitle pipeline runner for local audio inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol

from jp_learning_platform.application import (
    AudioInputDiscovery,
    SubtitlePipelineItemResult,
    SubtitlePipelineRequest,
    SubtitlePipelineResult,
)
from jp_learning_platform.domain import Document, PipelineContext
from jp_learning_platform.workflow.progress import (
    NoOpProgressReporter,
    PipelineProgressEvent,
    PipelineProgressStatus,
    ProgressReporter,
    StageArtifactRecord,
    StageArtifactRecorder,
)
from jp_learning_platform.workflow.qwen_repair_stage import QwenRepairStage, QwenRepairer
from jp_learning_platform.workflow.homophone_stage import (
    HomophoneResolutionStage,
    HomophoneResolver,
)
from jp_learning_platform.workflow.japanese_word_stage import (
    JapaneseWordNormalizationStage,
    JapaneseWordNormalizer,
)
from jp_learning_platform.workflow.readability_optimizer_stage import (
    ReadabilityOptimizer,
    ReadabilityOptimizerStage,
)
from jp_learning_platform.workflow.runtime import (
    ExecutionEngine,
    Stage,
    StageExecutionEvent,
    Workflow,
    create_pipeline,
)
from jp_learning_platform.workflow.sentence_boundary_stage import (
    SentenceBoundaryDetectionStage,
    SentenceBoundaryDetector,
    SentenceBoundaryResolver,
    SentenceBoundaryResolverStage,
)
from jp_learning_platform.workflow.subtitle_builder_stage import (
    SubtitleBuilder,
    SubtitleBuilderStage,
)
from jp_learning_platform.workflow.subtitle_merger_stage import (
    SubtitleMerger,
    SubtitleMergerStage,
)
from jp_learning_platform.workflow.subtitle_validator_stage import (
    SubtitleValidator,
    SubtitleValidatorStage,
)
from jp_learning_platform.workflow.subtitle_writer_stage import (
    SubtitleWriter,
    SubtitleWriterStage,
)
from jp_learning_platform.workflow.whisper_stage import WhisperStage, WhisperTranscriber
from jp_learning_platform.workflow.whisperx_alignment_stage import (
    WhisperXAligner,
    WhisperXAlignmentStage,
)

DEFAULT_SUBTITLE_OUTPUT_EXTENSION = ".srt"


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


def _normalize_output_extension(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("output_extension must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError("output_extension must not be empty.")

    if not normalized.startswith("."):
        raise ValueError("output_extension must start with a dot.")

    if "/" in normalized or "\\" in normalized:
        raise ValueError("output_extension must not contain path separators.")

    return normalized


@dataclass(frozen=True, slots=True)
class _PipelineRunProgress:
    source_path: Path
    output_path: Path
    file_index: int
    file_total: int
    reporter: ProgressReporter
    artifact_recorder: StageArtifactRecorder | None = None

    def emit(
        self,
        stage_name: str,
        status: PipelineProgressStatus,
        context: PipelineContext,
        elapsed_seconds: float | None = None,
        data: object | None = None,
        message: str = "",
    ) -> None:
        artifact_path = None
        if self.artifact_recorder is not None:
            artifact_path = self.artifact_recorder.record(
                StageArtifactRecord(
                    source_path=self.source_path,
                    output_path=self.output_path,
                    file_index=self.file_index,
                    file_total=self.file_total,
                    stage_name=stage_name,
                    status=status,
                    context=context,
                    elapsed_seconds=elapsed_seconds,
                    data=data,
                    message=message,
                )
            )

        self.reporter.report(
            PipelineProgressEvent(
                source_path=self.source_path,
                output_path=self.output_path,
                file_index=self.file_index,
                file_total=self.file_total,
                stage_name=stage_name,
                status=status,
                elapsed_seconds=elapsed_seconds,
                artifact_path=artifact_path,
                message=message,
            )
        )

    def stage_started(self, event: StageExecutionEvent) -> None:
        self.emit(
            stage_name=event.stage_name,
            status=PipelineProgressStatus.STARTED,
            context=event.context,
        )

    def stage_succeeded(self, event: StageExecutionEvent) -> None:
        self.emit(
            stage_name=event.stage_name,
            status=PipelineProgressStatus.SUCCEEDED,
            context=event.context,
            elapsed_seconds=event.elapsed_seconds,
            data=event.data,
        )

    def stage_failed(self, event: StageExecutionEvent) -> None:
        self.emit(
            stage_name=event.stage_name,
            status=PipelineProgressStatus.FAILED,
            context=event.context,
            elapsed_seconds=event.elapsed_seconds,
            message=event.error_message,
        )


@dataclass(frozen=True, slots=True)
class SubtitlePipelineRunner:
    """Run the local audio subtitle output pipeline."""

    audio_loader: AudioLoader
    transcriber: WhisperTranscriber
    builder: SubtitleBuilder
    writer: SubtitleWriter
    aligner: WhisperXAligner | None = None
    word_normalizer: JapaneseWordNormalizer | None = None
    sentence_boundary_detector: SentenceBoundaryDetector | None = None
    repairer: QwenRepairer | None = None
    homophone_resolver: HomophoneResolver | None = None
    sentence_boundary_resolver: SentenceBoundaryResolver | None = None
    merger: SubtitleMerger | None = None
    optimizer: ReadabilityOptimizer | None = None
    validator: SubtitleValidator | None = None
    discovery: AudioInputDiscovery = AudioInputDiscovery()
    engine: ExecutionEngine = ExecutionEngine()
    progress_reporter: ProgressReporter = NoOpProgressReporter()
    artifact_recorder: StageArtifactRecorder | None = None
    output_extension: str = DEFAULT_SUBTITLE_OUTPUT_EXTENSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_extension",
            _normalize_output_extension(self.output_extension),
        )

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
        file_total = len(audio_paths)
        for file_index, (source_path, output_path) in enumerate(
            zip(audio_paths, output_paths, strict=True),
            start=1,
        ):
            context = PipelineContext(
                run_id=f"transcribe-{source_path.stem}",
                document=Document(source_path=source_path),
                working_directory=self._working_directory_for(
                    source_path,
                    request.output_directory,
                ),
            )
            progress = _PipelineRunProgress(
                source_path=source_path,
                output_path=output_path,
                file_index=file_index,
                file_total=file_total,
                reporter=self.progress_reporter,
                artifact_recorder=self.artifact_recorder,
            )
            self._load_audio(source_path, context, progress)
            workflow = Workflow(
                name="audio-to-subtitle-output",
                pipeline=create_pipeline("audio-to-subtitle-output", self._stages()),
            )
            self.engine.execute(workflow, context, observer=progress)
            items.append(
                SubtitlePipelineItemResult(
                    source_path=source_path,
                    output_path=output_path,
                )
            )

        return SubtitlePipelineResult(items=tuple(items))

    def output_path_for(self, source_path: Path, output_directory: Path) -> Path:
        return Path(output_directory) / (
            f"{Path(source_path).stem}{self.output_extension}"
        )

    def _stages(self) -> tuple[Stage, ...]:
        stages: list[Stage] = [WhisperStage(self.transcriber)]

        if self.aligner is not None:
            stages.append(WhisperXAlignmentStage(self.aligner))

        if self.sentence_boundary_detector is not None:
            stages.append(
                SentenceBoundaryDetectionStage(self.sentence_boundary_detector)
            )

        if self.repairer is not None:
            stages.append(QwenRepairStage(self.repairer))

        if self.word_normalizer is not None:
            stages.append(JapaneseWordNormalizationStage(self.word_normalizer))

        if self.homophone_resolver is not None:
            stages.append(HomophoneResolutionStage(self.homophone_resolver))

        if self.sentence_boundary_resolver is not None:
            stages.append(
                SentenceBoundaryResolverStage(self.sentence_boundary_resolver)
            )

        stages.append(SubtitleBuilderStage(self.builder))

        if self.merger is not None:
            stages.append(SubtitleMergerStage(self.merger))

        if self.optimizer is not None:
            stages.append(ReadabilityOptimizerStage(self.optimizer))

        if self.validator is not None:
            stages.append(SubtitleValidatorStage(self.validator))

        stages.append(SubtitleWriterStage(self.writer))
        return tuple(stages)

    def _ensure_unique_outputs(self, output_paths: tuple[Path, ...]) -> None:
        seen: set[Path] = set()
        for output_path in output_paths:
            if output_path in seen:
                raise DuplicateSubtitleOutputError(output_path)
            seen.add(output_path)

    def _working_directory_for(
        self,
        source_path: Path,
        output_directory: Path,
    ) -> Path:
        if self.artifact_recorder is not None:
            return self.artifact_recorder.audio_directory(source_path)

        return output_directory / ".work" / source_path.stem

    def _load_audio(
        self,
        source_path: Path,
        context: PipelineContext,
        progress: _PipelineRunProgress,
    ) -> None:
        stage_name = "audio-loader"
        progress.emit(
            stage_name=stage_name,
            status=PipelineProgressStatus.STARTED,
            context=context,
        )
        started_at = monotonic()
        try:
            loaded_audio = self.audio_loader.load(source_path)
        except Exception as error:
            progress.emit(
                stage_name=stage_name,
                status=PipelineProgressStatus.FAILED,
                context=context,
                elapsed_seconds=monotonic() - started_at,
                message=str(error),
            )
            raise

        progress.emit(
            stage_name=stage_name,
            status=PipelineProgressStatus.SUCCEEDED,
            context=context,
            elapsed_seconds=monotonic() - started_at,
            data=loaded_audio,
        )


__all__ = [
    "AudioLoader",
    "DEFAULT_SUBTITLE_OUTPUT_EXTENSION",
    "DuplicateSubtitleOutputError",
    "SubtitlePipelineRunner",
    "SubtitlePipelineRunnerError",
]

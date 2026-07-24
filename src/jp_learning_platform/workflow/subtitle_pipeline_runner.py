"""Subtitle pipeline runner for local audio inputs."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import inspect
import json
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
    StagePhaseTiming,
)
from jp_learning_platform.workflow.homophone_stage import (
    HomophoneResolutionStage,
    HomophoneResolver,
)
from jp_learning_platform.workflow.qwen_repair_stage import QwenRepairStage, QwenRepairer
from jp_learning_platform.workflow.readability_optimizer_stage import (
    ReadabilityOptimizer,
    ReadabilityOptimizerStage,
)
from jp_learning_platform.workflow.sentence_boundary_stage import (
    SentenceBoundaryResolutionStage,
    SentenceBoundaryResolver,
)
from jp_learning_platform.workflow.word_normalization_stage import (
    WordNormalizationStage,
    WordNormalizer,
)
from jp_learning_platform.workflow.runtime import (
    ExecutionEngine,
    Stage,
    StageExecutionEvent,
    Workflow,
    create_pipeline,
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
DEFAULT_AUDIO_HASH_CHUNK_BYTES = 1024 * 1024


class AudioLoader(Protocol):
    """Audio loader contract required by the pipeline runner."""

    def load(self, source_path: Path) -> object:
        """Validate and load a local audio source."""


class PipelineContextCache(Protocol):
    """Content-addressed cache required by the runner."""

    def audio_digest(self, source_path: Path) -> str: ...

    def load_context(
        self, audio_digest: str, stage_fingerprint: str
    ) -> PipelineContext | None: ...

    def save_context(
        self,
        audio_digest: str,
        stage_fingerprint: str,
        context: PipelineContext,
    ) -> Path: ...

    def work_lock(self, audio_digest: str, pipeline_fingerprint: str) -> object: ...

    def normalized_audio_directory(self, audio_digest: str) -> Path: ...


class AudioNormalizer(Protocol):
    """Normalize uploaded media for audio-model stages."""

    def normalize(
        self,
        source_path: Path,
        cache_directory: Path,
        audio_digest: str,
    ) -> Path: ...


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
        if isinstance(event.data, tuple) and all(
            isinstance(item, StagePhaseTiming) for item in event.data
        ):
            for phase in event.data:
                self.emit(
                    stage_name=phase.phase_name,
                    status=PipelineProgressStatus.SUCCEEDED,
                    context=event.context,
                    elapsed_seconds=phase.elapsed_seconds,
                    data=phase,
                    message=f"component-of:{event.stage_name}",
                )
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

    def total_finished(
        self,
        *,
        elapsed_seconds: float,
        succeeded: bool,
        message: str = "",
    ) -> None:
        """Report end-to-end time without creating a stage artifact."""
        self.reporter.report(
            PipelineProgressEvent(
                source_path=self.source_path,
                output_path=self.output_path,
                file_index=self.file_index,
                file_total=self.file_total,
                stage_name="pipeline-total",
                status=(
                    PipelineProgressStatus.SUCCEEDED
                    if succeeded
                    else PipelineProgressStatus.FAILED
                ),
                elapsed_seconds=elapsed_seconds,
                message=message,
            )
        )


@dataclass(frozen=True, slots=True)
class SubtitlePipelineRunner:
    """Run the local audio subtitle output pipeline."""

    audio_loader: AudioLoader
    transcriber: WhisperTranscriber
    builder: SubtitleBuilder
    writer: SubtitleWriter
    aligner: WhisperXAligner | None = None
    repairer: QwenRepairer | None = None
    homophone_resolver: HomophoneResolver | None = None
    word_normalizer: WordNormalizer | None = None
    sentence_boundary_resolver: SentenceBoundaryResolver | None = None
    merger: SubtitleMerger | None = None
    optimizer: ReadabilityOptimizer | None = None
    validator: SubtitleValidator | None = None
    discovery: AudioInputDiscovery = AudioInputDiscovery()
    engine: ExecutionEngine = ExecutionEngine()
    progress_reporter: ProgressReporter = NoOpProgressReporter()
    artifact_recorder: StageArtifactRecorder | None = None
    cache: PipelineContextCache | None = None
    audio_normalizer: AudioNormalizer | None = None
    cache_namespace: str = "subtitle-pipeline-v1"
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
            file_started_at = monotonic()
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
            try:
                if self.cache is None:
                    self._execute_uncached(
                        source_path=source_path,
                        context=context,
                        progress=progress,
                    )
                else:
                    self._execute_cached(
                        source_path=source_path,
                        context=context,
                        progress=progress,
                    )
            except Exception as error:
                progress.total_finished(
                    elapsed_seconds=monotonic() - file_started_at,
                    succeeded=False,
                    message=str(error),
                )
                raise

            progress.total_finished(
                elapsed_seconds=monotonic() - file_started_at,
                succeeded=True,
            )
            items.append(
                SubtitlePipelineItemResult(
                    source_path=source_path,
                    output_path=output_path,
                )
            )

        return SubtitlePipelineResult(items=tuple(items))

    def _execute_uncached(
        self,
        *,
        source_path: Path,
        context: PipelineContext,
        progress: _PipelineRunProgress,
    ) -> None:
        self._load_audio(source_path, context, progress)
        stages = self._stages()
        processing_stages = stages[:-1]
        writer_stage = stages[-1]
        normalized = False
        for stage in processing_stages:
            if (
                not normalized
                and self.audio_normalizer is not None
                and _stage_requires_normalized_audio(stage)
            ):
                context = _rebind_context(
                    context,
                    source_path=self._normalize_audio(
                        source_path=source_path,
                        audio_digest=None,
                        cache_directory=context.working_directory / "audio",
                        context=context,
                        progress=progress,
                    ),
                    working_directory=context.working_directory,
                    run_id=context.run_id,
                )
                normalized = True
            context = self._execute_stages((stage,), context, progress)

        context = _rebind_context(
            context,
            source_path=source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
        )
        self._execute_stages((writer_stage,), context, progress)

    def _execute_cached(
        self,
        *,
        source_path: Path,
        context: PipelineContext,
        progress: _PipelineRunProgress,
    ) -> None:
        assert self.cache is not None
        cache_stage_name = "pipeline-cache"
        progress.emit(
            stage_name=cache_stage_name,
            status=PipelineProgressStatus.STARTED,
            context=context,
        )
        cache_started_at = monotonic()
        audio_digest = self.cache.audio_digest(source_path)
        stages = self._stages()
        processing_stages = stages[:-1]
        writer_stage = stages[-1]
        fingerprints = _cumulative_stage_fingerprints(
            self.cache_namespace,
            processing_stages,
            audio_normalizer=self.audio_normalizer,
        )
        pipeline_fingerprint = fingerprints[-1]

        with self.cache.work_lock(audio_digest, pipeline_fingerprint):
            cached_index, cached_context = self._longest_cached_context(
                audio_digest,
                fingerprints,
            )
            cache_message = (
                "complete-result-hit"
                if cached_index == len(processing_stages) - 1
                else "stage-prefix-hit"
                if cached_context is not None
                else "miss"
            )
            progress.emit(
                stage_name=cache_stage_name,
                status=PipelineProgressStatus.SUCCEEDED,
                context=cached_context or context,
                elapsed_seconds=monotonic() - cache_started_at,
                message=cache_message,
            )
            if cached_context is not None:
                context = _rebind_context(
                    cached_context,
                    source_path=source_path,
                    working_directory=context.working_directory,
                    run_id=context.run_id,
                )

            next_stage_index = cached_index + 1
            remaining_stages = processing_stages[next_stage_index:]
            needs_audio = any(
                isinstance(stage, WhisperStage | WhisperXAlignmentStage)
                for stage in remaining_stages
            )
            if needs_audio:
                self._load_audio(source_path, context, progress)

            normalized = False
            for stage_index, stage in enumerate(
                remaining_stages,
                start=next_stage_index,
            ):
                if (
                    not normalized
                    and self.audio_normalizer is not None
                    and _stage_requires_normalized_audio(stage)
                ):
                    processing_path = self._normalize_audio(
                        source_path=source_path,
                        audio_digest=audio_digest,
                        cache_directory=self.cache.normalized_audio_directory(
                            audio_digest
                        ),
                        context=context,
                        progress=progress,
                    )
                    context = _rebind_context(
                        context,
                        source_path=processing_path,
                        working_directory=context.working_directory,
                        run_id=context.run_id,
                    )
                    normalized = True
                context = self._execute_stages((stage,), context, progress)
                self.cache.save_context(
                    audio_digest,
                    fingerprints[stage_index],
                    context,
                )

            context = _rebind_context(
                context,
                source_path=source_path,
                working_directory=context.working_directory,
                run_id=context.run_id,
            )
            self._execute_stages((writer_stage,), context, progress)

    def _normalize_audio(
        self,
        *,
        source_path: Path,
        audio_digest: str | None,
        cache_directory: Path,
        context: PipelineContext,
        progress: _PipelineRunProgress,
    ) -> Path:
        assert self.audio_normalizer is not None
        stage_name = "audio-normalization"
        progress.emit(
            stage_name=stage_name,
            status=PipelineProgressStatus.STARTED,
            context=context,
        )
        started_at = monotonic()
        try:
            digest = audio_digest or _audio_digest(source_path)
            normalized_path = self.audio_normalizer.normalize(
                source_path,
                cache_directory,
                digest,
            )
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
            message=(
                "source-compatible"
                if normalized_path == source_path
                else "normalized-cache-ready"
            ),
        )
        return normalized_path

    def _longest_cached_context(
        self,
        audio_digest: str,
        fingerprints: tuple[str, ...],
    ) -> tuple[int, PipelineContext | None]:
        assert self.cache is not None
        for index in range(len(fingerprints) - 1, -1, -1):
            context = self.cache.load_context(audio_digest, fingerprints[index])
            if context is not None:
                return index, context
        return -1, None

    def _execute_stages(
        self,
        stages: tuple[Stage, ...],
        context: PipelineContext,
        progress: _PipelineRunProgress,
    ) -> PipelineContext:
        workflow = Workflow(
            name="audio-to-subtitle-output",
            pipeline=create_pipeline("audio-to-subtitle-output", stages),
        )
        results = self.engine.execute(workflow, context, observer=progress)
        return results[-1].context

    def output_path_for(self, source_path: Path, output_directory: Path) -> Path:
        return Path(output_directory) / (
            f"{Path(source_path).stem}{self.output_extension}"
        )

    def _stages(self) -> tuple[Stage, ...]:
        stages: list[Stage] = [WhisperStage(self.transcriber)]

        if self.aligner is not None:
            stages.append(WhisperXAlignmentStage(self.aligner))

        if self.repairer is not None:
            stages.append(QwenRepairStage(self.repairer))

        if self.homophone_resolver is not None:
            stages.append(HomophoneResolutionStage(self.homophone_resolver))

        if self.word_normalizer is not None:
            stages.append(WordNormalizationStage(self.word_normalizer))

        if self.sentence_boundary_resolver is not None:
            stages.append(
                SentenceBoundaryResolutionStage(self.sentence_boundary_resolver)
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
    "AudioNormalizer",
    "AudioLoader",
    "DEFAULT_SUBTITLE_OUTPUT_EXTENSION",
    "DuplicateSubtitleOutputError",
    "PipelineContextCache",
    "SubtitlePipelineRunner",
    "SubtitlePipelineRunnerError",
]


def _cumulative_stage_fingerprints(
    namespace: str,
    stages: tuple[Stage, ...],
    *,
    audio_normalizer: AudioNormalizer | None,
) -> tuple[str, ...]:
    cumulative: list[str] = []
    previous = sha256(namespace.encode("utf-8")).hexdigest()
    for stage in stages:
        stage_configuration = _stable_configuration(stage)
        if _stage_requires_normalized_audio(stage):
            stage_configuration = {
                "stage": stage_configuration,
                "audio_normalizer": _stable_configuration(audio_normalizer),
            }
        payload = json.dumps(
            stage_configuration,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        previous = sha256(f"{previous}:{payload}".encode("utf-8")).hexdigest()
        cumulative.append(previous)
    return tuple(cumulative)


def _stage_requires_normalized_audio(stage: Stage) -> bool:
    consumer = getattr(stage, "aligner", None)
    if consumer is None:
        consumer = getattr(stage, "transcriber", None)
    return bool(getattr(consumer, "requires_normalized_audio", False))


def _audio_digest(source_path: Path) -> str:
    digest = sha256()
    with Path(source_path).open("rb") as source:
        while chunk := source.read(DEFAULT_AUDIO_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_configuration(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_stable_configuration(item) for item in value]
    if isinstance(value, list | dict | set):
        return {"type": type(value).__name__}
    if is_dataclass(value):
        configuration: dict[str, object] = {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "implementation": _implementation_digest(type(value)),
        }
        for item in fields(value):
            if item.name.startswith("_") or "token" in item.name.lower():
                continue
            field_value = getattr(value, item.name)
            if isinstance(field_value, list | dict | set):
                continue
            configuration[item.name] = _stable_configuration(field_value)
        return configuration
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "implementation": _implementation_digest(type(value)),
    }


def _implementation_digest(value_type: type[object]) -> str | None:
    try:
        source = inspect.getsource(value_type)
    except (OSError, TypeError):
        return None
    return sha256(source.encode("utf-8")).hexdigest()


def _rebind_context(
    context: PipelineContext,
    *,
    source_path: Path,
    working_directory: Path,
    run_id: str,
) -> PipelineContext:
    return PipelineContext(
        run_id=run_id,
        working_directory=working_directory,
        document=Document(
            source_path=source_path,
            segments=context.document.segments,
            subtitles=context.document.subtitles,
        ),
    )

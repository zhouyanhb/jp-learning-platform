from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from jp_learning_platform.application import (
    AudioInputDiscovery,
    NoAudioInputsFoundError,
    SubtitlePipelineRequest,
)
from jp_learning_platform.domain import (
    Segment,
    Sentence,
    TimeRange,
    ValidationResult,
    Word,
)
from jp_learning_platform.infrastructure import (
    AudioLoader,
    CompositeSubtitleWriter,
    ListeningJsonWriter,
    SrtSubtitleWriter,
    StageArtifactStore,
    WordSubtitleBuilder,
)
from jp_learning_platform.workflow import (
    DuplicateSubtitleOutputError,
    PipelineProgressEvent,
    QwenRepair,
    QwenRepairRequest,
    ReadabilityOptimization,
    ReadabilityOptimizationRequest,
    SubtitleMerge,
    SubtitleMergeRequest,
    SubtitlePipelineRunner,
    SubtitleValidation,
    SubtitleValidationRequest,
    WhisperTranscript,
    WhisperTranscriptionRequest,
    WhisperXAlignment,
    WhisperXAlignmentRequest,
)


@dataclass(slots=True)
class FakeTranscriber:
    requests: list[WhisperTranscriptionRequest]

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        self.requests.append(request)
        words = (
            Word(text="日本語", time_range=TimeRange(0.0, 0.5), confidence=0.9),
            Word(text="です", time_range=TimeRange(0.6, 1.0), confidence=0.8),
        )
        sentence = Sentence(
            text=f"{request.source_path.stem}です。",
            time_range=TimeRange(0.0, 1.1),
            words=words,
        )
        segment = Segment(
            position=0,
            text=sentence.text,
            time_range=sentence.time_range,
            sentences=(sentence,),
        )
        return WhisperTranscript(
            source_path=request.source_path,
            segments=(segment,),
        )


@dataclass(slots=True)
class RecordingAligner:
    requests: list[WhisperXAlignmentRequest]

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        self.requests.append(request)
        return WhisperXAlignment(
            source_path=request.source_path,
            segments=request.segments,
        )


@dataclass(slots=True)
class RecordingRepairer:
    requests: list[QwenRepairRequest]

    def repair(self, request: QwenRepairRequest) -> QwenRepair:
        self.requests.append(request)
        return QwenRepair(
            source_path=request.source_path,
            segments=request.segments,
        )


@dataclass(slots=True)
class RecordingMerger:
    requests: list[SubtitleMergeRequest]

    def merge(self, request: SubtitleMergeRequest) -> SubtitleMerge:
        self.requests.append(request)
        return SubtitleMerge(
            source_path=request.source_path,
            subtitles=request.subtitles,
        )


@dataclass(slots=True)
class RecordingOptimizer:
    requests: list[ReadabilityOptimizationRequest]

    def optimize(
        self,
        request: ReadabilityOptimizationRequest,
    ) -> ReadabilityOptimization:
        self.requests.append(request)
        return ReadabilityOptimization(
            source_path=request.source_path,
            subtitles=request.subtitles,
        )


@dataclass(slots=True)
class RecordingValidator:
    requests: list[SubtitleValidationRequest]

    def validate(self, request: SubtitleValidationRequest) -> SubtitleValidation:
        self.requests.append(request)
        return SubtitleValidation(
            source_path=request.source_path,
            result=ValidationResult(),
        )


@dataclass(slots=True)
class RecordingProgressReporter:
    events: list[PipelineProgressEvent]

    def report(self, event: PipelineProgressEvent) -> None:
        self.events.append(event)


def _write_audio(path: Path) -> None:
    path.write_bytes(b"audio")


def _runner(output_directory: Path, transcriber: FakeTranscriber) -> SubtitlePipelineRunner:
    return SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        builder=WordSubtitleBuilder(),
        writer=SrtSubtitleWriter(output_directory=output_directory),
    )


def test_subtitle_pipeline_runner_generates_srt_for_single_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "lesson.mp3"
    output_directory = tmp_path / "output"
    _write_audio(audio_path)
    transcriber = FakeTranscriber(requests=[])

    result = _runner(output_directory, transcriber).run(
        SubtitlePipelineRequest(input_path=audio_path, output_directory=output_directory)
    )

    assert result.output_paths == (output_directory / "lesson.srt",)
    assert result.output_paths[0].read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,100\nlessonです。\n\n"
    )
    assert transcriber.requests[0].source_path == audio_path


def test_subtitle_pipeline_runner_can_use_json_primary_output_and_srt_export(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lesson.mp3"
    output_directory = tmp_path / "output"
    _write_audio(audio_path)
    transcriber = FakeTranscriber(requests=[])
    writer = CompositeSubtitleWriter(
        primary_writer=ListeningJsonWriter(output_directory=output_directory),
        export_writers=(SrtSubtitleWriter(output_directory=output_directory),),
    )
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        builder=WordSubtitleBuilder(),
        writer=writer,
        output_extension=".json",
    )

    result = runner.run(
        SubtitlePipelineRequest(input_path=audio_path, output_directory=output_directory)
    )

    assert result.output_paths == (output_directory / "lesson.json",)
    payload = json.loads(result.output_paths[0].read_text(encoding="utf-8"))
    assert payload["segments"][0]["sentences"][0]["words"][0]["text"] == "日本語"
    assert (output_directory / "lesson.srt").exists()


def test_subtitle_pipeline_runner_can_execute_quality_stages(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lesson.mp3"
    output_directory = tmp_path / "output"
    _write_audio(audio_path)
    transcriber = FakeTranscriber(requests=[])
    aligner = RecordingAligner(requests=[])
    repairer = RecordingRepairer(requests=[])
    merger = RecordingMerger(requests=[])
    optimizer = RecordingOptimizer(requests=[])
    validator = RecordingValidator(requests=[])
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        aligner=aligner,
        repairer=repairer,
        builder=WordSubtitleBuilder(),
        merger=merger,
        optimizer=optimizer,
        validator=validator,
        writer=SrtSubtitleWriter(output_directory=output_directory),
    )

    result = runner.run(
        SubtitlePipelineRequest(
            input_path=audio_path,
            output_directory=output_directory,
        )
    )

    assert result.output_paths == (output_directory / "lesson.srt",)
    assert len(aligner.requests) == 1
    assert len(repairer.requests) == 1
    assert len(merger.requests) == 1
    assert len(optimizer.requests) == 1
    assert len(validator.requests) == 1


def test_subtitle_pipeline_runner_records_progress_and_stage_artifacts(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lesson.mp3"
    output_directory = tmp_path / "output"
    _write_audio(audio_path)
    reporter = RecordingProgressReporter(events=[])
    artifact_store = StageArtifactStore(
        root_directory=output_directory / ".work",
        run_name="run-001",
    )
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=FakeTranscriber(requests=[]),
        aligner=RecordingAligner(requests=[]),
        repairer=RecordingRepairer(requests=[]),
        builder=WordSubtitleBuilder(),
        merger=RecordingMerger(requests=[]),
        optimizer=RecordingOptimizer(requests=[]),
        validator=RecordingValidator(requests=[]),
        writer=SrtSubtitleWriter(output_directory=output_directory),
        progress_reporter=reporter,
        artifact_recorder=artifact_store,
    )

    result = runner.run(
        SubtitlePipelineRequest(input_path=audio_path, output_directory=output_directory)
    )

    assert result.output_paths == (output_directory / "lesson.srt",)
    artifact_directory = output_directory / ".work" / "run-001" / "lesson"
    expected_artifacts = (
        "00_audio_load.json",
        "01_whisper.json",
        "02_align.json",
        "03_repair.json",
        "04_build.json",
        "05_merge.json",
        "06_readability.json",
        "07_validate.json",
        "08_write.json",
    )
    for artifact_name in expected_artifacts:
        assert (artifact_directory / artifact_name).exists()

    manifest = json.loads(
        (artifact_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["current_stage"] == "subtitle-writer"
    assert manifest["status"] == "succeeded"

    assert [event.stage_name for event in reporter.events[::2]] == [
        "audio-loader",
        "whisper",
        "whisperx-alignment",
        "qwen-repair",
        "subtitle-builder",
        "subtitle-merger",
        "readability-optimizer",
        "subtitle-validator",
        "subtitle-writer",
    ]
    assert all(event.file_index == 1 for event in reporter.events)
    assert all(event.file_total == 1 for event in reporter.events)


def test_subtitle_pipeline_runner_generates_srt_for_audio_folder(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "audio"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    _write_audio(input_directory / "b.wav")
    _write_audio(input_directory / "a.mp3")
    (input_directory / "notes.txt").write_text("skip", encoding="utf-8")
    transcriber = FakeTranscriber(requests=[])

    result = _runner(output_directory, transcriber).run(
        SubtitlePipelineRequest(
            input_path=input_directory,
            output_directory=output_directory,
        )
    )

    assert result.output_paths == (
        output_directory / "a.srt",
        output_directory / "b.srt",
    )
    assert [request.source_path.name for request in transcriber.requests] == [
        "a.mp3",
        "b.wav",
    ]


def test_audio_input_discovery_rejects_folder_without_audio(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")

    with pytest.raises(NoAudioInputsFoundError):
        AudioInputDiscovery().discover(tmp_path)


def test_subtitle_pipeline_runner_rejects_duplicate_output_paths(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "audio"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    _write_audio(input_directory / "lesson.mp3")
    _write_audio(input_directory / "lesson.wav")

    with pytest.raises(DuplicateSubtitleOutputError):
        _runner(output_directory, FakeTranscriber(requests=[])).run(
            SubtitlePipelineRequest(
                input_path=input_directory,
                output_directory=output_directory,
            )
        )

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
    LocalPipelineContextCache,
    SrtSubtitleWriter,
    StageArtifactStore,
    WordSubtitleBuilder,
)
from jp_learning_platform.workflow import (
    DuplicateSubtitleOutputError,
    HomophoneResolution,
    HomophoneResolutionDecision,
    HomophoneResolutionRequest,
    PipelineProgressEvent,
    StagePhaseTiming,
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
    requires_normalized_audio: bool = False
    phase_timings: tuple[StagePhaseTiming, ...] = ()

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        self.requests.append(request)
        return WhisperXAlignment(
            source_path=request.source_path,
            segments=request.segments,
            phase_timings=self.phase_timings,
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
class RecordingHomophoneResolver:
    requests: list[HomophoneResolutionRequest]

    def resolve(
        self,
        request: HomophoneResolutionRequest,
    ) -> HomophoneResolution:
        self.requests.append(request)
        return HomophoneResolution(
            source_path=request.source_path,
            segments=request.segments,
            decisions=(
                HomophoneResolutionDecision(
                    segment_position=0,
                    sentence_index=0,
                    original_text="日本語",
                    selected_text="日本語",
                    reading="にほんご",
                    accepted=False,
                    reason="no_same_reading_candidate",
                ),
            ),
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


@dataclass(slots=True)
class RecordingAudioNormalizer:
    requests: list[Path]

    def normalize(
        self,
        source_path: Path,
        cache_directory: Path,
        audio_digest: str,
    ) -> Path:
        self.requests.append(source_path)
        return source_path


@dataclass(slots=True)
class FailingAudioNormalizer:
    requests: list[Path]

    def normalize(
        self,
        source_path: Path,
        cache_directory: Path,
        audio_digest: str,
    ) -> Path:
        self.requests.append(source_path)
        raise RuntimeError("normalization failed")


@dataclass(slots=True)
class ExtractingAudioNormalizer:
    requests: list[Path]

    def normalize(
        self,
        source_path: Path,
        cache_directory: Path,
        audio_digest: str,
    ) -> Path:
        self.requests.append(source_path)
        cache_directory.mkdir(parents=True, exist_ok=True)
        extracted_path = cache_directory / "extracted.wav"
        extracted_path.write_bytes(b"extracted audio")
        return extracted_path


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
    homophone_resolver = RecordingHomophoneResolver(requests=[])
    merger = RecordingMerger(requests=[])
    optimizer = RecordingOptimizer(requests=[])
    validator = RecordingValidator(requests=[])
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        aligner=aligner,
        repairer=repairer,
        homophone_resolver=homophone_resolver,
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
    assert len(homophone_resolver.requests) == 1
    assert len(merger.requests) == 1
    assert len(optimizer.requests) == 1
    assert len(validator.requests) == 1


def test_runner_reuses_complete_result_for_identical_audio_and_configuration(
    tmp_path: Path,
) -> None:
    first_audio = tmp_path / "first.mp3"
    second_audio = tmp_path / "second.mp3"
    output_directory = tmp_path / "output"
    _write_audio(first_audio)
    _write_audio(second_audio)
    transcriber = FakeTranscriber(requests=[])
    normalizer = RecordingAudioNormalizer(requests=[])
    reporter = RecordingProgressReporter(events=[])
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        builder=WordSubtitleBuilder(),
        writer=SrtSubtitleWriter(output_directory=output_directory),
        cache=LocalPipelineContextCache(output_directory / ".cache"),
        audio_normalizer=normalizer,
        progress_reporter=reporter,
    )

    runner.run(
        SubtitlePipelineRequest(
            input_path=first_audio,
            output_directory=output_directory,
        )
    )
    result = runner.run(
        SubtitlePipelineRequest(
            input_path=second_audio,
            output_directory=output_directory,
        )
    )

    assert len(transcriber.requests) == 1
    assert normalizer.requests == []
    assert result.output_paths == (output_directory / "second.srt",)
    assert result.output_paths[0].exists()
    assert any(event.message == "complete-result-hit" for event in reporter.events)
    assert not any(
        event.stage_name == "audio-normalization" for event in reporter.events
    )


def test_runner_reuses_stage_prefix_when_later_configuration_changes(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lesson.mp3"
    output_directory = tmp_path / "output"
    _write_audio(audio_path)
    transcriber = FakeTranscriber(requests=[])
    normalizer = RecordingAudioNormalizer(requests=[])
    cache = LocalPipelineContextCache(output_directory / ".cache")
    base_arguments = {
        "audio_loader": AudioLoader(),
        "transcriber": transcriber,
        "builder": WordSubtitleBuilder(),
        "writer": SrtSubtitleWriter(output_directory=output_directory),
        "cache": cache,
        "audio_normalizer": normalizer,
    }
    SubtitlePipelineRunner(**base_arguments).run(
        SubtitlePipelineRequest(
            input_path=audio_path,
            output_directory=output_directory,
        )
    )
    optimizer = RecordingOptimizer(requests=[])

    SubtitlePipelineRunner(**base_arguments, optimizer=optimizer).run(
        SubtitlePipelineRequest(
            input_path=audio_path,
            output_directory=output_directory,
        )
    )

    assert len(transcriber.requests) == 1
    assert normalizer.requests == []
    assert len(optimizer.requests) == 1


def test_runner_normalizes_only_before_explicit_exact_sample_consumer(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lesson.mp3"
    output_directory = tmp_path / "output"
    _write_audio(audio_path)
    transcriber = FakeTranscriber(requests=[])
    aligner = RecordingAligner(requests=[], requires_normalized_audio=True)
    cache = LocalPipelineContextCache(output_directory / ".cache")
    failing_normalizer = FailingAudioNormalizer(requests=[])
    request = SubtitlePipelineRequest(
        input_path=audio_path,
        output_directory=output_directory,
    )

    with pytest.raises(RuntimeError, match="normalization failed"):
        SubtitlePipelineRunner(
            audio_loader=AudioLoader(),
            transcriber=transcriber,
            aligner=aligner,
            builder=WordSubtitleBuilder(),
            writer=SrtSubtitleWriter(output_directory=output_directory),
            cache=cache,
            audio_normalizer=failing_normalizer,
        ).run(request)

    successful_normalizer = RecordingAudioNormalizer(requests=[])
    SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        aligner=aligner,
        builder=WordSubtitleBuilder(),
        writer=SrtSubtitleWriter(output_directory=output_directory),
        cache=cache,
        audio_normalizer=successful_normalizer,
    ).run(request)

    assert len(transcriber.requests) == 1
    assert failing_normalizer.requests == [audio_path]
    assert successful_normalizer.requests == [audio_path]
    assert len(aligner.requests) == 1


def test_runner_keeps_required_normalization_when_cache_is_disabled(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "lesson.mp3"
    output_directory = tmp_path / "output"
    _write_audio(audio_path)
    transcriber = FakeTranscriber(requests=[])
    aligner = RecordingAligner(requests=[], requires_normalized_audio=True)
    normalizer = RecordingAudioNormalizer(requests=[])
    reporter = RecordingProgressReporter(events=[])

    SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        aligner=aligner,
        builder=WordSubtitleBuilder(),
        writer=SrtSubtitleWriter(output_directory=output_directory),
        cache=None,
        audio_normalizer=normalizer,
        progress_reporter=reporter,
    ).run(
        SubtitlePipelineRequest(
            input_path=audio_path,
            output_directory=output_directory,
        )
    )

    succeeded_stages = [
        event.stage_name
        for event in reporter.events
        if event.status.value == "succeeded"
    ]
    assert normalizer.requests == [audio_path]
    assert succeeded_stages.index("whisper") < succeeded_stages.index(
        "audio-normalization"
    )
    assert succeeded_stages.index("audio-normalization") < succeeded_stages.index(
        "whisperx-alignment"
    )


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
        "07_build.json",
        "08_merge.json",
        "09_readability.json",
        "10_validate.json",
        "11_write.json",
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
        "pipeline-total",
    ]
    assert reporter.events[-1].elapsed_seconds is not None
    assert reporter.events[-1].status.value == "succeeded"
    assert all(event.file_index == 1 for event in reporter.events)
    assert all(event.file_total == 1 for event in reporter.events)


def test_runner_reports_alignment_phase_timings(tmp_path: Path) -> None:
    audio_path = tmp_path / "lesson.mp3"
    output_directory = tmp_path / "output"
    _write_audio(audio_path)
    reporter = RecordingProgressReporter(events=[])
    artifact_store = StageArtifactStore(
        root_directory=output_directory / ".work",
        run_name="phase-timing",
    )
    phase_timings = (
        StagePhaseTiming("whisperx-forced-alignment", 2.0),
        StagePhaseTiming("pyannote-diarization", 5.0),
        StagePhaseTiming("speaker-assignment", 0.25),
    )
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=FakeTranscriber(requests=[]),
        aligner=RecordingAligner(requests=[], phase_timings=phase_timings),
        builder=WordSubtitleBuilder(),
        writer=SrtSubtitleWriter(output_directory=output_directory),
        progress_reporter=reporter,
        artifact_recorder=artifact_store,
    )

    runner.run(
        SubtitlePipelineRequest(
            input_path=audio_path,
            output_directory=output_directory,
        )
    )

    succeeded = {
        event.stage_name: event.elapsed_seconds
        for event in reporter.events
        if event.status.value == "succeeded"
    }
    assert succeeded["whisperx-forced-alignment"] == 2.0
    assert succeeded["pyannote-diarization"] == 5.0
    assert succeeded["speaker-assignment"] == 0.25
    artifact_directory = artifact_store.audio_directory(audio_path)
    assert (artifact_directory / "02a_forced_alignment.json").exists()
    assert (artifact_directory / "02b_pyannote_diarization.json").exists()
    assert (artifact_directory / "02c_speaker_assignment.json").exists()


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


def test_runner_extracts_and_reuses_video_audio_before_transcription(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "lesson.mp4"
    output_directory = tmp_path / "output"
    video_path.write_bytes(b"video with audio")
    transcriber = FakeTranscriber(requests=[])
    extractor = ExtractingAudioNormalizer(requests=[])
    reporter = RecordingProgressReporter(events=[])
    artifact_store = StageArtifactStore(root_directory=output_directory / ".work")
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        builder=WordSubtitleBuilder(),
        writer=SrtSubtitleWriter(output_directory=output_directory),
        cache=LocalPipelineContextCache(output_directory / ".cache"),
        audio_normalizer=extractor,
        progress_reporter=reporter,
        artifact_recorder=artifact_store,
    )
    request = SubtitlePipelineRequest(video_path, output_directory)

    first = runner.run(request)
    second = runner.run(request)

    assert first.output_paths == second.output_paths == (
        output_directory / "lesson.srt",
    )
    assert [item.source_path for item in first.items] == [video_path]
    assert len(transcriber.requests) == 1
    assert transcriber.requests[0].source_path.suffix == ".wav"
    assert extractor.requests == [video_path]
    succeeded = [
        event.stage_name
        for event in reporter.events
        if event.status.value == "succeeded"
    ]
    assert succeeded.index("video-audio-extraction") < succeeded.index(
        "audio-loader"
    )
    assert any(event.message == "complete-result-hit" for event in reporter.events)
    assert (
        artifact_store.audio_directory(video_path)
        / "00_video_audio_extraction.json"
    ).exists()


def test_runner_reuses_audio_stage_result_across_different_videos(
    tmp_path: Path,
) -> None:
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mkv"
    first_video.write_bytes(b"first video container")
    second_video.write_bytes(b"second video container")
    output_directory = tmp_path / "output"
    transcriber = FakeTranscriber(requests=[])
    extractor = ExtractingAudioNormalizer(requests=[])
    reporter = RecordingProgressReporter(events=[])
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=transcriber,
        builder=WordSubtitleBuilder(),
        writer=SrtSubtitleWriter(output_directory=output_directory),
        cache=LocalPipelineContextCache(output_directory / ".cache"),
        audio_normalizer=extractor,
        progress_reporter=reporter,
    )

    first = runner.run(SubtitlePipelineRequest(first_video, output_directory))
    second = runner.run(SubtitlePipelineRequest(second_video, output_directory))

    assert first.output_paths == (output_directory / "first.srt",)
    assert second.output_paths == (output_directory / "second.srt",)
    assert extractor.requests == [first_video, second_video]
    assert len(transcriber.requests) == 1
    assert any(
        event.stage_name == "audio-content-cache"
        and event.message == "complete-result-hit"
        for event in reporter.events
    )


def test_audio_input_discovery_rejects_folder_without_audio(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")

    with pytest.raises(NoAudioInputsFoundError):
        AudioInputDiscovery().discover(tmp_path)


def test_audio_input_discovery_accepts_supported_video_containers(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / name for name in ("a.mp4", "b.mkv", "c.webm"))
    for path in paths:
        path.write_bytes(b"video")

    discovery = AudioInputDiscovery()

    assert discovery.discover(tmp_path) == paths
    assert all(discovery.is_video(path) for path in paths)


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

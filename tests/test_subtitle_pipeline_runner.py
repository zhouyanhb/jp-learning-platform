from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jp_learning_platform.application import (
    AudioInputDiscovery,
    NoAudioInputsFoundError,
    SubtitlePipelineRequest,
)
from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import AudioLoader, SrtSubtitleWriter, WordSubtitleBuilder
from jp_learning_platform.workflow import (
    DuplicateSubtitleOutputError,
    SubtitlePipelineRunner,
    WhisperTranscript,
    WhisperTranscriptionRequest,
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

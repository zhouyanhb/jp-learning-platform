from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Subtitle,
    TimeRange,
)
from jp_learning_platform.workflow import (
    InvalidWhisperTranscriberError,
    InvalidWhisperTranscriptError,
    StageResult,
    WhisperStage,
    WhisperTranscript,
    WhisperTranscriptionRequest,
)


@dataclass(slots=True)
class FakeTranscriber:
    transcript: WhisperTranscript
    requests: list[WhisperTranscriptionRequest]

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        self.requests.append(request)
        return self.transcript


@dataclass(frozen=True, slots=True)
class InvalidTranscriptTranscriber:
    def transcribe(self, request: WhisperTranscriptionRequest) -> object:
        return request


def _segment(position: int = 0, text: str = "こんにちは") -> Segment:
    return Segment(
        position=position,
        text=text,
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.5),
    )


def _context(source_path: Path) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path),
        working_directory=source_path.parent / "work",
    )


def test_whisper_stage_transcribes_document_source(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    segment = _segment()
    transcriber = FakeTranscriber(
        transcript=WhisperTranscript(source_path=source_path, segments=(segment,)),
        requests=[],
    )

    result = WhisperStage(transcriber=transcriber).run(_context(source_path))

    assert isinstance(result, StageResult)
    assert result.stage_name == "whisper"
    assert transcriber.requests == [
        WhisperTranscriptionRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
        )
    ]
    assert result.context.document.source_path == source_path
    assert result.context.document.segments == (segment,)
    assert result.context.run_id == "run-001"
    assert result.context.working_directory == tmp_path / "work"


def test_whisper_stage_preserves_existing_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = Subtitle(
        index=1,
        text="字幕",
        time_range=TimeRange(start_seconds=0.0, end_seconds=1.0),
    )
    context = PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path, subtitles=(subtitle,)),
        working_directory=tmp_path / "work",
    )
    transcriber = FakeTranscriber(
        transcript=WhisperTranscript(source_path=source_path, segments=(_segment(),)),
        requests=[],
    )

    result = WhisperStage(transcriber=transcriber).run(context)

    assert result.context.document.subtitles == (subtitle,)


def test_whisper_stage_accepts_custom_stage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    transcriber = FakeTranscriber(
        transcript=WhisperTranscript(source_path=source_path),
        requests=[],
    )
    stage = WhisperStage(transcriber=transcriber, name="  whisper-transcribe  ")

    result = stage.run(_context(source_path))

    assert stage.name == "whisper-transcribe"
    assert result.stage_name == "whisper-transcribe"


def test_whisper_stage_rejects_invalid_transcriber() -> None:
    with pytest.raises(InvalidWhisperTranscriberError):
        WhisperStage(transcriber=object())


def test_whisper_stage_rejects_invalid_transcript_return(tmp_path: Path) -> None:
    stage = WhisperStage(transcriber=InvalidTranscriptTranscriber())

    with pytest.raises(InvalidWhisperTranscriptError, match="WhisperTranscript"):
        stage.run(_context(tmp_path / "input.wav"))


def test_whisper_stage_rejects_mismatched_transcript_source(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    transcriber = FakeTranscriber(
        transcript=WhisperTranscript(source_path=tmp_path / "other.wav"),
        requests=[],
    )
    stage = WhisperStage(transcriber=transcriber)

    with pytest.raises(InvalidWhisperTranscriptError, match="source path"):
        stage.run(_context(source_path))


def test_whisper_transcript_requires_segments() -> None:
    with pytest.raises(TypeError, match="segments"):
        WhisperTranscript(source_path=Path("input.wav"), segments=(object(),))


def test_whisper_transcript_is_immutable() -> None:
    transcript = WhisperTranscript(source_path=Path("input.wav"), segments=(_segment(),))

    with pytest.raises(FrozenInstanceError):
        transcript.segments = ()

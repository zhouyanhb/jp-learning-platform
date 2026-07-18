from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import (
    DiarizingWhisperXAligner,
    PyannoteAuthTokenError,
    PyannoteSpeakerDiarizer,
    SpeakerTurn,
)
from jp_learning_platform.workflow import (
    WhisperXAlignment,
    WhisperXAlignmentRequest,
)


@dataclass(frozen=True, slots=True)
class FakeTurn:
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class FakeDiarization:
    records: tuple[tuple[FakeTurn, str], ...]

    def itertracks(
        self,
        yield_label: bool,
    ) -> tuple[tuple[FakeTurn, None, str], ...]:
        assert yield_label
        return tuple((turn, None, speaker) for turn, speaker in self.records)


@dataclass(slots=True)
class FakePipeline:
    diarization: object
    audio_paths: list[str]

    def __call__(self, audio_path: str) -> object:
        self.audio_paths.append(audio_path)
        return self.diarization


@dataclass(frozen=True, slots=True)
class FakePipelineOutput:
    speaker_diarization: FakeDiarization


@dataclass(frozen=True, slots=True)
class FakeIterablePipelineOutput:
    speaker_diarization: tuple[tuple[FakeTurn, str], ...]


@dataclass(slots=True)
class FakeAligner:
    alignment: WhisperXAlignment
    requests: list[WhisperXAlignmentRequest]

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        self.requests.append(request)
        return self.alignment


def _mixed_speaker_segment() -> Segment:
    words = (
        Word(
            text="そう",
            time_range=TimeRange(0.0, 0.4),
            confidence=0.9,
        ),
        Word(
            text="はい",
            time_range=TimeRange(0.6, 1.0),
            confidence=0.9,
        ),
    )
    sentence = Sentence(
        text="そうはい",
        time_range=TimeRange(0.0, 1.0),
        words=words,
    )
    return Segment(
        position=0,
        text="そうはい",
        time_range=TimeRange(0.0, 1.0),
        sentences=(sentence,),
    )


def _diarizer(diarization: FakeDiarization) -> PyannoteSpeakerDiarizer:
    diarizer = PyannoteSpeakerDiarizer(auth_token="token")
    diarizer._pipeline = FakePipeline(diarization=diarization, audio_paths=[])
    return diarizer


def test_pyannote_speaker_diarizer_accepts_pipeline_output_wrapper(
    tmp_path: Path,
) -> None:
    diarization = FakeDiarization(
        records=((FakeTurn(0.0, 0.5), "SPEAKER_00"),)
    )
    diarizer = PyannoteSpeakerDiarizer(auth_token="token")
    diarizer._pipeline = FakePipeline(
        diarization=FakePipelineOutput(speaker_diarization=diarization),
        audio_paths=[],
    )

    turns = diarizer.diarize(tmp_path / "audio.mp3")

    assert turns == (SpeakerTurn("SPEAKER_00", TimeRange(0.0, 0.5)),)


def test_pyannote_speaker_diarizer_accepts_iterable_pipeline_output(
    tmp_path: Path,
) -> None:
    diarizer = PyannoteSpeakerDiarizer(auth_token="token")
    diarizer._pipeline = FakePipeline(
        diarization=FakeIterablePipelineOutput(
            speaker_diarization=((FakeTurn(0.0, 0.5), "SPEAKER_00"),)
        ),
        audio_paths=[],
    )

    turns = diarizer.diarize(tmp_path / "audio.mp3")

    assert turns == (SpeakerTurn("SPEAKER_00", TimeRange(0.0, 0.5)),)


def test_pyannote_speaker_diarizer_extracts_turns(tmp_path: Path) -> None:
    diarization = FakeDiarization(
        records=(
            (FakeTurn(0.0, 0.5), "SPEAKER_00"),
            (FakeTurn(0.5, 1.0), "SPEAKER_01"),
        )
    )
    diarizer = _diarizer(diarization)

    turns = diarizer.diarize(tmp_path / "audio.mp3")

    assert turns == (
        SpeakerTurn("SPEAKER_00", TimeRange(0.0, 0.5)),
        SpeakerTurn("SPEAKER_01", TimeRange(0.5, 1.0)),
    )


def test_pyannote_speaker_diarizer_assigns_and_splits_speaker_runs(
    tmp_path: Path,
) -> None:
    diarization = FakeDiarization(
        records=(
            (FakeTurn(0.0, 0.5), "SPEAKER_00"),
            (FakeTurn(0.5, 1.1), "SPEAKER_01"),
        )
    )

    segments = _diarizer(diarization).assign_speakers(
        tmp_path / "audio.mp3",
        (_mixed_speaker_segment(),),
    )

    assert tuple(segment.text for segment in segments) == ("そう", "はい")
    assert tuple(segment.speaker_id for segment in segments) == (
        "SPEAKER_00",
        "SPEAKER_01",
    )
    assert tuple(segment.position for segment in segments) == (0, 1)
    assert tuple(
        segment.sentences[0].words[0].speaker_id for segment in segments
    ) == ("SPEAKER_00", "SPEAKER_01")


def test_diarizing_whisperx_aligner_wraps_base_alignment(tmp_path: Path) -> None:
    source_path = tmp_path / "audio.mp3"
    base_aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(_mixed_speaker_segment(),),
        ),
        requests=[],
    )
    diarization = FakeDiarization(
        records=(
            (FakeTurn(0.0, 0.5), "SPEAKER_00"),
            (FakeTurn(0.5, 1.1), "SPEAKER_01"),
        )
    )
    request = WhisperXAlignmentRequest(
        source_path=source_path,
        working_directory=tmp_path / "work",
        run_id="run-001",
        segments=(_mixed_speaker_segment(),),
    )

    result = DiarizingWhisperXAligner(
        base_aligner=base_aligner,
        diarizer=_diarizer(diarization),
    ).align(request)

    assert base_aligner.requests == [request]
    assert tuple(segment.speaker_id for segment in result.segments) == (
        "SPEAKER_00",
        "SPEAKER_01",
    )


def test_pyannote_speaker_diarizer_requires_auth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JP_TEST_HF_TOKEN", raising=False)

    with pytest.raises(PyannoteAuthTokenError):
        PyannoteSpeakerDiarizer(
            auth_token=None,
            token_environment_variable="JP_TEST_HF_TOKEN",
        )._load_pipeline()

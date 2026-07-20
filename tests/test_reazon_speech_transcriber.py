from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from jp_learning_platform.infrastructure import (
    DEFAULT_REAZON_SPEECH_MODEL_ID,
    ReazonSpeechTranscriber,
)
from jp_learning_platform.workflow import WhisperTranscriptionRequest


class FakeAudio:
    def __init__(self, waveform: object, samplerate: int) -> None:
        self.waveform = waveform
        self.samplerate = samplerate


def _request(source_path: Path) -> WhisperTranscriptionRequest:
    return WhisperTranscriptionRequest(
        source_path=source_path,
        working_directory=source_path.parent / "work",
        run_id="run-001",
    )


def _install_reazonspeech_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: object | list[object],
    audio: object | None = None,
) -> list[tuple[str, object]]:
    calls: list[tuple[str, object]] = []
    fake_model = object()
    fake_audio = audio if audio is not None else FakeAudio([0.0] * 12, 1)
    reazonspeech_module = types.ModuleType("reazonspeech")
    nemo_module = types.ModuleType("reazonspeech.nemo")
    asr_module = types.ModuleType("reazonspeech.nemo.asr")
    reazonspeech_module.__path__ = []
    nemo_module.__path__ = []
    reazonspeech_module.nemo = nemo_module
    nemo_module.asr = asr_module

    class FakeTranscribeConfig:
        def __init__(self, *, verbose: bool) -> None:
            self.verbose = verbose

    def load_model(*, device: str) -> object:
        calls.append(("load_model", device))
        return fake_model

    def audio_from_path(source_path: str) -> object:
        calls.append(("audio_from_path", source_path))
        return fake_audio

    def audio_from_numpy(waveform: object, samplerate: int) -> FakeAudio:
        calls.append(("audio_from_numpy", (len(waveform), samplerate)))
        return FakeAudio(waveform, samplerate)

    def transcribe(model: object, audio: object, config: object) -> object:
        calls.append(("transcribe", (model, audio, config)))
        if isinstance(result, list):
            return result.pop(0)
        return result

    asr_module.TranscribeConfig = FakeTranscribeConfig
    asr_module.audio_from_numpy = audio_from_numpy
    asr_module.audio_from_path = audio_from_path
    asr_module.load_model = load_model
    asr_module.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "reazonspeech", reazonspeech_module)
    monkeypatch.setitem(sys.modules, "reazonspeech.nemo", nemo_module)
    monkeypatch.setitem(sys.modules, "reazonspeech.nemo.asr", asr_module)
    return calls


def test_reazon_speech_transcriber_maps_segments_to_domain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    result = types.SimpleNamespace(
        text="これから音を聞いてください。音がよく聞こえないときは手を挙げてください。",
        segments=[
            {
                "start_seconds": 4.2,
                "end_seconds": 6.0,
                "text": "これから音を聞いてください。",
            },
            {
                "start_seconds": 6.81,
                "end_seconds": 10.79,
                "text": "音がよく聞こえないときは手を挙げてください。",
            },
        ],
    )
    calls = _install_reazonspeech_module(monkeypatch, result=result)

    output = ReazonSpeechTranscriber(device="cpu").transcribe(_request(source_path))

    assert output.source_path == source_path
    assert tuple(segment.text for segment in output.segments) == (
        "これから音を聞いてください。",
        "音がよく聞こえないときは手を挙げてください。",
    )
    assert output.segments[0].time_range.start_seconds == 4.2
    assert output.segments[1].time_range.end_seconds == 10.79
    assert output.segments[0].sentences[0].words == ()
    assert calls[0] == ("load_model", "cpu")
    assert calls[1] == ("audio_from_path", str(source_path))
    assert calls[2][0] == "transcribe"


def test_reazon_speech_transcriber_chunks_long_audio_with_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    chunk_results = [
        types.SimpleNamespace(
            text="一つ目",
            segments=[{"start_seconds": 1.0, "end_seconds": 3.0, "text": "一つ目"}],
        ),
        types.SimpleNamespace(
            text="重複 二つ目",
            segments=[
                {"start_seconds": 0.2, "end_seconds": 0.8, "text": "重複"},
                {"start_seconds": 2.0, "end_seconds": 4.0, "text": "二つ目"},
            ],
        ),
        types.SimpleNamespace(
            text="三つ目",
            segments=[{"start_seconds": 2.0, "end_seconds": 4.0, "text": "三つ目"}],
        ),
    ]
    calls = _install_reazonspeech_module(
        monkeypatch,
        result=chunk_results,
        audio=FakeAudio([0.0] * 24, 1),
    )

    output = ReazonSpeechTranscriber(
        chunk_length_seconds=10,
        chunk_overlap_seconds=2,
    ).transcribe(_request(source_path))

    assert tuple(segment.text for segment in output.segments) == (
        "一つ目",
        "二つ目",
        "三つ目",
    )
    assert tuple(
        (
            segment.time_range.start_seconds,
            segment.time_range.end_seconds,
        )
        for segment in output.segments
    ) == (
        (1.0, 3.0),
        (10.0, 12.0),
        (18.0, 20.0),
    )
    assert [
        call for call in calls if call[0] == "audio_from_numpy"
    ] == [
        ("audio_from_numpy", (10, 1)),
        ("audio_from_numpy", (10, 1)),
        ("audio_from_numpy", (8, 1)),
    ]


def test_reazon_speech_transcriber_filters_noise_and_resolves_overlaps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    result = types.SimpleNamespace(
        text="いつでもいいです問題1ではまず質問を聞いてください前半後半",
        segments=[
            {"start_seconds": 0.0, "end_seconds": 0.08, "text": "。"},
            {
                "start_seconds": 1.0,
                "end_seconds": 5.0,
                "text": "いつでもいいです問題1ではまず質問を聞いてください",
            },
            {
                "start_seconds": 0.9,
                "end_seconds": 1.8,
                "text": "いつでもいいです。",
            },
            {"start_seconds": 6.0, "end_seconds": 8.0, "text": "前半"},
            {"start_seconds": 7.0, "end_seconds": 9.0, "text": "後半"},
        ],
    )
    _install_reazonspeech_module(monkeypatch, result=result)

    output = ReazonSpeechTranscriber().transcribe(_request(source_path))

    assert tuple(segment.text for segment in output.segments) == (
        "いつでもいいです問題1ではまず質問を聞いてください",
        "前半",
        "後半",
    )
    assert tuple(
        (
            segment.time_range.start_seconds,
            segment.time_range.end_seconds,
        )
        for segment in output.segments
    ) == (
        (1.0, 5.0),
        (6.0, 7.5),
        (7.5, 9.0),
    )


def test_reazon_speech_transcriber_falls_back_to_full_text_without_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    _install_reazonspeech_module(
        monkeypatch,
        result=types.SimpleNamespace(text="これからN2の試験を始めます。"),
    )

    output = ReazonSpeechTranscriber().transcribe(_request(source_path))

    assert tuple(segment.text for segment in output.segments) == (
        "これからN2の試験を始めます。",
    )
    assert output.segments[0].time_range.start_seconds == 0.0


def test_reazon_speech_transcriber_uses_nemo_for_custom_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    model_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    transcribe_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fake_nemo = types.ModuleType("nemo")
    fake_collections = types.ModuleType("nemo.collections")
    fake_asr = types.ModuleType("nemo.collections.asr")
    fake_nemo.__path__ = []
    fake_collections.__path__ = []
    fake_nemo.collections = fake_collections
    fake_collections.asr = fake_asr

    class FakeModel:
        def transcribe(self, *args: object, **kwargs: object) -> list[str]:
            transcribe_calls.append((args, kwargs))
            return ["問題用紙にメモを取っても構いません。"]

    class FakeASRModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            model_calls.append((args, kwargs))
            return FakeModel()

    fake_asr.models = types.SimpleNamespace(ASRModel=FakeASRModel)
    monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", fake_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_asr)

    output = ReazonSpeechTranscriber(model_id="custom/reazon").transcribe(
        _request(source_path)
    )

    assert model_calls == [((), {"model_name": "custom/reazon"})]
    assert transcribe_calls == [
        ((), {"audio": [str(source_path)], "batch_size": 16})
    ]
    assert tuple(segment.text for segment in output.segments) == (
        "問題用紙にメモを取っても構いません。",
    )


def test_reazon_speech_transcriber_normalizes_default_alias() -> None:
    transcriber = ReazonSpeechTranscriber(model_id="reazonspeech-nemo-v2")

    assert transcriber.model_id == DEFAULT_REAZON_SPEECH_MODEL_ID


def test_reazon_speech_transcriber_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ReazonSpeechTranscriber(batch_size=0)

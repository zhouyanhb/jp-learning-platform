from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from jp_learning_platform.infrastructure import (
    DEFAULT_KOTOBA_WHISPER_MODEL_ID,
    KOTOBA_WHISPER_V2_0_MODEL_ID,
    KotobaWhisperTranscriber,
)
from jp_learning_platform.workflow import WhisperTranscriptionRequest


class RecordingKotobaPipeline:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.source_path = ""
        self.options: dict[str, object] = {}

    def __call__(self, source_path: str, **options: object) -> dict[str, object]:
        self.source_path = source_path
        self.options = options
        return self.output


def _request(source_path: Path) -> WhisperTranscriptionRequest:
    return WhisperTranscriptionRequest(
        source_path=source_path,
        working_directory=source_path.parent / "work",
        run_id="run-001",
    )


def test_kotoba_whisper_transcriber_maps_timestamp_chunks_to_segments(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    pipeline = RecordingKotobaPipeline(
        {
            "text": "これから音を聞いてください。音がよく聞こえないときは手を挙げてください。",
            "chunks": [
                {
                    "timestamp": (4.2, 6.0),
                    "text": "これから音を聞いてください。",
                },
                {
                    "timestamp": (6.81, 10.79),
                    "text": "音がよく聞こえないときは手を挙げてください。",
                },
            ],
        }
    )
    transcriber = KotobaWhisperTranscriber()
    transcriber._pipeline = pipeline

    result = transcriber.transcribe(_request(source_path))

    assert result.source_path == source_path
    assert pipeline.source_path == str(source_path)
    assert pipeline.options == {
        "chunk_length_s": 15.0,
        "batch_size": 16,
        "return_timestamps": True,
        "generate_kwargs": {
            "language": "ja",
            "task": "transcribe",
        },
    }
    assert tuple(segment.text for segment in result.segments) == (
        "これから音を聞いてください。",
        "音がよく聞こえないときは手を挙げてください。",
    )
    assert result.segments[0].time_range.start_seconds == 4.2
    assert result.segments[0].time_range.end_seconds == 6.0
    assert result.segments[0].sentences[0].words == ()
    assert result.segments[1].position == 1


def test_kotoba_whisper_transcriber_normalizes_v2_0_alias() -> None:
    transcriber = KotobaWhisperTranscriber(model_id="kotoba-whisper-v2.0")

    assert transcriber.model_id == KOTOBA_WHISPER_V2_0_MODEL_ID


def test_kotoba_whisper_transcriber_defaults_to_v2_1() -> None:
    transcriber = KotobaWhisperTranscriber()

    assert transcriber.model_id == DEFAULT_KOTOBA_WHISPER_MODEL_ID
    assert not transcriber.trust_remote_code


def test_kotoba_whisper_transcriber_uses_transformers_dtype_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    model_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processor_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_transformers = types.ModuleType("transformers")
    fake_processor = types.SimpleNamespace(
        tokenizer=object(),
        feature_extractor=object(),
    )
    fake_model = object()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(
            *args: object,
            **kwargs: object,
        ) -> types.SimpleNamespace:
            processor_calls.append((args, kwargs))
            return fake_processor

    class FakeAutoModelForSpeechSeq2Seq:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            model_calls.append((args, kwargs))
            return fake_model

    def pipeline(*args: object, **kwargs: object) -> RecordingKotobaPipeline:
        calls.append((args, kwargs))
        return RecordingKotobaPipeline({"text": "日本語"})

    fake_transformers.pipeline = pipeline
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoModelForSpeechSeq2Seq = FakeAutoModelForSpeechSeq2Seq
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    KotobaWhisperTranscriber()._load_pipeline()

    assert processor_calls == [
        (
            ("kotoba-tech/kotoba-whisper-v2.0",),
            {"trust_remote_code": False},
        )
    ]
    assert model_calls == [
        (
            ("kotoba-tech/kotoba-whisper-v2.1",),
            {
                "dtype": "float32",
                "use_safetensors": True,
                "trust_remote_code": False,
            },
        )
    ]
    assert calls
    args, kwargs = calls[0]
    assert args == ("automatic-speech-recognition",)
    assert kwargs["model"] is fake_model
    assert "torch_dtype" not in kwargs


def test_kotoba_whisper_transcriber_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        KotobaWhisperTranscriber(batch_size=0)

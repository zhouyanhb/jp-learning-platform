from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from jp_learning_platform.infrastructure import FasterWhisperTranscriber
from jp_learning_platform.workflow import WhisperTranscriptionRequest


class RecordingWhisperModel:
    def __init__(self) -> None:
        self.source_path = ""
        self.options: dict[str, object] = {}

    def transcribe(
        self,
        source_path: str,
        **options: object,
    ) -> tuple[tuple[object, ...], object]:
        self.source_path = source_path
        self.options = options
        return (), object()


def test_faster_whisper_transcriber_uses_centralized_default_options(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    model = RecordingWhisperModel()
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
        )
    )

    assert result.source_path == source_path
    assert result.segments == ()
    assert model.source_path == str(source_path)
    assert model.options == {
        "language": "ja",
        "initial_prompt": (
            "これは日本語学習教材の書き起こしです。"
            "自然な日本語の句読点を使用します。"
        ),
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 600},
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
    }


class RetryWhisperModel:
    def __init__(self, responses: list[tuple[object, ...]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def transcribe(self, source_path: str, **options: object) -> tuple[tuple[object, ...], object]:
        self.calls.append(options)
        return self.responses.pop(0), object()


def _external_segment(
    text: str,
    start: float,
    end: float,
    confidence: float,
) -> object:
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        words=(
            SimpleNamespace(
                word=text,
                start=start,
                end=end,
                probability=confidence,
            ),
        ),
    )


def test_retries_only_low_confidence_segments_with_reliable_audio_context(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    reliable = _external_segment("今日は", 0.0, 1.0, 0.95)
    uncertain = _external_segment("日本こ", 1.1, 2.0, 0.4)
    repaired = _external_segment("日本語", 1.1, 2.0, 0.9)
    model = RetryWhisperModel([(reliable, uncertain), (repaired,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert tuple(segment.text for segment in result.segments) == ("今日は", "日本語")
    assert len(model.calls) == 2
    assert model.calls[0]["initial_prompt"] == (
        "これは日本語学習教材の書き起こしです。"
        "自然な日本語の句読点を使用します。"
    )
    assert model.calls[1]["initial_prompt"] == "今日は"
    assert model.calls[1]["clip_timestamps"] == [1.1, 2.0]


def test_keeps_first_pass_when_retry_confidence_does_not_improve(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = _external_segment("日本語", 0.0, 1.0, 0.5)
    retry = _external_segment("日本こ", 0.0, 1.0, 0.52)
    model = RetryWhisperModel([(original,), (retry,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert tuple(segment.text for segment in result.segments) == ("日本語",)

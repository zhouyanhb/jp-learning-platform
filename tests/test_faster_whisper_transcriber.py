from __future__ import annotations

from pathlib import Path

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
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 350},
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
    }

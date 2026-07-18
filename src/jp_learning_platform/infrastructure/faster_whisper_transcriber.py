"""faster-whisper transcription adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.workflow.whisper_stage import (
    WhisperTranscript,
    WhisperTranscriptionRequest,
)

DEFAULT_WHISPER_MODEL_SIZE = "large-v3"
DEFAULT_WHISPER_LANGUAGE = "ja"
DEFAULT_WHISPER_DEVICE = "cpu"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_WHISPER_BEAM_SIZE = 5
DEFAULT_VAD_MIN_SILENCE_MS = 350


class FasterWhisperDependencyError(RuntimeError):
    """Raised when faster-whisper is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "faster-whisper is required for transcription. "
            "Install it with: python -m pip install -e '.[asr]'"
        )


@dataclass(slots=True)
class FasterWhisperTranscriber:
    """Transcribe Japanese audio into word-aware domain segments."""

    model_size: str = DEFAULT_WHISPER_MODEL_SIZE
    language: str = DEFAULT_WHISPER_LANGUAGE
    device: str = DEFAULT_WHISPER_DEVICE
    compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE
    beam_size: int = DEFAULT_WHISPER_BEAM_SIZE
    vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS
    _model: Any | None = field(default=None, init=False, repr=False)

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        if not isinstance(request, WhisperTranscriptionRequest):
            raise TypeError("request must be a WhisperTranscriptionRequest.")

        model = self._load_model()
        external_segments, _info = model.transcribe(
            str(request.source_path),
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self.vad_min_silence_ms},
            condition_on_previous_text=False,
        )

        segments: list[Segment] = []
        for external_segment in external_segments:
            text = str(getattr(external_segment, "text", "")).strip()
            if text:
                segments.append(self._convert_segment(len(segments), external_segment))

        return WhisperTranscript(
            source_path=request.source_path,
            segments=tuple(segments),
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise FasterWhisperDependencyError() from error

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        return self._model

    def _convert_segment(self, position: int, external_segment: Any) -> Segment:
        text = str(getattr(external_segment, "text", "")).strip()
        start_seconds = float(getattr(external_segment, "start"))
        end_seconds = float(getattr(external_segment, "end"))
        words = tuple(
            self._convert_word(external_word)
            for external_word in (getattr(external_segment, "words", None) or ())
            if str(getattr(external_word, "word", "")).strip()
        )

        if words:
            start_seconds = min(start_seconds, words[0].time_range.start_seconds)
            end_seconds = max(end_seconds, words[-1].time_range.end_seconds)

        time_range = TimeRange(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        sentence = Sentence(
            text=text,
            time_range=time_range,
            words=words,
            speaker_id=_speaker_id(external_segment),
        )
        return Segment(
            position=position,
            text=text,
            time_range=time_range,
            sentences=(sentence,),
            speaker_id=_speaker_id(external_segment),
        )

    def _convert_word(self, external_word: Any) -> Word:
        probability = getattr(external_word, "probability", None)
        return Word(
            text=str(getattr(external_word, "word", "")).strip(),
            time_range=TimeRange(
                start_seconds=float(getattr(external_word, "start")),
                end_seconds=float(getattr(external_word, "end")),
            ),
            confidence=float(probability) if probability is not None else None,
            speaker_id=_speaker_id(external_word),
        )


def _speaker_id(source: Any) -> str | None:
    value = getattr(source, "speaker", getattr(source, "speaker_id", None))
    if value is None:
        return None

    return str(value).strip() or None


__all__ = [
    "DEFAULT_WHISPER_COMPUTE_TYPE",
    "DEFAULT_WHISPER_DEVICE",
    "DEFAULT_WHISPER_MODEL_SIZE",
    "FasterWhisperDependencyError",
    "FasterWhisperTranscriber",
]

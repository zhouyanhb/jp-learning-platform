"""faster-whisper transcription adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG,
)
from jp_learning_platform.workflow.whisper_stage import (
    WhisperTranscript,
    WhisperTranscriptionRequest,
)

DEFAULT_WHISPER_MODEL_SIZE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.model_size
DEFAULT_WHISPER_LANGUAGE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.language
DEFAULT_WHISPER_DEVICE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.device
DEFAULT_WHISPER_COMPUTE_TYPE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.compute_type
DEFAULT_WHISPER_BEAM_SIZE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.beam_size
DEFAULT_WHISPER_BEST_OF = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.best_of
DEFAULT_WHISPER_TEMPERATURE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.temperature
DEFAULT_WHISPER_WORD_TIMESTAMPS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.word_timestamps
)
DEFAULT_WHISPER_VAD_FILTER = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.vad_filter
DEFAULT_VAD_MIN_SILENCE_MS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.vad_min_silence_ms
)
DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.condition_on_previous_text
)
DEFAULT_WHISPER_HALLUCINATION_SILENCE_THRESHOLD_SECONDS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.hallucination_silence_threshold_seconds
)


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
    best_of: int = DEFAULT_WHISPER_BEST_OF
    temperature: float = DEFAULT_WHISPER_TEMPERATURE
    word_timestamps: bool = DEFAULT_WHISPER_WORD_TIMESTAMPS
    vad_filter: bool = DEFAULT_WHISPER_VAD_FILTER
    vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS
    condition_on_previous_text: bool = DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT
    hallucination_silence_threshold_seconds: float = (
        DEFAULT_WHISPER_HALLUCINATION_SILENCE_THRESHOLD_SECONDS
    )
    _model: Any | None = field(default=None, init=False, repr=False)

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        if not isinstance(request, WhisperTranscriptionRequest):
            raise TypeError("request must be a WhisperTranscriptionRequest.")

        model = self._load_model()
        external_segments, _info = model.transcribe(
            str(request.source_path),
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
            vad_parameters={"min_silence_duration_ms": self.vad_min_silence_ms},
            condition_on_previous_text=self.condition_on_previous_text,
            hallucination_silence_threshold=(
                self.hallucination_silence_threshold_seconds
            ),
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
            start_seconds = min(
                start_seconds,
                *(word.time_range.start_seconds for word in words),
            )
            end_seconds = max(
                end_seconds,
                *(word.time_range.end_seconds for word in words),
            )

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
    "DEFAULT_VAD_MIN_SILENCE_MS",
    "DEFAULT_WHISPER_BEAM_SIZE",
    "DEFAULT_WHISPER_BEST_OF",
    "DEFAULT_WHISPER_COMPUTE_TYPE",
    "DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT",
    "DEFAULT_WHISPER_DEVICE",
    "DEFAULT_WHISPER_HALLUCINATION_SILENCE_THRESHOLD_SECONDS",
    "DEFAULT_WHISPER_LANGUAGE",
    "DEFAULT_WHISPER_MODEL_SIZE",
    "DEFAULT_WHISPER_TEMPERATURE",
    "DEFAULT_WHISPER_VAD_FILTER",
    "DEFAULT_WHISPER_WORD_TIMESTAMPS",
    "FasterWhisperDependencyError",
    "FasterWhisperTranscriber",
]

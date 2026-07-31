"""faster-whisper transcription adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite
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
DEFAULT_WHISPER_INITIAL_PROMPT = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.initial_prompt
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
DEFAULT_RETRY_CONFIDENCE_THRESHOLD = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_confidence_threshold
)
DEFAULT_RETRY_CONTEXT_CONFIDENCE_THRESHOLD = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_context_confidence_threshold
)
DEFAULT_RETRY_MIN_CONFIDENCE_IMPROVEMENT = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_min_confidence_improvement
)
DEFAULT_RETRY_MAX_SEGMENTS = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_max_segments


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
    initial_prompt: str = DEFAULT_WHISPER_INITIAL_PROMPT
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
    retry_confidence_threshold: float = DEFAULT_RETRY_CONFIDENCE_THRESHOLD
    retry_context_confidence_threshold: float = (
        DEFAULT_RETRY_CONTEXT_CONFIDENCE_THRESHOLD
    )
    retry_min_confidence_improvement: float = (
        DEFAULT_RETRY_MIN_CONFIDENCE_IMPROVEMENT
    )
    retry_max_segments: int = DEFAULT_RETRY_MAX_SEGMENTS
    _model: Any | None = field(default=None, init=False, repr=False)

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        if not isinstance(request, WhisperTranscriptionRequest):
            raise TypeError("request must be a WhisperTranscriptionRequest.")

        model = self._load_model()
        source_path = str(request.source_path)
        external_segments, _info = model.transcribe(
            source_path,
            **self._transcription_options(),
        )
        selected_segments = self._retry_low_confidence_segments(
            model,
            source_path,
            tuple(external_segments),
        )

        segments: list[Segment] = []
        for external_segment in selected_segments:
            text = str(getattr(external_segment, "text", "")).strip()
            if text:
                segments.append(self._convert_segment(len(segments), external_segment))

        return WhisperTranscript(
            source_path=request.source_path,
            segments=tuple(segments),
        )

    def _transcription_options(self) -> dict[str, object]:
        return {
            "language": self.language,
            "initial_prompt": self.initial_prompt,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "vad_parameters": {
                "min_silence_duration_ms": self.vad_min_silence_ms,
            },
            # The first pass is intentionally neutral. Context is introduced only
            # for bounded retries whose preceding text is already high confidence.
            "condition_on_previous_text": self.condition_on_previous_text,
            "hallucination_silence_threshold": (
                self.hallucination_silence_threshold_seconds
            ),
        }

    def _retry_low_confidence_segments(
        self,
        model: Any,
        source_path: str,
        segments: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        if self.retry_max_segments <= 0:
            return segments

        selected: list[Any] = []
        retries = 0
        reliable_context = ""
        for segment in segments:
            confidence = _external_segment_confidence((segment,))
            replacement = (segment,)
            if (
                confidence < self.retry_confidence_threshold
                and retries < self.retry_max_segments
            ):
                retries += 1
                retry_segments, _info = model.transcribe(
                    source_path,
                    **self._retry_options(segment, reliable_context),
                )
                candidate = tuple(retry_segments)
                candidate_confidence = _external_segment_confidence(candidate)
                if candidate and isfinite(candidate_confidence) and (
                    not isfinite(confidence)
                    or candidate_confidence
                    >= confidence + self.retry_min_confidence_improvement
                ):
                    replacement = candidate
                    confidence = candidate_confidence

            selected.extend(replacement)
            if confidence >= self.retry_context_confidence_threshold:
                reliable_context = "".join(
                    str(getattr(item, "text", "")).strip()
                    for item in replacement
                )

        return tuple(selected)

    def _retry_options(
        self,
        segment: Any,
        reliable_context: str,
    ) -> dict[str, object]:
        options = self._transcription_options()
        options.update(
            {
                "clip_timestamps": [
                    float(getattr(segment, "start")),
                    float(getattr(segment, "end")),
                ],
                "initial_prompt": reliable_context or self.initial_prompt,
                "condition_on_previous_text": False,
            }
        )
        return options

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
            asr_boundary_word_indexes=_text_boundary_word_indexes(text, words),
        )
        return Segment(
            position=position,
            text=text,
            time_range=time_range,
            sentences=(sentence,),
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
        )


def _external_segment_confidence(segments: tuple[Any, ...]) -> float:
    probabilities = tuple(
        float(probability)
        for segment in segments
        for word in (getattr(segment, "words", None) or ())
        if (probability := getattr(word, "probability", None)) is not None
    )
    if not probabilities:
        return -inf

    return sum(probabilities) / len(probabilities)


def _text_boundary_word_indexes(text: str, words: tuple[Word, ...]) -> tuple[int, ...]:
    boundary_offsets: set[int] = set()
    offset = 0
    for character in text:
        if character.isspace():
            if offset:
                boundary_offsets.add(offset)
        else:
            offset += 1
    indexes: list[int] = []
    word_offset = 0
    for index, word in enumerate(words[:-1], start=1):
        word_offset += len("".join(word.text.split()))
        if word_offset in boundary_offsets:
            indexes.append(index)
    return tuple(indexes)


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
    "DEFAULT_RETRY_CONFIDENCE_THRESHOLD",
    "DEFAULT_RETRY_CONTEXT_CONFIDENCE_THRESHOLD",
    "DEFAULT_RETRY_MAX_SEGMENTS",
    "DEFAULT_RETRY_MIN_CONFIDENCE_IMPROVEMENT",
    "FasterWhisperDependencyError",
    "FasterWhisperTranscriber",
]

"""WhisperX alignment adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_WHISPERX_ALIGNMENT_CONFIG,
)
from jp_learning_platform.workflow.whisperx_alignment_stage import (
    WhisperXAlignment,
    WhisperXAlignmentRequest,
)

DEFAULT_WHISPERX_LANGUAGE = DEFAULT_WHISPERX_ALIGNMENT_CONFIG.language_code


class WhisperXDependencyError(RuntimeError):
    """Raised when WhisperX is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "whisperx is required for forced alignment. "
            "Install it with: python -m pip install -e '.[align]'"
        )


@dataclass(frozen=True, slots=True)
class PassthroughWhisperXAligner:
    """Keep existing segment timings while still running the alignment stage."""

    @property
    def requires_normalized_audio(self) -> bool:
        """The pass-through adapter never decodes the source audio."""
        return False

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        if not isinstance(request, WhisperXAlignmentRequest):
            raise TypeError("request must be a WhisperXAlignmentRequest.")

        return WhisperXAlignment(
            source_path=request.source_path,
            segments=request.segments,
        )


@dataclass(slots=True)
class WhisperXAlignerAdapter:
    """Align transcribed Japanese segments with WhisperX."""

    device: str
    language_code: str = DEFAULT_WHISPERX_LANGUAGE
    _align_model: Any | None = field(default=None, init=False, repr=False)
    _metadata: Any | None = field(default=None, init=False, repr=False)

    @property
    def requires_normalized_audio(self) -> bool:
        """WhisperX performs its own supported-media decoding."""
        return False

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        if not isinstance(request, WhisperXAlignmentRequest):
            raise TypeError("request must be a WhisperXAlignmentRequest.")

        whisperx = self._load_dependency()
        align_model, metadata = self._load_model(whisperx)
        audio = whisperx.load_audio(str(request.source_path))
        aligned = whisperx.align(
            self._to_whisperx_segments(request.segments),
            align_model,
            metadata,
            audio,
            self.device,
        )
        segments = self._to_domain_segments(aligned.get("segments", ()))
        if not segments:
            segments = request.segments

        return WhisperXAlignment(
            source_path=request.source_path,
            segments=segments,
        )

    def _load_dependency(self) -> Any:
        try:
            import whisperx
        except ImportError as error:
            raise WhisperXDependencyError() from error

        return whisperx

    def _load_model(self, whisperx: Any) -> tuple[Any, Any]:
        if self._align_model is None or self._metadata is None:
            self._align_model, self._metadata = whisperx.load_align_model(
                language_code=self.language_code,
                device=self.device,
            )

        return self._align_model, self._metadata

    def _to_whisperx_segments(
        self,
        segments: tuple[Segment, ...],
    ) -> list[dict[str, object]]:
        return [
            {
                "start": segment.time_range.start_seconds,
                "end": segment.time_range.end_seconds,
                "text": segment.text,
            }
            for segment in segments
        ]

    def _to_domain_segments(self, external_segments: Any) -> tuple[Segment, ...]:
        segments: list[Segment] = []
        for external_segment in external_segments:
            text = str(_value(external_segment, "text", "")).strip()
            if not text:
                continue

            start_seconds = float(_value(external_segment, "start", 0.0))
            end_seconds = float(_value(external_segment, "end", start_seconds))
            words = tuple(
                word
                for word in (
                    self._to_domain_word(external_word)
                    for external_word in _value(external_segment, "words", ())
                )
                if word is not None
            )

            if words:
                start_seconds = min(start_seconds, words[0].time_range.start_seconds)
                end_seconds = max(end_seconds, words[-1].time_range.end_seconds)

            time_range = TimeRange(start_seconds, end_seconds)
            sentence = Sentence(
                text=text,
                time_range=time_range,
                words=words,
                asr_boundary_word_indexes=_text_boundary_word_indexes(text, words),
            )
            segments.append(
                Segment(
                    position=len(segments),
                    text=text,
                    time_range=time_range,
                    sentences=(sentence,),
                )
            )

        return tuple(segments)

    def _to_domain_word(self, external_word: Any) -> Word | None:
        text = str(_value(external_word, "word", "")).strip()
        start = _value(external_word, "start", None)
        end = _value(external_word, "end", None)
        if not text or start is None or end is None:
            return None

        confidence = _value(
            external_word,
            "score",
            _value(external_word, "probability", None),
        )
        return Word(
            text=text,
            time_range=TimeRange(float(start), float(end)),
            confidence=float(confidence) if confidence is not None else None,
        )


def _value(source: Any, key: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)

    return getattr(source, key, default)


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
    "DEFAULT_WHISPERX_LANGUAGE",
    "PassthroughWhisperXAligner",
    "WhisperXAlignerAdapter",
    "WhisperXDependencyError",
]

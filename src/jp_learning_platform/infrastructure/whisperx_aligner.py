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
        segments = self._to_domain_segments(
            aligned.get("segments", ()),
            request.segments,
        )
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

    def _to_domain_segments(
        self,
        external_segments: Any,
        source_segments: tuple[Segment, ...] = (),
    ) -> tuple[Segment, ...]:
        segments: list[Segment] = []
        for external_segment in external_segments:
            text = str(_value(external_segment, "text", "")).strip()
            if not text:
                continue

            source_segment = (
                source_segments[len(segments)]
                if len(segments) < len(source_segments)
                else None
            )
            start_seconds = float(_value(external_segment, "start", 0.0))
            end_seconds = float(_value(external_segment, "end", start_seconds))
            aligned_words = tuple(
                word
                for word in (
                    self._to_domain_word(external_word)
                    for external_word in _value(external_segment, "words", ())
                )
                if word is not None
            )
            words = _project_source_words(source_segment, aligned_words, text)

            if words:
                start_seconds = min(
                    start_seconds,
                    *(word.time_range.start_seconds for word in words),
                )
                end_seconds = max(
                    end_seconds,
                    *(word.time_range.end_seconds for word in words),
                )

            time_range = TimeRange(start_seconds, end_seconds)
            speaker_id = _speaker_id(external_segment) or _common_speaker_id(words)
            sentence = Sentence(
                text=text,
                time_range=time_range,
                words=words,
                speaker_id=speaker_id,
            )
            segments.append(
                Segment(
                    position=len(segments),
                    text=text,
                    time_range=time_range,
                    sentences=(sentence,),
                    speaker_id=speaker_id,
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
            speaker_id=_speaker_id(external_word),
        )


def _speaker_id(source: Any) -> str | None:
    speaker = _value(source, "speaker", _value(source, "speaker_id", None))
    if speaker is None:
        return None

    return str(speaker).strip() or None


def _common_speaker_id(words: tuple[Word, ...]) -> str | None:
    speaker_ids = tuple(
        dict.fromkeys(word.speaker_id for word in words if word.speaker_id is not None)
    )
    if len(speaker_ids) == 1:
        return speaker_ids[0]

    return None


def _value(source: Any, key: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)

    return getattr(source, key, default)


def _project_source_words(
    source_segment: Segment | None,
    aligned_words: tuple[Word, ...],
    aligned_text: str,
) -> tuple[Word, ...]:
    if source_segment is None or not aligned_words:
        return aligned_words

    source_words = tuple(
        word for sentence in source_segment.sentences for word in sentence.words
    )
    if not source_words:
        return aligned_words

    source_text = _compact_text("".join(word.text for word in source_words))
    aligned_words_text = _compact_text("".join(word.text for word in aligned_words))
    if source_text != aligned_words_text and source_text != _compact_text(aligned_text):
        return aligned_words

    projected_words: list[Word] = []
    aligned_index = 0
    for source_index, source_word in enumerate(source_words):
        target_text = _compact_text(source_word.text)
        if not target_text:
            return aligned_words

        pieces: list[Word] = []
        collected_text = ""
        while aligned_index < len(aligned_words) and len(collected_text) < len(target_text):
            piece = aligned_words[aligned_index]
            pieces.append(piece)
            collected_text += _compact_text(piece.text)
            aligned_index += 1

        if collected_text != target_text:
            return aligned_words

        previous_source_word = (
            source_words[source_index - 1] if source_index > 0 else None
        )
        next_source_word = (
            source_words[source_index + 1]
            if source_index + 1 < len(source_words)
            else None
        )
        pieces_tuple = tuple(pieces)
        projected_words.append(
            Word(
                text=source_word.text,
                time_range=_projected_time_range(
                    source_word,
                    pieces_tuple,
                    previous_source_word,
                    next_source_word,
                ),
                confidence=_mean_confidence(pieces_tuple, source_word.confidence),
                speaker_id=_common_word_speaker_id(pieces_tuple)
                or source_word.speaker_id,
            )
        )

    if aligned_index != len(aligned_words):
        return aligned_words

    return tuple(projected_words)


def _compact_text(text: str) -> str:
    return "".join(str(text).split())


def _projected_time_range(
    source_word: Word,
    aligned_pieces: tuple[Word, ...],
    previous_source_word: Word | None,
    next_source_word: Word | None,
) -> TimeRange:
    if not aligned_pieces:
        return source_word.time_range

    aligned_time_range = TimeRange(
        aligned_pieces[0].time_range.start_seconds,
        aligned_pieces[-1].time_range.end_seconds,
    )
    if _looks_like_pause_swallowing(
        source_word,
        aligned_time_range,
        previous_source_word,
        next_source_word,
    ):
        return source_word.time_range

    return aligned_time_range


def _looks_like_pause_swallowing(
    source_word: Word,
    aligned_time_range: TimeRange,
    previous_source_word: Word | None,
    next_source_word: Word | None,
) -> bool:
    source_duration = source_word.time_range.duration_seconds
    aligned_duration = aligned_time_range.duration_seconds
    if aligned_duration > max(source_duration * 2.5, source_duration + 0.4):
        return True

    if previous_source_word is not None:
        previous_gap = (
            source_word.time_range.start_seconds
            - previous_source_word.time_range.end_seconds
        )
        start_drift = abs(
            aligned_time_range.start_seconds - source_word.time_range.start_seconds
        )
        if previous_gap >= 0.4 and start_drift > 0.25:
            return True

    if next_source_word is not None:
        next_gap = (
            next_source_word.time_range.start_seconds
            - source_word.time_range.end_seconds
        )
        if (
            next_gap >= 0.4
            and aligned_time_range.end_seconds
            > next_source_word.time_range.start_seconds - 0.05
        ):
            return True

    return False


def _mean_confidence(
    aligned_pieces: tuple[Word, ...],
    fallback: float | None,
) -> float | None:
    confidences = tuple(
        word.confidence for word in aligned_pieces if word.confidence is not None
    )
    if not confidences:
        return fallback

    return sum(confidences) / len(confidences)


def _common_word_speaker_id(words: tuple[Word, ...]) -> str | None:
    speaker_ids = tuple(
        dict.fromkeys(word.speaker_id for word in words if word.speaker_id is not None)
    )
    if len(speaker_ids) == 1:
        return speaker_ids[0]

    return None


__all__ = [
    "DEFAULT_WHISPERX_LANGUAGE",
    "PassthroughWhisperXAligner",
    "WhisperXAlignerAdapter",
    "WhisperXDependencyError",
]

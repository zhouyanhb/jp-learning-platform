"""Acoustic sentence boundary detection and resolution adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from jp_learning_platform.domain import (
    Segment,
    Sentence,
    SentenceBoundaryCandidate,
    TimeRange,
    Word,
)
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG,
    DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG,
)
from jp_learning_platform.workflow.sentence_boundary_stage import (
    SentenceBoundaryDetection,
    SentenceBoundaryDetectionRequest,
    SentenceBoundaryResolution,
    SentenceBoundaryResolutionRequest,
)

DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS = (
    DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG.min_pause_seconds
)
DEFAULT_SENTENCE_BOUNDARY_MIN_VAD_SILENCE_SECONDS = (
    DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG.min_vad_silence_seconds
)
DEFAULT_SENTENCE_BOUNDARY_TARGET_SAMPLE_RATE = (
    DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG.target_sample_rate
)
DEFAULT_SENTENCE_BOUNDARY_FRAME_SECONDS = (
    DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG.frame_seconds
)
DEFAULT_SENTENCE_BOUNDARY_ENERGY_THRESHOLD_RATIO = (
    DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG.energy_threshold_ratio
)
DEFAULT_SENTENCE_BOUNDARY_MIN_ACOUSTIC_SCORE = (
    DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG.min_acoustic_score
)
DEFAULT_SENTENCE_BOUNDARY_MIN_SENTENCE_SECONDS = (
    DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG.min_sentence_seconds
)
DEFAULT_SENTENCE_BOUNDARY_MIN_SENTENCE_CHARS = (
    DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG.min_sentence_chars
)

_JAPANESE_TERMINAL_MARKS = ("。", "？", "！", "?", "!")
_JAPANESE_BOUNDARY_TRAILING_MARKS = (*_JAPANESE_TERMINAL_MARKS, "、", ",")
_BOUNDARY_CONTINUING_PARTICLE_TYPES = (
    "格助詞",
    "係助詞",
    "副助詞",
    "接続助詞",
    "準体助詞",
)
_BOUNDARY_COMPLETE_PARTICLE_TYPES = ("終助詞",)
_FALLBACK_CONTINUING_PARTICLES = (
    "は",
    "も",
    "が",
    "を",
    "に",
    "へ",
    "で",
    "と",
    "から",
    "より",
    "まで",
    "なら",
    "ならば",
    "ので",
    "のに",
    "けど",
    "けれど",
    "ても",
    "でも",
    "ながら",
    "たり",
)
_FALLBACK_CONNECTIVE_ENDINGS = (
    "たら",
    "だら",
    "れば",
    "ても",
    "でも",
)
_NONBREAKING_CONNECTION_WORD_PAIRS = frozenset(
    {
        ("これ", "から"),
        ("それ", "から"),
        ("あれ", "から"),
        ("ここ", "から"),
        ("そこ", "から"),
        ("あそこ", "から"),
        ("それ", "では"),
        ("それ", "でも"),
        ("それ", "なら"),
        ("これ", "では"),
        ("これ", "でも"),
        ("この", "後"),
        ("その", "後"),
        ("あの", "後"),
        ("この", "あと"),
        ("その", "あと"),
        ("あの", "あと"),
        ("この", "ため"),
        ("その", "ため"),
        ("あの", "ため"),
        ("この", "うえ"),
        ("その", "うえ"),
    }
)


class TorchVadDependencyError(RuntimeError):
    """Raised when torch/torchaudio VAD dependencies are unavailable."""

    def __init__(self) -> None:
        super().__init__(
            "torch and torchaudio are required for acoustic sentence boundary "
            "detection. Install them with: python -m pip install -r requirements.txt"
        )


@dataclass(frozen=True, slots=True)
class WordGapSentenceBoundaryDetector:
    """Generate candidate sentence boundaries from aligned word gaps."""

    min_pause_seconds: float = DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS
    source: str = "word-gap"

    def detect(
        self,
        request: SentenceBoundaryDetectionRequest,
    ) -> SentenceBoundaryDetection:
        if not isinstance(request, SentenceBoundaryDetectionRequest):
            raise TypeError("request must be a SentenceBoundaryDetectionRequest.")

        return SentenceBoundaryDetection(
            source_path=request.source_path,
            candidates=tuple(
                _word_gap_candidates(
                    request.segments,
                    min_pause_seconds=self.min_pause_seconds,
                    source=self.source,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class TorchVadSentenceBoundaryDetector:
    """Use torch waveform energy to confirm pause candidates near word gaps."""

    min_pause_seconds: float = DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS
    min_vad_silence_seconds: float = (
        DEFAULT_SENTENCE_BOUNDARY_MIN_VAD_SILENCE_SECONDS
    )
    target_sample_rate: int = DEFAULT_SENTENCE_BOUNDARY_TARGET_SAMPLE_RATE
    frame_seconds: float = DEFAULT_SENTENCE_BOUNDARY_FRAME_SECONDS
    energy_threshold_ratio: float = (
        DEFAULT_SENTENCE_BOUNDARY_ENERGY_THRESHOLD_RATIO
    )
    fallback_to_word_gap: bool = True

    def detect(
        self,
        request: SentenceBoundaryDetectionRequest,
    ) -> SentenceBoundaryDetection:
        if not isinstance(request, SentenceBoundaryDetectionRequest):
            raise TypeError("request must be a SentenceBoundaryDetectionRequest.")

        word_gap_candidates = tuple(
            _word_gap_candidates(
                request.segments,
                min_pause_seconds=self.min_pause_seconds,
                source="word-gap-fallback",
            )
        )
        if not word_gap_candidates:
            return SentenceBoundaryDetection(source_path=request.source_path)

        try:
            waveform, sample_rate = self._load_waveform(request.source_path)
            speech_rms = _speech_rms(waveform, sample_rate, request.segments)
        except Exception as error:
            if self.fallback_to_word_gap:
                return SentenceBoundaryDetection(
                    source_path=request.source_path,
                    candidates=word_gap_candidates,
                )

            if isinstance(error, ImportError):
                raise TorchVadDependencyError() from error
            raise

        threshold = speech_rms * self.energy_threshold_ratio
        frame_samples = max(1, int(self.frame_seconds * sample_rate))
        candidates: list[SentenceBoundaryCandidate] = []
        for candidate in word_gap_candidates:
            vad_silence_seconds = _low_energy_seconds(
                waveform,
                sample_rate,
                candidate.pause_time_range,
                threshold,
                frame_samples,
            )
            if vad_silence_seconds < self.min_vad_silence_seconds:
                continue

            pause_seconds = candidate.pause_time_range.duration_seconds
            acoustic_score = min(
                1.0,
                0.4 * _bounded_ratio(pause_seconds, self.min_pause_seconds)
                + 0.6
                * _bounded_ratio(
                    vad_silence_seconds,
                    self.min_vad_silence_seconds,
                ),
            )
            candidates.append(
                SentenceBoundaryCandidate(
                    segment_position=candidate.segment_position,
                    after_word_index=candidate.after_word_index,
                    boundary_time_seconds=candidate.boundary_time_seconds,
                    pause_time_range=candidate.pause_time_range,
                    acoustic_score=acoustic_score,
                    source="torch-energy-vad",
                )
            )

        return SentenceBoundaryDetection(
            source_path=request.source_path,
            candidates=tuple(candidates),
        )

    def _load_waveform(self, source_path: Path) -> tuple[Any, int]:
        try:
            import torch
            import torchaudio
        except ImportError as error:
            raise error

        waveform, sample_rate = torchaudio.load(str(source_path))
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
            sample_rate = self.target_sample_rate

        return waveform.to(dtype=torch.float32), int(sample_rate)


@dataclass(slots=True)
class JapaneseBoundaryCompletenessAnalyzer:
    """Decide whether the left side of an acoustic pause is sentence-complete."""

    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _mode: Any | None = field(default=None, init=False, repr=False)
    _sudachi_unavailable: bool = field(default=False, init=False, repr=False)

    def is_complete_left_boundary(self, words: tuple[Word, ...]) -> bool:
        raw_text = _words_text(words)
        if _ends_with_terminal_mark(raw_text):
            return True

        text = _strip_boundary_trailing_marks(raw_text)
        if not text:
            return False

        morphemes = self._tokenize(text)
        if morphemes:
            return _morphemes_end_as_complete_sentence(morphemes)

        return _fallback_words_end_as_complete_sentence(words)

    def _tokenize(self, text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if self._sudachi_unavailable:
            return ()

        try:
            tokenizer, mode = self._load_tokenizer()
            return tuple(
                (str(morpheme.surface()), tuple(morpheme.part_of_speech()))
                for morpheme in tokenizer.tokenize(text, mode)
            )
        except Exception:
            self._sudachi_unavailable = True
            return ()

    def _load_tokenizer(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._mode is None:
            from sudachipy import dictionary
            from sudachipy import tokenizer

            self._tokenizer = dictionary.Dictionary().create()
            self._mode = tokenizer.Tokenizer.SplitMode.C

        return self._tokenizer, self._mode


@dataclass(frozen=True, slots=True)
class AcousticSentenceBoundaryResolver:
    """Apply acoustic boundary candidates to repaired segment text."""

    min_acoustic_score: float = DEFAULT_SENTENCE_BOUNDARY_MIN_ACOUSTIC_SCORE
    min_sentence_seconds: float = DEFAULT_SENTENCE_BOUNDARY_MIN_SENTENCE_SECONDS
    min_sentence_chars: int = DEFAULT_SENTENCE_BOUNDARY_MIN_SENTENCE_CHARS
    boundary_analyzer: JapaneseBoundaryCompletenessAnalyzer = field(
        default_factory=JapaneseBoundaryCompletenessAnalyzer,
    )

    def resolve(
        self,
        request: SentenceBoundaryResolutionRequest,
    ) -> SentenceBoundaryResolution:
        if not isinstance(request, SentenceBoundaryResolutionRequest):
            raise TypeError("request must be a SentenceBoundaryResolutionRequest.")

        candidates_by_segment = _candidates_by_segment(
            request.candidates,
            min_acoustic_score=self.min_acoustic_score,
        )
        return SentenceBoundaryResolution(
            source_path=request.source_path,
            segments=tuple(
                self._resolve_segment(
                    segment,
                    candidates_by_segment.get(segment.position, ()),
                )
                for segment in request.segments
            ),
        )

    def _resolve_segment(
        self,
        segment: Segment,
        candidates: tuple[SentenceBoundaryCandidate, ...],
    ) -> Segment:
        words = _segment_words(segment)
        if len(words) < 2 or not candidates:
            return segment

        boundaries = self._valid_boundaries(words, candidates)
        if not boundaries:
            return segment

        word_chunks = _word_chunks(words, boundaries)
        fallback_texts = tuple(_words_text(chunk) for chunk in word_chunks)
        text_chunks = (
            _split_text_by_terminal_marks(segment.text, len(word_chunks))
            or _split_text_by_spaces(segment.text, len(word_chunks))
            or fallback_texts
        )
        sentences = tuple(
            Sentence(
                text=text,
                time_range=_time_range_for_words(chunk),
                words=tuple(chunk),
                speaker_id=_common_speaker_id(chunk) or segment.speaker_id,
            )
            for text, chunk in zip(text_chunks, word_chunks, strict=True)
        )
        return Segment(
            position=segment.position,
            text=" ".join(sentence.text for sentence in sentences),
            time_range=segment.time_range,
            sentences=sentences,
            speaker_id=segment.speaker_id,
        )

    def _valid_boundaries(
        self,
        words: tuple[Word, ...],
        candidates: tuple[SentenceBoundaryCandidate, ...],
    ) -> tuple[int, ...]:
        boundaries: list[int] = []
        previous_boundary = -1
        for candidate in sorted(
            candidates,
            key=lambda item: (item.boundary_time_seconds, -item.acoustic_score),
        ):
            boundary = _boundary_index_for_time(
                words,
                candidate.boundary_time_seconds,
            )
            if boundary is None:
                continue

            if boundary <= previous_boundary or boundary >= len(words) - 1:
                continue

            if _splits_nonbreaking_connection(words, boundary):
                continue

            left = words[previous_boundary + 1 : boundary + 1]
            right = words[boundary + 1 :]
            if not self._valid_chunk(left) or not self._valid_chunk(right):
                continue

            if not self.boundary_analyzer.is_complete_left_boundary(left):
                continue

            boundaries.append(boundary)
            previous_boundary = boundary

        return tuple(boundaries)

    def _valid_chunk(self, words: tuple[Word, ...]) -> bool:
        if not words:
            return False

        time_range = _time_range_for_words(words)
        duration_seconds = time_range.duration_seconds
        return (
            duration_seconds >= self.min_sentence_seconds
            and len(_words_text(words)) >= self.min_sentence_chars
        )


def _word_gap_candidates(
    segments: Iterable[Segment],
    min_pause_seconds: float,
    source: str,
) -> Iterable[SentenceBoundaryCandidate]:
    for segment in segments:
        words = _segment_words(segment)
        for index, (current_word, next_word) in enumerate(
            zip(words, words[1:], strict=False)
        ):
            pause_start = current_word.time_range.end_seconds
            pause_end = next_word.time_range.start_seconds
            pause_seconds = pause_end - pause_start
            if pause_seconds < min_pause_seconds:
                continue

            yield SentenceBoundaryCandidate(
                segment_position=segment.position,
                after_word_index=index,
                boundary_time_seconds=(pause_start + pause_end) / 2,
                pause_time_range=TimeRange(pause_start, pause_end),
                acoustic_score=_bounded_ratio(pause_seconds, min_pause_seconds),
                source=source,
            )


def _bounded_ratio(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0

    return max(0.0, min(1.0, value / denominator))


def _segment_words(segment: Segment) -> tuple[Word, ...]:
    return tuple(word for sentence in segment.sentences for word in sentence.words)


def _speech_rms(waveform: Any, sample_rate: int, segments: tuple[Segment, ...]) -> float:
    energy = 0.0
    samples = 0
    for segment in segments:
        for word in _segment_words(segment):
            chunk = _waveform_slice(waveform, sample_rate, word.time_range)
            sample_count = int(chunk.numel())
            if sample_count == 0:
                continue

            energy += float(chunk.square().sum().item())
            samples += sample_count

    if samples == 0:
        sample_count = int(waveform.numel())
        if sample_count == 0:
            return 0.0

        return math.sqrt(float(waveform.square().sum().item()) / sample_count)

    return math.sqrt(energy / samples)


def _low_energy_seconds(
    waveform: Any,
    sample_rate: int,
    time_range: TimeRange,
    threshold: float,
    frame_samples: int,
) -> float:
    chunk = _waveform_slice(waveform, sample_rate, time_range)
    sample_count = int(chunk.numel())
    if sample_count == 0:
        return 0.0

    low_energy_samples = 0
    for start in range(0, sample_count, frame_samples):
        frame = chunk[start : start + frame_samples]
        frame_count = int(frame.numel())
        if frame_count == 0:
            continue

        frame_rms = math.sqrt(float(frame.square().mean().item()))
        if frame_rms <= threshold:
            low_energy_samples += frame_count

    return low_energy_samples / sample_rate


def _waveform_slice(waveform: Any, sample_rate: int, time_range: TimeRange) -> Any:
    start_sample = max(0, int(time_range.start_seconds * sample_rate))
    end_sample = max(start_sample, int(time_range.end_seconds * sample_rate))
    return waveform[start_sample:end_sample]


def _candidates_by_segment(
    candidates: Iterable[SentenceBoundaryCandidate],
    min_acoustic_score: float,
) -> dict[int, tuple[SentenceBoundaryCandidate, ...]]:
    grouped: dict[int, list[SentenceBoundaryCandidate]] = {}
    for candidate in candidates:
        if candidate.acoustic_score < min_acoustic_score:
            continue

        grouped.setdefault(candidate.segment_position, []).append(candidate)

    return {
        segment_position: tuple(candidates)
        for segment_position, candidates in grouped.items()
    }


def _word_chunks(
    words: tuple[Word, ...],
    boundaries: tuple[int, ...],
) -> tuple[tuple[Word, ...], ...]:
    chunks: list[tuple[Word, ...]] = []
    start = 0
    for boundary in boundaries:
        chunks.append(words[start : boundary + 1])
        start = boundary + 1

    chunks.append(words[start:])
    return tuple(chunks)


def _time_range_for_words(words: tuple[Word, ...]) -> TimeRange:
    return TimeRange(
        min(word.time_range.start_seconds for word in words),
        max(word.time_range.end_seconds for word in words),
    )


def _boundary_index_for_time(
    words: tuple[Word, ...],
    boundary_time_seconds: float,
) -> int | None:
    for index, (current_word, next_word) in enumerate(
        zip(words, words[1:], strict=False)
    ):
        if (
            current_word.time_range.end_seconds
            <= boundary_time_seconds
            <= next_word.time_range.start_seconds
        ):
            return index

    before_boundary = tuple(
        index
        for index, word in enumerate(words[:-1])
        if word.time_range.end_seconds <= boundary_time_seconds
    )
    if not before_boundary:
        return None

    return before_boundary[-1]


def _split_text_by_terminal_marks(text: str, expected_count: int) -> tuple[str, ...]:
    chunks: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character not in _JAPANESE_TERMINAL_MARKS:
            continue

        chunk = text[start : index + 1].strip()
        if chunk:
            chunks.append(chunk)
        start = index + 1

    tail = text[start:].strip()
    if tail:
        chunks.append(tail)

    if len(chunks) != expected_count:
        return ()

    return tuple(chunks)


def _split_text_by_spaces(text: str, expected_count: int) -> tuple[str, ...]:
    chunks = tuple(chunk for chunk in text.split() if chunk)
    if len(chunks) != expected_count:
        return ()

    return chunks


def _words_text(words: tuple[Word, ...]) -> str:
    return "".join(word.text for word in words)


def _ends_with_terminal_mark(text: str) -> bool:
    return text.strip().endswith(_JAPANESE_TERMINAL_MARKS)


def _morphemes_end_as_complete_sentence(
    morphemes: tuple[tuple[str, tuple[str, ...]], ...],
) -> bool:
    significant = tuple(
        (surface, pos)
        for surface, pos in morphemes
        if _strip_boundary_trailing_marks(surface)
    )
    if not significant:
        return False

    _, pos = significant[-1]
    primary_pos = pos[0] if len(pos) > 0 else ""
    sub_pos = pos[1] if len(pos) > 1 else ""
    if primary_pos == "助詞":
        if sub_pos in _BOUNDARY_COMPLETE_PARTICLE_TYPES:
            return True

        if sub_pos in _BOUNDARY_CONTINUING_PARTICLE_TYPES:
            return False

        return False

    if primary_pos == "接続詞":
        return False

    if primary_pos in ("動詞", "形容詞", "助動詞"):
        return not _morpheme_has_connective_form(pos)

    return True


def _morpheme_has_connective_form(pos: tuple[str, ...]) -> bool:
    conjugation_form = pos[5] if len(pos) > 5 else ""
    return any(
        marker in conjugation_form
        for marker in ("仮定形", "連用形", "接続")
    )


def _fallback_words_end_as_complete_sentence(words: tuple[Word, ...]) -> bool:
    significant_words = tuple(
        word for word in words if _strip_boundary_trailing_marks(word.text)
    )
    if not significant_words:
        return False

    last_text = _strip_boundary_trailing_marks(significant_words[-1].text)
    if not last_text:
        return False

    if last_text in _FALLBACK_CONTINUING_PARTICLES:
        return False

    if last_text.endswith(_FALLBACK_CONNECTIVE_ENDINGS):
        return False

    return True


def _splits_nonbreaking_connection(words: tuple[Word, ...], boundary: int) -> bool:
    if boundary < 0 or boundary >= len(words) - 1:
        return False

    left_text = _normalize_connection_boundary_word(words[boundary].text)
    right_text = _normalize_connection_boundary_word(words[boundary + 1].text)
    if not left_text or not right_text:
        return False

    return (left_text, right_text) in _NONBREAKING_CONNECTION_WORD_PAIRS


def _normalize_connection_boundary_word(text: str) -> str:
    return _strip_boundary_trailing_marks(text).strip()


def _strip_boundary_trailing_marks(text: str) -> str:
    return text.strip().rstrip("".join(_JAPANESE_BOUNDARY_TRAILING_MARKS))


def _common_speaker_id(words: tuple[Word, ...]) -> str | None:
    speaker_ids = tuple(
        dict.fromkeys(word.speaker_id for word in words if word.speaker_id is not None)
    )
    if len(speaker_ids) == 1:
        return speaker_ids[0]

    return None


__all__ = [
    "AcousticSentenceBoundaryResolver",
    "DEFAULT_SENTENCE_BOUNDARY_ENERGY_THRESHOLD_RATIO",
    "DEFAULT_SENTENCE_BOUNDARY_FRAME_SECONDS",
    "DEFAULT_SENTENCE_BOUNDARY_MIN_ACOUSTIC_SCORE",
    "DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS",
    "DEFAULT_SENTENCE_BOUNDARY_MIN_SENTENCE_CHARS",
    "DEFAULT_SENTENCE_BOUNDARY_MIN_SENTENCE_SECONDS",
    "DEFAULT_SENTENCE_BOUNDARY_MIN_VAD_SILENCE_SECONDS",
    "DEFAULT_SENTENCE_BOUNDARY_TARGET_SAMPLE_RATE",
    "TorchVadDependencyError",
    "TorchVadSentenceBoundaryDetector",
    "WordGapSentenceBoundaryDetector",
]

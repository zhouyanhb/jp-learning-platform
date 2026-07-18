"""pyannote.audio speaker diarization adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_PYANNOTE_DIARIZATION_CONFIG,
)
from jp_learning_platform.workflow.whisperx_alignment_stage import (
    WhisperXAligner,
    WhisperXAlignment,
    WhisperXAlignmentRequest,
)

DEFAULT_PYANNOTE_DIARIZATION_MODEL = (
    DEFAULT_PYANNOTE_DIARIZATION_CONFIG.model_name
)
DEFAULT_HF_TOKEN_ENVIRONMENT_VARIABLE = (
    DEFAULT_PYANNOTE_DIARIZATION_CONFIG.token_environment_variable
)


class PyannoteDependencyError(RuntimeError):
    """Raised when pyannote.audio is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "pyannote.audio is required for speaker diarization. "
            "Install it with: python -m pip install -e '.[diarization]'"
        )


class PyannoteAuthTokenError(RuntimeError):
    """Raised when pyannote diarization needs a Hugging Face token."""

    def __init__(self, environment_variable: str) -> None:
        super().__init__(
            "pyannote.audio speaker diarization requires a Hugging Face token. "
            f"Pass --hf-token or set {environment_variable}."
        )


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """One speaker-labeled time interval from diarization output."""

    speaker_id: str
    time_range: TimeRange

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str):
            raise TypeError("speaker_id must be a string.")

        speaker_id = self.speaker_id.strip()
        if not speaker_id:
            raise ValueError("speaker_id must not be empty.")

        if not isinstance(self.time_range, TimeRange):
            raise TypeError("time_range must be a TimeRange.")

        object.__setattr__(self, "speaker_id", speaker_id)


@dataclass(slots=True)
class PyannoteSpeakerDiarizer:
    """Assign pyannote speaker labels to aligned domain segments."""

    model_name: str = DEFAULT_PYANNOTE_DIARIZATION_MODEL
    auth_token: str | None = None
    token_environment_variable: str = DEFAULT_HF_TOKEN_ENVIRONMENT_VARIABLE
    _pipeline: Any | None = field(default=None, init=False, repr=False)

    def diarize(self, source_path: Path) -> tuple[SpeakerTurn, ...]:
        pipeline = self._load_pipeline()
        diarization = pipeline(str(Path(source_path)))
        return _speaker_turns_from_diarization(diarization)

    def assign_speakers(
        self,
        source_path: Path,
        segments: tuple[Segment, ...],
    ) -> tuple[Segment, ...]:
        turns = self.diarize(source_path)
        if not turns:
            return segments

        assigned_segments: list[Segment] = []
        for segment in segments:
            assigned_segments.extend(_speaker_segments(segment, turns))

        return tuple(
            Segment(
                position=position,
                text=segment.text,
                time_range=segment.time_range,
                sentences=segment.sentences,
                speaker_id=segment.speaker_id,
            )
            for position, segment in enumerate(assigned_segments)
        )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            token = self._auth_token()
            if token is None:
                raise PyannoteAuthTokenError(self.token_environment_variable)

            try:
                from pyannote.audio import Pipeline
            except ImportError as error:
                raise PyannoteDependencyError() from error

            self._pipeline = _load_pyannote_pipeline(
                Pipeline,
                model_name=self.model_name,
                token=token,
            )

        return self._pipeline

    def _auth_token(self) -> str | None:
        token = self.auth_token
        if token is None:
            token = os.getenv(self.token_environment_variable)

        if token is None:
            return None

        normalized = token.strip()
        return normalized or None


@dataclass(frozen=True, slots=True)
class DiarizingWhisperXAligner:
    """Wrap an aligner and apply pyannote speaker labels to its segments."""

    base_aligner: WhisperXAligner
    diarizer: PyannoteSpeakerDiarizer

    def __post_init__(self) -> None:
        if not callable(getattr(self.base_aligner, "align", None)):
            raise TypeError("base_aligner must define a callable align method.")

        if not isinstance(self.diarizer, PyannoteSpeakerDiarizer):
            raise TypeError("diarizer must be a PyannoteSpeakerDiarizer.")

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        alignment = self.base_aligner.align(request)
        if not isinstance(alignment, WhisperXAlignment):
            raise TypeError("base_aligner must return a WhisperXAlignment.")

        return WhisperXAlignment(
            source_path=alignment.source_path,
            segments=self.diarizer.assign_speakers(
                alignment.source_path,
                alignment.segments,
            ),
        )


def _load_pyannote_pipeline(
    pipeline_class: Any,
    model_name: str,
    token: str,
) -> Any:
    try:
        return pipeline_class.from_pretrained(model_name, token=token)
    except TypeError:
        return pipeline_class.from_pretrained(model_name, use_auth_token=token)


def _speaker_turns_from_diarization(diarization: Any) -> tuple[SpeakerTurn, ...]:
    annotation = getattr(diarization, "speaker_diarization", diarization)
    turns: list[SpeakerTurn] = []
    for turn, speaker in _iter_speaker_records(annotation):
        speaker_id = str(speaker).strip()
        if not speaker_id:
            continue

        turns.append(
            SpeakerTurn(
                speaker_id=speaker_id,
                time_range=TimeRange(
                    start_seconds=float(turn.start),
                    end_seconds=float(turn.end),
                ),
            )
        )

    return tuple(turns)


def _iter_speaker_records(annotation: Any) -> Any:
    if callable(getattr(annotation, "itertracks", None)):
        for turn, _track, speaker in annotation.itertracks(yield_label=True):
            yield turn, speaker
        return

    yield from annotation


def _speaker_segments(
    segment: Segment,
    turns: tuple[SpeakerTurn, ...],
) -> list[Segment]:
    sentences = segment.sentences or (
        Sentence(
            text=segment.text,
            time_range=segment.time_range,
            words=(),
            speaker_id=segment.speaker_id,
        ),
    )
    segments: list[Segment] = []
    for sentence in sentences:
        assigned_words = tuple(_assign_word(word, turns) for word in sentence.words)
        if assigned_words:
            for speaker_id, words in _speaker_word_runs(assigned_words):
                text = (
                    sentence.text
                    if len(words) == len(assigned_words)
                    else _word_text(words)
                )
                time_range = TimeRange(
                    words[0].time_range.start_seconds,
                    words[-1].time_range.end_seconds,
                )
                assigned_sentence = Sentence(
                    text=text,
                    time_range=time_range,
                    words=words,
                    speaker_id=speaker_id,
                )
                segments.append(
                    Segment(
                        position=0,
                        text=text,
                        time_range=time_range,
                        sentences=(assigned_sentence,),
                        speaker_id=speaker_id,
                    )
                )
            continue

        speaker_id = (
            _speaker_for_time_range(sentence.time_range, turns)
            or sentence.speaker_id
            or segment.speaker_id
        )
        assigned_sentence = Sentence(
            text=sentence.text,
            time_range=sentence.time_range,
            words=(),
            speaker_id=speaker_id,
        )
        segments.append(
            Segment(
                position=0,
                text=sentence.text,
                time_range=sentence.time_range,
                sentences=(assigned_sentence,),
                speaker_id=speaker_id,
            )
        )

    return segments


def _assign_word(word: Word, turns: tuple[SpeakerTurn, ...]) -> Word:
    return Word(
        text=word.text,
        time_range=word.time_range,
        confidence=word.confidence,
        speaker_id=_speaker_for_time_range(word.time_range, turns) or word.speaker_id,
    )


def _speaker_word_runs(
    words: tuple[Word, ...],
) -> tuple[tuple[str | None, tuple[Word, ...]], ...]:
    if not words:
        return ()

    runs: list[tuple[str | None, tuple[Word, ...]]] = []
    current_speaker_id = words[0].speaker_id
    current_words: list[Word] = []
    for word in words:
        if word.speaker_id != current_speaker_id and current_words:
            runs.append((current_speaker_id, tuple(current_words)))
            current_words = []
            current_speaker_id = word.speaker_id

        current_words.append(word)

    if current_words:
        runs.append((current_speaker_id, tuple(current_words)))

    return tuple(runs)


def _speaker_for_time_range(
    time_range: TimeRange,
    turns: tuple[SpeakerTurn, ...],
) -> str | None:
    overlap_by_speaker: dict[str, float] = {}
    for turn in turns:
        overlap_seconds = _overlap_seconds(time_range, turn.time_range)
        if overlap_seconds <= 0:
            continue

        overlap_by_speaker[turn.speaker_id] = (
            overlap_by_speaker.get(turn.speaker_id, 0.0) + overlap_seconds
        )

    if not overlap_by_speaker:
        return None

    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def _overlap_seconds(left: TimeRange, right: TimeRange) -> float:
    return max(
        0.0,
        min(left.end_seconds, right.end_seconds)
        - max(left.start_seconds, right.start_seconds),
    )


def _word_text(words: tuple[Word, ...]) -> str:
    return "".join(word.text for word in words).strip()


__all__ = [
    "DEFAULT_HF_TOKEN_ENVIRONMENT_VARIABLE",
    "DEFAULT_PYANNOTE_DIARIZATION_MODEL",
    "DiarizingWhisperXAligner",
    "PyannoteAuthTokenError",
    "PyannoteDependencyError",
    "PyannoteSpeakerDiarizer",
    "SpeakerTurn",
]

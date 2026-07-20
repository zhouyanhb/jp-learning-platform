"""ReazonSpeech NeMo transcription adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter, signature
from math import isfinite
from pathlib import Path
from typing import Any
from unicodedata import category

from jp_learning_platform.domain import Segment, Sentence, TimeRange
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG,
)
from jp_learning_platform.workflow.whisper_stage import (
    WhisperTranscript,
    WhisperTranscriptionRequest,
)

DEFAULT_REAZON_SPEECH_MODEL_ID = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.reazon_model_id
)
DEFAULT_REAZON_SPEECH_DEVICE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.device
DEFAULT_REAZON_SPEECH_BATCH_SIZE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.batch_size
DEFAULT_REAZON_SPEECH_CHUNK_LENGTH_SECONDS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.chunk_length_seconds
)
DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.chunk_overlap_seconds
)

_REAZON_MODEL_ALIASES = {
    "reazon-speech": DEFAULT_REAZON_SPEECH_MODEL_ID,
    "reazonspeech": DEFAULT_REAZON_SPEECH_MODEL_ID,
    "reazonspeech-nemo-v2": DEFAULT_REAZON_SPEECH_MODEL_ID,
    "reazon-nemo-v2": DEFAULT_REAZON_SPEECH_MODEL_ID,
    "nemo-v2": DEFAULT_REAZON_SPEECH_MODEL_ID,
    DEFAULT_REAZON_SPEECH_MODEL_ID: DEFAULT_REAZON_SPEECH_MODEL_ID,
}


class ReazonSpeechDependencyError(RuntimeError):
    """Raised when ReazonSpeech NeMo runtime support is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "ReazonSpeech NeMo ASR support is required for reazon-speech "
            "transcription. Install the official runtime with: "
            "python -m pip install '"
            "git+https://github.com/reazon-research/ReazonSpeech.git"
            "#subdirectory=pkg/nemo-asr'"
        )


@dataclass(frozen=True, slots=True)
class _AudioChunk:
    audio: Any
    start_seconds: float
    end_seconds: float
    usable_start_seconds: float
    usable_end_seconds: float

    @property
    def duration_seconds(self) -> float | None:
        duration = self.end_seconds - self.start_seconds
        if not isfinite(duration):
            return None

        return max(duration, 0.0)


@dataclass(slots=True)
class ReazonSpeechTranscriber:
    """Transcribe Japanese audio with reazon-research/reazonspeech-nemo-v2."""

    model_id: str = DEFAULT_REAZON_SPEECH_MODEL_ID
    device: str = DEFAULT_REAZON_SPEECH_DEVICE
    batch_size: int = DEFAULT_REAZON_SPEECH_BATCH_SIZE
    chunk_length_seconds: float = DEFAULT_REAZON_SPEECH_CHUNK_LENGTH_SECONDS
    chunk_overlap_seconds: float = DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS
    verbose: bool = False
    _model: Any | None = field(default=None, init=False, repr=False)
    _backend: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_id = _normalize_model_id(self.model_id)
        self.device = _normalize_non_empty_text(self.device, "device")
        self.batch_size = _normalize_positive_int(self.batch_size, "batch_size")
        self.chunk_length_seconds = _normalize_positive_float(
            self.chunk_length_seconds,
            "chunk_length_seconds",
        )
        self.chunk_overlap_seconds = _normalize_non_negative_float(
            self.chunk_overlap_seconds,
            "chunk_overlap_seconds",
        )
        if self.chunk_overlap_seconds >= self.chunk_length_seconds:
            raise ValueError(
                "chunk_overlap_seconds must be smaller than chunk_length_seconds."
            )
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a bool.")

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        if not isinstance(request, WhisperTranscriptionRequest):
            raise TypeError("request must be a WhisperTranscriptionRequest.")

        model = self._load_model()
        if self._backend == "reazonspeech":
            segments = self._transcribe_with_reazonspeech(
                model=model,
                source_path=request.source_path,
            )
        else:
            result = self._transcribe_with_nemo(
                model=model,
                source_path=request.source_path,
            )
            segments = _segments_from_result(
                result,
                source_path=request.source_path,
            )

        return WhisperTranscript(
            source_path=request.source_path,
            segments=segments,
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if self.model_id == DEFAULT_REAZON_SPEECH_MODEL_ID:
            try:
                from reazonspeech.nemo.asr import load_model
            except ImportError:
                pass
            else:
                self._model = _load_reazonspeech_model(
                    load_model,
                    device=self.device,
                )
                self._backend = "reazonspeech"
                return self._model

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as error:
            raise ReazonSpeechDependencyError() from error

        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_id,
        )
        self._backend = "nemo"
        return self._model

    def _transcribe_with_reazonspeech(
        self,
        *,
        model: Any,
        source_path: Path,
    ) -> tuple[Segment, ...]:
        try:
            import reazonspeech.nemo.asr as reazon_asr
        except ImportError as error:
            raise ReazonSpeechDependencyError() from error

        audio = reazon_asr.audio_from_path(str(source_path))
        chunks = _audio_chunks(
            reazon_asr=reazon_asr,
            audio=audio,
            chunk_length_seconds=self.chunk_length_seconds,
            chunk_overlap_seconds=self.chunk_overlap_seconds,
        )
        config_class = getattr(reazon_asr, "TranscribeConfig", None)
        config = config_class(verbose=self.verbose) if config_class is not None else None

        segments: list[Segment] = []
        for chunk in chunks:
            result = _call_reazonspeech_transcribe(
                reazon_asr=reazon_asr,
                model=model,
                audio=chunk.audio,
                config=config,
            )
            chunk_segments = _segments_from_result(
                result,
                fallback_duration_seconds=chunk.duration_seconds,
                time_offset_seconds=chunk.start_seconds,
            )
            segments.extend(
                segment
                for segment in (
                    _clip_segment_to_chunk(segment, chunk)
                    for segment in chunk_segments
                )
                if segment is not None and _segment_midpoint_is_usable(segment, chunk)
            )

        return _normalize_reazon_segments(tuple(segments))

    def _transcribe_with_nemo(
        self,
        *,
        model: Any,
        source_path: Path,
    ) -> dict[str, object]:
        output = model.transcribe(
            audio=[str(source_path)],
            batch_size=self.batch_size,
        )
        first_output = _first_transcription_output(output)
        return {"text": _text_from_external(first_output), "segments": ()}


def _normalize_model_id(value: str) -> str:
    model_id = _normalize_non_empty_text(value, "model_id")
    return _REAZON_MODEL_ALIASES.get(model_id, model_id)


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


def _normalize_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")

    return value


def _normalize_positive_float(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")

    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a number.") from error

    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive.")

    return normalized


def _normalize_non_negative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")

    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a number.") from error

    if normalized < 0:
        raise ValueError(f"{field_name} must be non-negative.")

    return normalized


def _load_reazonspeech_model(load_model: Any, *, device: str) -> Any:
    if _callable_accepts_keyword(load_model, "device"):
        return load_model(device=device)

    return load_model()


def _callable_accepts_keyword(function: Any, keyword: str) -> bool:
    try:
        parameters = signature(function).parameters.values()
    except (TypeError, ValueError):
        return False

    for parameter in parameters:
        if parameter.kind == Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True

    return False


def _audio_chunks(
    *,
    reazon_asr: Any,
    audio: Any,
    chunk_length_seconds: float,
    chunk_overlap_seconds: float,
) -> tuple[_AudioChunk, ...]:
    waveform = getattr(audio, "waveform", None)
    sample_rate = _sample_rate(audio)
    if waveform is None or sample_rate is None:
        return (
            _AudioChunk(
                audio=audio,
                start_seconds=0.0,
                end_seconds=float("inf"),
                usable_start_seconds=0.0,
                usable_end_seconds=float("inf"),
            ),
        )

    try:
        num_samples = len(waveform)
    except TypeError:
        return (
            _AudioChunk(
                audio=audio,
                start_seconds=0.0,
                end_seconds=float("inf"),
                usable_start_seconds=0.0,
                usable_end_seconds=float("inf"),
            ),
        )

    if num_samples <= 0:
        return ()

    chunk_samples = max(1, int(round(chunk_length_seconds * sample_rate)))
    overlap_samples = int(round(chunk_overlap_seconds * sample_rate))
    if num_samples <= chunk_samples:
        duration_seconds = float(num_samples) / float(sample_rate)
        return (
            _AudioChunk(
                audio=audio,
                start_seconds=0.0,
                end_seconds=duration_seconds,
                usable_start_seconds=0.0,
                usable_end_seconds=duration_seconds,
            ),
        )

    step_samples = max(1, chunk_samples - overlap_samples)
    raw_chunks: list[tuple[Any, int, int]] = []
    start_sample = 0
    while start_sample < num_samples:
        end_sample = min(start_sample + chunk_samples, num_samples)
        chunk_waveform = waveform[start_sample:end_sample]
        raw_chunks.append(
            (
                _audio_from_waveform(
                    reazon_asr=reazon_asr,
                    source_audio=audio,
                    waveform=chunk_waveform,
                    sample_rate=sample_rate,
                ),
                start_sample,
                end_sample,
            )
        )
        if end_sample >= num_samples:
            break
        start_sample += step_samples

    usable_margin_seconds = chunk_overlap_seconds / 2.0
    chunks: list[_AudioChunk] = []
    for index, (chunk_audio, start_sample, end_sample) in enumerate(raw_chunks):
        start_seconds = float(start_sample) / float(sample_rate)
        end_seconds = float(end_sample) / float(sample_rate)
        chunks.append(
            _AudioChunk(
                audio=chunk_audio,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                usable_start_seconds=(
                    start_seconds if index == 0 else start_seconds + usable_margin_seconds
                ),
                usable_end_seconds=(
                    end_seconds
                    if index == len(raw_chunks) - 1
                    else end_seconds - usable_margin_seconds
                ),
            )
        )

    return tuple(chunks)


def _sample_rate(audio: Any) -> int | None:
    value = getattr(audio, "samplerate", getattr(audio, "sample_rate", None))
    try:
        sample_rate = int(value)
    except (TypeError, ValueError):
        return None

    if sample_rate <= 0:
        return None

    return sample_rate


def _audio_from_waveform(
    *,
    reazon_asr: Any,
    source_audio: Any,
    waveform: Any,
    sample_rate: int,
) -> Any:
    audio_from_numpy = getattr(reazon_asr, "audio_from_numpy", None)
    if callable(audio_from_numpy):
        return audio_from_numpy(waveform, sample_rate)

    return type(source_audio)(waveform, sample_rate)


def _call_reazonspeech_transcribe(
    *,
    reazon_asr: Any,
    model: Any,
    audio: Any,
    config: Any | None,
) -> Any:
    if config is None:
        return reazon_asr.transcribe(model, audio)

    return reazon_asr.transcribe(model, audio, config)


def _clip_segment_to_chunk(segment: Segment, chunk: _AudioChunk) -> Segment | None:
    start_seconds = max(segment.time_range.start_seconds, chunk.start_seconds)
    end_seconds = (
        min(segment.time_range.end_seconds, chunk.end_seconds)
        if isfinite(chunk.end_seconds)
        else segment.time_range.end_seconds
    )
    if end_seconds < start_seconds:
        return None

    if (
        start_seconds == segment.time_range.start_seconds
        and end_seconds == segment.time_range.end_seconds
    ):
        return segment

    return _segment(
        position=segment.position,
        text=segment.text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _segment_midpoint_is_usable(segment: Segment, chunk: _AudioChunk) -> bool:
    midpoint = (
        segment.time_range.start_seconds + segment.time_range.end_seconds
    ) / 2.0
    return chunk.usable_start_seconds <= midpoint <= chunk.usable_end_seconds


def _deduplicate_segments(segments: tuple[Segment, ...]) -> tuple[Segment, ...]:
    accepted: list[Segment] = []
    for segment in sorted(
        segments,
        key=lambda value: (
            value.time_range.start_seconds,
            value.time_range.end_seconds,
            value.text,
        ),
    ):
        if not _has_substantive_text(segment.text):
            continue

        if accepted and _segments_overlap_as_duplicates(accepted[-1], segment):
            if _segment_content_length(segment) > _segment_content_length(
                accepted[-1]
            ):
                accepted[-1] = segment
            continue
        accepted.append(segment)

    return tuple(accepted)


def _segments_overlap_as_duplicates(left: Segment, right: Segment) -> bool:
    left_text = _normalize_text_for_duplicate_check(left.text)
    right_text = _normalize_text_for_duplicate_check(right.text)
    if not left_text or not right_text:
        return False

    if not (
        left_text == right_text
        or left_text in right_text
        or right_text in left_text
    ):
        return False

    overlap_seconds = min(
        left.time_range.end_seconds,
        right.time_range.end_seconds,
    ) - max(left.time_range.start_seconds, right.time_range.start_seconds)
    if overlap_seconds <= 0:
        return False

    shorter_duration = min(
        left.time_range.duration_seconds,
        right.time_range.duration_seconds,
    )
    if shorter_duration <= 0:
        return True

    return overlap_seconds / shorter_duration >= 0.5


def _normalize_reazon_segments(segments: tuple[Segment, ...]) -> tuple[Segment, ...]:
    filtered_segments = tuple(
        segment for segment in segments if _has_substantive_text(segment.text)
    )
    deduplicated_segments = _deduplicate_segments(filtered_segments)
    return _renumber_segments(_resolve_segment_overlaps(deduplicated_segments))


def _resolve_segment_overlaps(segments: tuple[Segment, ...]) -> tuple[Segment, ...]:
    resolved: list[Segment] = []
    for segment in sorted(
        segments,
        key=lambda value: (
            value.time_range.start_seconds,
            value.time_range.end_seconds,
            value.text,
        ),
    ):
        if not resolved:
            resolved.append(segment)
            continue

        previous = resolved[-1]
        if segment.time_range.start_seconds < previous.time_range.end_seconds:
            if _segments_overlap_as_duplicates(previous, segment):
                if _segment_content_length(segment) > _segment_content_length(
                    previous
                ):
                    resolved[-1] = segment
                continue

            previous, segment = _split_segment_overlap(previous, segment)
            resolved[-1] = previous
            if segment.time_range.duration_seconds <= 0:
                continue

        resolved.append(segment)

    return tuple(resolved)


def _split_segment_overlap(left: Segment, right: Segment) -> tuple[Segment, Segment]:
    if right.time_range.end_seconds <= left.time_range.end_seconds:
        if _segment_content_length(right) > _segment_content_length(left):
            return _segment_with_time_range(
                left,
                TimeRange(left.time_range.start_seconds, right.time_range.start_seconds),
            ), right
        return left, _segment_with_time_range(
            right,
            TimeRange(left.time_range.end_seconds, left.time_range.end_seconds),
        )

    boundary_seconds = (
        right.time_range.start_seconds + left.time_range.end_seconds
    ) / 2.0
    boundary_seconds = min(
        right.time_range.end_seconds,
        max(left.time_range.start_seconds, boundary_seconds),
    )
    return (
        _segment_with_time_range(
            left,
            TimeRange(left.time_range.start_seconds, boundary_seconds),
        ),
        _segment_with_time_range(
            right,
            TimeRange(boundary_seconds, right.time_range.end_seconds),
        ),
    )


def _segment_with_time_range(segment: Segment, time_range: TimeRange) -> Segment:
    return _segment(
        position=segment.position,
        text=segment.text,
        start_seconds=time_range.start_seconds,
        end_seconds=time_range.end_seconds,
    )


def _segment_content_length(segment: Segment) -> int:
    return len(_normalize_text_for_duplicate_check(segment.text))


def _normalize_text_for_duplicate_check(text: str) -> str:
    return "".join(
        character
        for character in str(text)
        if _is_substantive_character(character)
    )


def _has_substantive_text(text: str) -> bool:
    return any(_is_substantive_character(character) for character in str(text))


def _is_substantive_character(character: str) -> bool:
    character_category = category(character)
    return not (
        character_category.startswith("P")
        or character_category.startswith("Z")
        or character_category.startswith("C")
    )



def _segments_from_result(
    result: Any,
    *,
    source_path: Path | None = None,
    fallback_duration_seconds: float | None = None,
    time_offset_seconds: float = 0.0,
) -> tuple[Segment, ...]:
    if fallback_duration_seconds is None and source_path is not None:
        fallback_duration_seconds = _audio_duration_seconds(source_path)
    external_segments = _external_value(result, "segments")
    if isinstance(external_segments, (list, tuple)):
        segments = tuple(
            segment
            for segment in (
                _segment_from_external(
                    external_segment,
                    position=position,
                    fallback_duration_seconds=fallback_duration_seconds,
                    time_offset_seconds=time_offset_seconds,
                )
                for position, external_segment in enumerate(external_segments)
            )
            if segment is not None
        )
        if segments:
            return _renumber_segments(segments)

    text = _text_from_external(result)
    if not _has_substantive_text(text):
        return ()

    return (
        _segment(
            position=0,
            text=text,
            start_seconds=time_offset_seconds,
            end_seconds=time_offset_seconds + (fallback_duration_seconds or 0.0),
        ),
    )


def _segment_from_external(
    external_segment: Any,
    *,
    position: int,
    fallback_duration_seconds: float | None,
    time_offset_seconds: float,
) -> Segment | None:
    text = _text_from_external(external_segment)
    if not _has_substantive_text(text):
        return None

    start_seconds, end_seconds = _external_time_range_seconds(external_segment)
    if start_seconds is None:
        start_seconds = 0.0
    if end_seconds is None:
        end_seconds = fallback_duration_seconds or start_seconds
    if end_seconds < start_seconds:
        end_seconds = start_seconds

    start_seconds += time_offset_seconds
    end_seconds += time_offset_seconds
    return _segment(
        position=position,
        text=text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _external_time_range_seconds(source: Any) -> tuple[float | None, float | None]:
    start_seconds = _optional_seconds(
        _external_value(source, "start_seconds", "start", "start_time")
    )
    end_seconds = _optional_seconds(
        _external_value(source, "end_seconds", "end", "end_time")
    )
    time_range = _external_value(source, "time_range")
    if start_seconds is None and time_range is not None:
        start_seconds = _optional_seconds(
            _external_value(time_range, "start_seconds", "start", "start_time")
        )
    if end_seconds is None and time_range is not None:
        end_seconds = _optional_seconds(
            _external_value(time_range, "end_seconds", "end", "end_time")
        )

    return start_seconds, end_seconds


def _external_value(source: Any, *names: str) -> Any:
    if isinstance(source, dict):
        for name in names:
            if name in source:
                return source[name]
        return None

    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value

    return None


def _optional_seconds(value: object) -> float | None:
    if value is None:
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None

    if seconds < 0:
        return None

    return seconds


def _text_from_external(source: Any) -> str:
    text = _external_value(source, "text")
    if text is None:
        text = source

    return str(text).strip()


def _first_transcription_output(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        if not output:
            return ""
        return output[0]

    return output


def _segment(
    *,
    position: int,
    text: str,
    start_seconds: float,
    end_seconds: float,
) -> Segment:
    time_range = TimeRange(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    sentence = Sentence(
        text=text,
        time_range=time_range,
        words=(),
    )
    return Segment(
        position=position,
        text=text,
        time_range=time_range,
        sentences=(sentence,),
    )


def _renumber_segments(segments: tuple[Segment, ...]) -> tuple[Segment, ...]:
    return tuple(
        Segment(
            position=position,
            text=segment.text,
            time_range=segment.time_range,
            sentences=segment.sentences,
            speaker_id=segment.speaker_id,
        )
        for position, segment in enumerate(segments)
    )


def _audio_duration_seconds(source_path: Path) -> float | None:
    try:
        import torchaudio
    except ImportError:
        return None

    try:
        info = torchaudio.info(str(source_path))
    except Exception:
        return None

    sample_rate = getattr(info, "sample_rate", 0)
    num_frames = getattr(info, "num_frames", 0)
    if sample_rate <= 0 or num_frames < 0:
        return None

    return float(num_frames) / float(sample_rate)


__all__ = [
    "DEFAULT_REAZON_SPEECH_BATCH_SIZE",
    "DEFAULT_REAZON_SPEECH_CHUNK_LENGTH_SECONDS",
    "DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS",
    "DEFAULT_REAZON_SPEECH_DEVICE",
    "DEFAULT_REAZON_SPEECH_MODEL_ID",
    "ReazonSpeechDependencyError",
    "ReazonSpeechTranscriber",
]

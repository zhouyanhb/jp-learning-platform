"""Kotoba Whisper transcription adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jp_learning_platform.domain import Segment, Sentence, TimeRange
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG,
)
from jp_learning_platform.workflow.whisper_stage import (
    WhisperTranscript,
    WhisperTranscriptionRequest,
)

DEFAULT_KOTOBA_WHISPER_MODEL_ID = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.model_id
DEFAULT_KOTOBA_WHISPER_LANGUAGE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.language
DEFAULT_KOTOBA_WHISPER_TASK = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.task
DEFAULT_KOTOBA_WHISPER_DEVICE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.device
DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.chunk_length_seconds
)
DEFAULT_KOTOBA_WHISPER_BATCH_SIZE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.batch_size
KOTOBA_WHISPER_V2_0_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0"
KOTOBA_WHISPER_V2_1_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.1"

_KOTOBA_MODEL_ALIASES = {
    "2.0": KOTOBA_WHISPER_V2_0_MODEL_ID,
    "v2.0": KOTOBA_WHISPER_V2_0_MODEL_ID,
    "kotoba-whisper-v2.0": KOTOBA_WHISPER_V2_0_MODEL_ID,
    KOTOBA_WHISPER_V2_0_MODEL_ID: KOTOBA_WHISPER_V2_0_MODEL_ID,
    "2.1": KOTOBA_WHISPER_V2_1_MODEL_ID,
    "v2.1": KOTOBA_WHISPER_V2_1_MODEL_ID,
    "kotoba-whisper-v2.1": KOTOBA_WHISPER_V2_1_MODEL_ID,
    KOTOBA_WHISPER_V2_1_MODEL_ID: KOTOBA_WHISPER_V2_1_MODEL_ID,
}


class KotobaWhisperDependencyError(RuntimeError):
    """Raised when transformers or torch support is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "transformers, torch, and Kotoba Whisper runtime support are required "
            "for Kotoba transcription. Install them with: "
            "python -m pip install -e '.[asr]'"
        )


@dataclass(slots=True)
class KotobaWhisperTranscriber:
    """Transcribe Japanese audio with kotoba-tech/kotoba-whisper models."""

    model_id: str = DEFAULT_KOTOBA_WHISPER_MODEL_ID
    language: str = DEFAULT_KOTOBA_WHISPER_LANGUAGE
    task: str = DEFAULT_KOTOBA_WHISPER_TASK
    device: str = DEFAULT_KOTOBA_WHISPER_DEVICE
    chunk_length_seconds: float = DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS
    batch_size: int = DEFAULT_KOTOBA_WHISPER_BATCH_SIZE
    trust_remote_code: bool = False
    punctuator: bool = True
    _pipeline: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_id = _normalize_model_id(self.model_id)
        self.language = _normalize_non_empty_text(self.language, "language")
        self.task = _normalize_non_empty_text(self.task, "task")
        self.device = _normalize_non_empty_text(self.device, "device")
        self.chunk_length_seconds = _normalize_positive_float(
            self.chunk_length_seconds,
            "chunk_length_seconds",
        )
        self.batch_size = _normalize_positive_int(self.batch_size, "batch_size")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("trust_remote_code must be a bool.")
        if not isinstance(self.punctuator, bool):
            raise TypeError("punctuator must be a bool.")

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        if not isinstance(request, WhisperTranscriptionRequest):
            raise TypeError("request must be a WhisperTranscriptionRequest.")

        output = self._load_pipeline()(
            str(request.source_path),
            chunk_length_s=self.chunk_length_seconds,
            batch_size=self.batch_size,
            return_timestamps=True,
            generate_kwargs={
                "language": self.language,
                "task": self.task,
            },
        )
        segments = _segments_from_pipeline_output(
            output,
            source_path=request.source_path,
        )
        return WhisperTranscript(
            source_path=request.source_path,
            segments=segments,
        )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                import torch
                from transformers import (
                    AutoModelForSpeechSeq2Seq,
                    AutoProcessor,
                    pipeline,
                )
            except ImportError as error:
                raise KotobaWhisperDependencyError() from error

            dtype = _dtype_for_model(
                model_id=self.model_id,
                device=self.device,
                torch=torch,
            )
            if self.trust_remote_code and _uses_kotoba_v2_1_pipeline(self.model_id):
                pipeline_kwargs: dict[str, object] = {
                    "model": self.model_id,
                    "dtype": dtype,
                    "device": _transformers_device(self.device),
                    "model_kwargs": _model_kwargs(self.device),
                    "batch_size": self.batch_size,
                    "trust_remote_code": self.trust_remote_code,
                }
                pipeline_kwargs["punctuator"] = self.punctuator
                self._pipeline = pipeline(**pipeline_kwargs)
            else:
                processor_model_id = _processor_model_id_for(self.model_id)
                processor = AutoProcessor.from_pretrained(
                    processor_model_id,
                    trust_remote_code=False,
                )
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    dtype=dtype,
                    use_safetensors=True,
                    trust_remote_code=False,
                    **_model_kwargs(self.device),
                )
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device=_transformers_device(self.device),
                    batch_size=self.batch_size,
                )

        return self._pipeline


def _normalize_model_id(value: str) -> str:
    model_id = _normalize_non_empty_text(value, "model_id")
    return _KOTOBA_MODEL_ALIASES.get(model_id, model_id)


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


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


def _normalize_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")

    return value


def _uses_kotoba_v2_1_pipeline(model_id: str) -> bool:
    return model_id.rstrip("/") == KOTOBA_WHISPER_V2_1_MODEL_ID


def _processor_model_id_for(model_id: str) -> str:
    if _uses_kotoba_v2_1_pipeline(model_id):
        return KOTOBA_WHISPER_V2_0_MODEL_ID

    return model_id


def _dtype_for_model(*, model_id: str, device: str, torch: Any) -> Any:
    if _is_cuda_device(device):
        if _uses_kotoba_v2_1_pipeline(model_id):
            return torch.float16

        return torch.bfloat16

    return torch.float32


def _is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def _transformers_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"

    return device


def _model_kwargs(device: str) -> dict[str, object]:
    if _is_cuda_device(device):
        return {"attn_implementation": "sdpa"}

    return {}


def _segments_from_pipeline_output(
    output: Any,
    *,
    source_path: Path,
) -> tuple[Segment, ...]:
    if not isinstance(output, dict):
        raise TypeError("Kotoba Whisper pipeline must return a dict.")

    fallback_duration_seconds = (
        _audio_duration_seconds(source_path)
        if _needs_fallback_duration(output)
        else None
    )
    chunks = output.get("chunks")
    if isinstance(chunks, list):
        segments = tuple(
            segment
            for segment in (
                _segment_from_chunk(
                    chunk,
                    position=position,
                    fallback_duration_seconds=fallback_duration_seconds,
                )
                for position, chunk in enumerate(chunks)
            )
            if segment is not None
        )
        if segments:
            return _renumber_segments(segments)

    text = str(output.get("text", "")).strip()
    if not text:
        return ()

    end_seconds = fallback_duration_seconds or 0.0
    return (
        _segment(
            position=0,
            text=text,
            start_seconds=0.0,
            end_seconds=end_seconds,
        ),
    )


def _needs_fallback_duration(output: dict[str, object]) -> bool:
    chunks = output.get("chunks")
    if not isinstance(chunks, list):
        return bool(str(output.get("text", "")).strip())

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        if not str(chunk.get("text", "")).strip():
            continue

        timestamp = chunk.get("timestamp", chunk.get("timestamps"))
        _start_seconds, end_seconds = _timestamp_seconds(timestamp)
        if end_seconds is None:
            return True

    return False


def _segment_from_chunk(
    chunk: object,
    *,
    position: int,
    fallback_duration_seconds: float | None,
) -> Segment | None:
    if not isinstance(chunk, dict):
        return None

    text = str(chunk.get("text", "")).strip()
    if not text:
        return None

    timestamp = chunk.get("timestamp", chunk.get("timestamps"))
    start_seconds, end_seconds = _timestamp_seconds(timestamp)
    if start_seconds is None:
        start_seconds = 0.0
    if end_seconds is None:
        end_seconds = fallback_duration_seconds or start_seconds

    if end_seconds < start_seconds:
        end_seconds = start_seconds

    return _segment(
        position=position,
        text=text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _timestamp_seconds(timestamp: object) -> tuple[float | None, float | None]:
    if not isinstance(timestamp, (tuple, list)):
        return None, None

    if len(timestamp) < 2:
        return None, None

    return _optional_seconds(timestamp[0]), _optional_seconds(timestamp[1])


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
    "DEFAULT_KOTOBA_WHISPER_BATCH_SIZE",
    "DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS",
    "DEFAULT_KOTOBA_WHISPER_DEVICE",
    "DEFAULT_KOTOBA_WHISPER_LANGUAGE",
    "DEFAULT_KOTOBA_WHISPER_MODEL_ID",
    "DEFAULT_KOTOBA_WHISPER_TASK",
    "KOTOBA_WHISPER_V2_0_MODEL_ID",
    "KOTOBA_WHISPER_V2_1_MODEL_ID",
    "KotobaWhisperDependencyError",
    "KotobaWhisperTranscriber",
]

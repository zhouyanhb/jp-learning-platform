"""Centralized configuration defaults for local subtitle pipeline adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WhisperTranscriptionConfig:
    """Default ASR transcription settings."""

    backend: str = "kotoba-whisper"
    model_id: str = "kotoba-tech/kotoba-whisper-v2.1"
    reazon_model_id: str = "reazon-research/reazonspeech-nemo-v2"
    model_size: str = "large-v3"
    language: str = "ja"
    task: str = "transcribe"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    word_timestamps: bool = True
    vad_filter: bool = True
    vad_min_silence_ms: int = 350
    condition_on_previous_text: bool = False
    hallucination_silence_threshold_seconds: float = 2.0
    chunk_length_seconds: float = 15.0
    chunk_overlap_seconds: float = 2.0
    batch_size: int = 16


@dataclass(frozen=True, slots=True)
class WhisperXAlignmentConfig:
    """Default WhisperX forced-alignment settings."""

    language_code: str = "ja"


@dataclass(frozen=True, slots=True)
class PyannoteDiarizationConfig:
    """Default pyannote.audio speaker diarization settings."""

    model_name: str = "pyannote/speaker-diarization-3.1"
    token_environment_variable: str = "HF_TOKEN"


@dataclass(frozen=True, slots=True)
class QwenRepairConfig:
    """Default llama.cpp Qwen repair generation settings."""

    model_path: Path = Path("models/Qwen2.5-14B-Instruct-Q4_K_M.gguf")
    context_size: int = 4096
    threads: int = 8
    gpu_layers: int = 0
    max_tokens: int = 128
    temperature: float = 0.03
    top_p: float = 0.9
    repeat_penalty: float = 1.1


@dataclass(frozen=True, slots=True)
class QwenRepairSafetyConfig:
    """Default safety thresholds for accepting Qwen transcript repairs."""

    max_length_delta_ratio: float = 0.2
    max_content_change_ratio: float = 0.2


@dataclass(frozen=True, slots=True)
class SentenceBoundaryDetectionConfig:
    """Default acoustic sentence boundary candidate settings."""

    min_pause_seconds: float = 0.45
    min_vad_silence_seconds: float = 0.3
    target_sample_rate: int = 16000
    frame_seconds: float = 0.02
    energy_threshold_ratio: float = 0.18


@dataclass(frozen=True, slots=True)
class SentenceBoundaryResolutionConfig:
    """Default final sentence boundary resolution settings."""

    min_acoustic_score: float = 0.55
    min_sentence_seconds: float = 0.4
    min_sentence_chars: int = 4


@dataclass(frozen=True, slots=True)
class SubtitleMergeConfig:
    """Default conservative subtitle merge settings."""

    max_gap_seconds: float = 0.35
    max_chars: int = 42
    terminal_marks: tuple[str, ...] = ("。", "？", "！")


@dataclass(frozen=True, slots=True)
class ReadabilityConfig:
    """Default Japanese subtitle readability normalization settings."""

    japanese_comma: str = "、"
    japanese_period: str = "。"


DEFAULT_WHISPER_TRANSCRIPTION_CONFIG = WhisperTranscriptionConfig()
DEFAULT_WHISPERX_ALIGNMENT_CONFIG = WhisperXAlignmentConfig()
DEFAULT_PYANNOTE_DIARIZATION_CONFIG = PyannoteDiarizationConfig()
DEFAULT_QWEN_REPAIR_CONFIG = QwenRepairConfig()
DEFAULT_QWEN_MODEL_PATH = DEFAULT_QWEN_REPAIR_CONFIG.model_path
DEFAULT_QWEN_REPAIR_SAFETY_CONFIG = QwenRepairSafetyConfig()
DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG = SentenceBoundaryDetectionConfig()
DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG = SentenceBoundaryResolutionConfig()
DEFAULT_SUBTITLE_MERGE_CONFIG = SubtitleMergeConfig()
DEFAULT_READABILITY_CONFIG = ReadabilityConfig()


__all__ = [
    "DEFAULT_PYANNOTE_DIARIZATION_CONFIG",
    "DEFAULT_QWEN_MODEL_PATH",
    "DEFAULT_QWEN_REPAIR_CONFIG",
    "DEFAULT_QWEN_REPAIR_SAFETY_CONFIG",
    "DEFAULT_READABILITY_CONFIG",
    "DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG",
    "DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG",
    "DEFAULT_SUBTITLE_MERGE_CONFIG",
    "DEFAULT_WHISPER_TRANSCRIPTION_CONFIG",
    "DEFAULT_WHISPERX_ALIGNMENT_CONFIG",
    "PyannoteDiarizationConfig",
    "QwenRepairConfig",
    "QwenRepairSafetyConfig",
    "ReadabilityConfig",
    "SentenceBoundaryDetectionConfig",
    "SentenceBoundaryResolutionConfig",
    "SubtitleMergeConfig",
    "WhisperTranscriptionConfig",
    "WhisperXAlignmentConfig",
]

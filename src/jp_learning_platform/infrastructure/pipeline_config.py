"""Centralized configuration defaults for local subtitle pipeline adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WhisperTranscriptionConfig:
    """Default faster-whisper transcription settings."""

    model_size: str = "large-v3"
    language: str = "ja"
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
class HomophonePrefilterConfig:
    """Default risk-based homophone target prefilter settings."""

    max_targets_per_sentence: int = 3


@dataclass(frozen=True, slots=True)
class SubtitleMergeConfig:
    """Default conservative subtitle merge settings."""

    max_gap_seconds: float = 0.35
    max_chars: int = 42
    terminal_marks: tuple[str, ...] = ("。", "？", "！")


@dataclass(frozen=True, slots=True)
class SentenceBoundaryConfig:
    """Default pause-aware Japanese sentence boundary settings."""

    min_pause_seconds: float = 0.5
    max_dependent_continuation_gap_seconds: float = 0.2
    terminal_marks: tuple[str, ...] = ("。", "？", "！")
    sentence_final_suffixes: tuple[str, ...] = (
        "ください",
        "下さい",
        "くださいね",
        "下さいね",
        "ます",
        "ました",
        "ません",
        "ませんか",
        "ましょう",
        "です",
        "でした",
        "でしょう",
        "だ",
        "だった",
    )
    dependent_continuation_prefixes: tuple[str, ...] = (
        "とき",
        "時",
        "場合",
        "ため",
        "ので",
        "のに",
        "なら",
        "けれど",
        "けど",
    )


@dataclass(frozen=True, slots=True)
class ReadabilityConfig:
    """Default Japanese subtitle readability normalization settings."""

    japanese_comma: str = "、"
    japanese_period: str = "。"
    sentence_initial_discourse_markers: tuple[str, ...] = (
        "それでは",
        "ところで",
        "しかし",
        "では",
        "さて",
    )
    non_discourse_prefixes: tuple[str, ...] = (
        "ではありません",
        "ではない",
        "ではなく",
        "ではなければ",
    )


DEFAULT_WHISPER_TRANSCRIPTION_CONFIG = WhisperTranscriptionConfig()
DEFAULT_WHISPERX_ALIGNMENT_CONFIG = WhisperXAlignmentConfig()
DEFAULT_PYANNOTE_DIARIZATION_CONFIG = PyannoteDiarizationConfig()
DEFAULT_QWEN_REPAIR_CONFIG = QwenRepairConfig()
DEFAULT_QWEN_REPAIR_SAFETY_CONFIG = QwenRepairSafetyConfig()
DEFAULT_HOMOPHONE_PREFILTER_CONFIG = HomophonePrefilterConfig()
DEFAULT_SUBTITLE_MERGE_CONFIG = SubtitleMergeConfig()
DEFAULT_SENTENCE_BOUNDARY_CONFIG = SentenceBoundaryConfig()
DEFAULT_READABILITY_CONFIG = ReadabilityConfig()


__all__ = [
    "DEFAULT_PYANNOTE_DIARIZATION_CONFIG",
    "DEFAULT_HOMOPHONE_PREFILTER_CONFIG",
    "DEFAULT_QWEN_REPAIR_CONFIG",
    "DEFAULT_QWEN_REPAIR_SAFETY_CONFIG",
    "DEFAULT_READABILITY_CONFIG",
    "DEFAULT_SENTENCE_BOUNDARY_CONFIG",
    "DEFAULT_SUBTITLE_MERGE_CONFIG",
    "DEFAULT_WHISPER_TRANSCRIPTION_CONFIG",
    "DEFAULT_WHISPERX_ALIGNMENT_CONFIG",
    "PyannoteDiarizationConfig",
    "HomophonePrefilterConfig",
    "QwenRepairConfig",
    "QwenRepairSafetyConfig",
    "ReadabilityConfig",
    "SentenceBoundaryConfig",
    "SubtitleMergeConfig",
    "WhisperTranscriptionConfig",
    "WhisperXAlignmentConfig",
]

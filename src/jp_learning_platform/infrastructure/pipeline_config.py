"""Centralized configuration defaults for local subtitle pipeline adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WhisperTranscriptionConfig:
    """Default faster-whisper transcription settings."""

    model_size: str = "turbo"
    language: str = "ja"
    initial_prompt: str = (
        "これは日本語学習教材の書き起こしです。"
        "自然な日本語の句読点を使用します。"
    )
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    word_timestamps: bool = True
    vad_filter: bool = True
    vad_min_silence_ms: int = 600
    condition_on_previous_text: bool = False
    hallucination_silence_threshold_seconds: float = 2.0
    retry_confidence_threshold: float = 0.65
    retry_context_confidence_threshold: float = 0.85
    retry_min_confidence_improvement: float = 0.05
    retry_max_segments: int = 12


@dataclass(frozen=True, slots=True)
class WhisperXAlignmentConfig:
    """Default WhisperX forced-alignment settings."""

    language_code: str = "ja"


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
class HomophoneConfidencePolicyConfig:
    """Evidence thresholds for changing acoustically confident homophones."""

    high_asr_confidence: float = 0.9
    high_confidence_min_score_ratio: float = 200.0


@dataclass(frozen=True, slots=True)
class SubtitleMergeConfig:
    """Default conservative subtitle merge settings."""

    max_gap_seconds: float = 0.35
    max_chars: int = 42
    max_duration_seconds: float = 10.0
    terminal_marks: tuple[str, ...] = ("。", "？", "！", "?", "!")


@dataclass(frozen=True, slots=True)
class SubtitleDisplayConfig:
    """Limits for turning one linguistic sentence into timed display cues."""

    max_chars: int = 42
    max_duration_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class SentenceBoundaryConfig:
    """Default pause-aware Japanese sentence boundary settings."""

    min_pause_seconds: float = 0.5
    max_dependent_continuation_gap_seconds: float = 0.2
    max_continuation_candidate_gap_seconds: float = 0.5
    max_syntactic_dependency_gap_seconds: float = 1.0
    continuation_score_threshold: int = 4
    close_timing_evidence_score: int = 1
    dependent_prefix_evidence_score: int = 3
    functional_continuation_evidence_score: int = 2
    incomplete_left_evidence_score: int = 2
    question_answer_min_pause_ratio: float = 0.3
    question_answer_min_relative_pause: float = 0.35
    connective_response_min_pause_ratio: float = 0.8
    connective_response_min_relative_pause: float = 0.75
    max_connective_continuation_gap_seconds: float = 1.25
    connective_merge_score_margin: int = 1
    numbering_region_min_sequence_length: int = 3
    numbering_region_max_item_gap_seconds: float = 45.0
    numbering_region_min_body_characters: int = 2
    extended_word_duration_seconds: float = 3.0
    extended_word_seconds_per_character: float = 1.0
    max_aligned_word_seconds_per_character: float = 0.5
    terminal_marks: tuple[str, ...] = ("。", "？", "！", "?", "!")
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
DEFAULT_QWEN_REPAIR_CONFIG = QwenRepairConfig()
DEFAULT_QWEN_REPAIR_SAFETY_CONFIG = QwenRepairSafetyConfig()
DEFAULT_HOMOPHONE_PREFILTER_CONFIG = HomophonePrefilterConfig()
DEFAULT_HOMOPHONE_CONFIDENCE_POLICY_CONFIG = HomophoneConfidencePolicyConfig()
DEFAULT_SUBTITLE_DISPLAY_CONFIG = SubtitleDisplayConfig()
DEFAULT_SUBTITLE_MERGE_CONFIG = SubtitleMergeConfig()
DEFAULT_SENTENCE_BOUNDARY_CONFIG = SentenceBoundaryConfig()
DEFAULT_READABILITY_CONFIG = ReadabilityConfig()


__all__ = [
    "DEFAULT_HOMOPHONE_PREFILTER_CONFIG",
    "DEFAULT_HOMOPHONE_CONFIDENCE_POLICY_CONFIG",
    "DEFAULT_QWEN_REPAIR_CONFIG",
    "DEFAULT_QWEN_REPAIR_SAFETY_CONFIG",
    "DEFAULT_READABILITY_CONFIG",
    "DEFAULT_SENTENCE_BOUNDARY_CONFIG",
    "DEFAULT_SUBTITLE_DISPLAY_CONFIG",
    "DEFAULT_SUBTITLE_MERGE_CONFIG",
    "DEFAULT_WHISPER_TRANSCRIPTION_CONFIG",
    "DEFAULT_WHISPERX_ALIGNMENT_CONFIG",
    "HomophonePrefilterConfig",
    "HomophoneConfidencePolicyConfig",
    "QwenRepairConfig",
    "QwenRepairSafetyConfig",
    "ReadabilityConfig",
    "SentenceBoundaryConfig",
    "SubtitleMergeConfig",
    "SubtitleDisplayConfig",
    "WhisperTranscriptionConfig",
    "WhisperXAlignmentConfig",
]

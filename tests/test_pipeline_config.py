from __future__ import annotations

from jp_learning_platform.infrastructure import (
    DEFAULT_HOMOPHONE_CONFIDENCE_POLICY_CONFIG,
    DEFAULT_HOMOPHONE_PREFILTER_CONFIG,
    DEFAULT_QWEN_REPAIR_CONFIG,
    DEFAULT_QWEN_REPAIR_SAFETY_CONFIG,
    DEFAULT_READABILITY_CONFIG,
    DEFAULT_SENTENCE_BOUNDARY_CONFIG,
    DEFAULT_SUBTITLE_MERGE_CONFIG,
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG,
    DEFAULT_WHISPERX_ALIGNMENT_CONFIG,
)


def test_pipeline_config_centralizes_asr_defaults() -> None:
    config = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG

    assert config.model_size == "large-v3"
    assert config.language == "ja"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.beam_size == 5
    assert config.best_of == 5
    assert config.temperature == 0.0
    assert config.word_timestamps
    assert config.vad_filter
    assert config.vad_min_silence_ms == 600
    assert not config.condition_on_previous_text
    assert config.hallucination_silence_threshold_seconds == 2.0
    assert config.retry_confidence_threshold == 0.65
    assert config.retry_context_confidence_threshold == 0.85
    assert config.retry_min_confidence_improvement == 0.05
    assert config.retry_max_segments == 12


def test_pipeline_config_centralizes_quality_defaults() -> None:
    assert DEFAULT_HOMOPHONE_CONFIDENCE_POLICY_CONFIG.high_asr_confidence == 0.9
    assert (
        DEFAULT_HOMOPHONE_CONFIDENCE_POLICY_CONFIG.high_confidence_min_score_ratio
        == 200.0
    )
    assert DEFAULT_HOMOPHONE_PREFILTER_CONFIG.max_targets_per_sentence == 3
    assert DEFAULT_WHISPERX_ALIGNMENT_CONFIG.language_code == "ja"
    assert DEFAULT_QWEN_REPAIR_CONFIG.context_size == 4096
    assert DEFAULT_QWEN_REPAIR_CONFIG.threads == 8
    assert DEFAULT_QWEN_REPAIR_CONFIG.max_tokens == 128
    assert DEFAULT_QWEN_REPAIR_CONFIG.temperature == 0.03
    assert DEFAULT_QWEN_REPAIR_CONFIG.top_p == 0.9
    assert DEFAULT_QWEN_REPAIR_CONFIG.repeat_penalty == 1.1
    assert DEFAULT_QWEN_REPAIR_SAFETY_CONFIG.max_length_delta_ratio == 0.2
    assert DEFAULT_QWEN_REPAIR_SAFETY_CONFIG.max_content_change_ratio == 0.2
    assert DEFAULT_SENTENCE_BOUNDARY_CONFIG.min_pause_seconds == 0.5
    assert DEFAULT_SENTENCE_BOUNDARY_CONFIG.extended_word_duration_seconds == 3.0
    assert (
        DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_aligned_word_seconds_per_character
        == 0.5
    )
    assert (
        DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_dependent_continuation_gap_seconds
        == 0.2
    )
    assert (
        DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_continuation_candidate_gap_seconds
        == 0.5
    )
    assert DEFAULT_SENTENCE_BOUNDARY_CONFIG.continuation_score_threshold == 4
    assert "とき" in DEFAULT_SENTENCE_BOUNDARY_CONFIG.dependent_continuation_prefixes
    assert DEFAULT_SENTENCE_BOUNDARY_CONFIG.terminal_marks == (
        "。",
        "？",
        "！",
        "?",
        "!",
    )
    assert "ください" in DEFAULT_SENTENCE_BOUNDARY_CONFIG.sentence_final_suffixes
    assert "ましょう" in DEFAULT_SENTENCE_BOUNDARY_CONFIG.sentence_final_suffixes
    assert DEFAULT_SUBTITLE_MERGE_CONFIG.max_gap_seconds == 0.35
    assert DEFAULT_SUBTITLE_MERGE_CONFIG.max_chars == 42
    assert DEFAULT_SUBTITLE_MERGE_CONFIG.max_duration_seconds == 10.0
    assert DEFAULT_SUBTITLE_MERGE_CONFIG.terminal_marks == (
        "。",
        "？",
        "！",
        "?",
        "!",
    )
    assert DEFAULT_READABILITY_CONFIG.japanese_comma == "、"
    assert DEFAULT_READABILITY_CONFIG.japanese_period == "。"
    assert "では" in DEFAULT_READABILITY_CONFIG.sentence_initial_discourse_markers
    assert "ではありません" in DEFAULT_READABILITY_CONFIG.non_discourse_prefixes

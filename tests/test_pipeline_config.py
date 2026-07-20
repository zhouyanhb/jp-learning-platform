from __future__ import annotations

from pathlib import Path

from jp_learning_platform.infrastructure import (
    DEFAULT_PYANNOTE_DIARIZATION_CONFIG,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_QWEN_REPAIR_CONFIG,
    DEFAULT_QWEN_REPAIR_SAFETY_CONFIG,
    DEFAULT_READABILITY_CONFIG,
    DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG,
    DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG,
    DEFAULT_SUBTITLE_MERGE_CONFIG,
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG,
    DEFAULT_WHISPERX_ALIGNMENT_CONFIG,
)


def test_pipeline_config_centralizes_asr_defaults() -> None:
    config = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG

    assert config.backend == "kotoba-whisper"
    assert config.model_id == "kotoba-tech/kotoba-whisper-v2.1"
    assert config.reazon_model_id == "reazon-research/reazonspeech-nemo-v2"
    assert config.model_size == "large-v3"
    assert config.language == "ja"
    assert config.task == "transcribe"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.beam_size == 5
    assert config.best_of == 5
    assert config.temperature == 0.0
    assert config.word_timestamps
    assert config.vad_filter
    assert config.vad_min_silence_ms == 350
    assert not config.condition_on_previous_text
    assert config.hallucination_silence_threshold_seconds == 2.0
    assert config.chunk_length_seconds == 15.0
    assert config.chunk_overlap_seconds == 2.0
    assert config.batch_size == 16


def test_pipeline_config_centralizes_quality_defaults() -> None:
    assert DEFAULT_WHISPERX_ALIGNMENT_CONFIG.language_code == "ja"
    assert (
        DEFAULT_PYANNOTE_DIARIZATION_CONFIG.model_name
        == "pyannote/speaker-diarization-3.1"
    )
    assert DEFAULT_PYANNOTE_DIARIZATION_CONFIG.token_environment_variable == "HF_TOKEN"
    assert (
        DEFAULT_QWEN_MODEL_PATH
        == DEFAULT_QWEN_REPAIR_CONFIG.model_path
        == Path("models/Qwen2.5-14B-Instruct-Q4_K_M.gguf")
    )
    assert DEFAULT_QWEN_REPAIR_CONFIG.context_size == 4096
    assert DEFAULT_QWEN_REPAIR_CONFIG.threads == 8
    assert DEFAULT_QWEN_REPAIR_CONFIG.max_tokens == 128
    assert DEFAULT_QWEN_REPAIR_CONFIG.temperature == 0.03
    assert DEFAULT_QWEN_REPAIR_CONFIG.top_p == 0.9
    assert DEFAULT_QWEN_REPAIR_CONFIG.repeat_penalty == 1.1
    assert DEFAULT_QWEN_REPAIR_SAFETY_CONFIG.max_length_delta_ratio == 0.2
    assert DEFAULT_QWEN_REPAIR_SAFETY_CONFIG.max_content_change_ratio == 0.2
    assert DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG.min_pause_seconds == 0.45
    assert (
        DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG.min_vad_silence_seconds
        == 0.3
    )
    assert DEFAULT_SENTENCE_BOUNDARY_DETECTION_CONFIG.target_sample_rate == 16000
    assert DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG.min_acoustic_score == 0.55
    assert DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG.min_sentence_seconds == 0.4
    assert DEFAULT_SENTENCE_BOUNDARY_RESOLUTION_CONFIG.min_sentence_chars == 4
    assert DEFAULT_SUBTITLE_MERGE_CONFIG.max_gap_seconds == 0.35
    assert DEFAULT_SUBTITLE_MERGE_CONFIG.max_chars == 42
    assert DEFAULT_SUBTITLE_MERGE_CONFIG.terminal_marks == ("。", "？", "！")
    assert DEFAULT_READABILITY_CONFIG.japanese_comma == "、"
    assert DEFAULT_READABILITY_CONFIG.japanese_period == "。"

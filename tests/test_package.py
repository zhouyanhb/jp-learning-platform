from __future__ import annotations

from io import StringIO
from pathlib import Path
import tomllib

import pytest

import jp_learning_platform
from jp_learning_platform.__main__ import (
    _build_homophone_resolver,
    _build_repairer,
    _build_transcriber,
    build_parser,
    main,
)
from jp_learning_platform.infrastructure import (
    DEFAULT_KOTOBA_WHISPER_BATCH_SIZE,
    DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS,
    DEFAULT_KOTOBA_WHISPER_MODEL_ID,
    DEFAULT_HOMOPHONE_MODEL_ID,
    DEFAULT_HOMOPHONE_SCORE_MARGIN,
    DEFAULT_HOMOPHONE_TOP_K,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS,
    DEFAULT_REAZON_SPEECH_MODEL_ID,
    DisabledQwenRepairer,
    FasterWhisperTranscriber,
    BertHomophoneResolver,
    KOTOBA_WHISPER_V2_0_MODEL_ID,
    KotobaWhisperTranscriber,
    LlamaCppQwenRepairer,
    ReazonSpeechTranscriber,
)


def test_package_exposes_release_version() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.0.0"
    assert jp_learning_platform.__version__ == pyproject["project"]["version"]
    assert pyproject["project"]["optional-dependencies"]["asr"] == [
        "transformers>=4.39",
        "accelerate>=0.28",
        "torch>=2.0",
        "torchaudio>=2.0",
        "faster-whisper>=1.0",
    ]
    assert pyproject["project"]["optional-dependencies"]["diarization"] == [
        "pyannote.audio>=3.1"
    ]
    assert pyproject["project"]["optional-dependencies"]["vad"] == [
        "torch>=2.0",
        "torchaudio>=2.0",
    ]
    assert pyproject["project"]["optional-dependencies"]["japanese"] == [
        "sudachipy>=0.6",
        "sudachidict-core>=2024.1",
    ]
    assert pyproject["project"]["optional-dependencies"]["homophone"] == [
        "transformers>=4.39",
        "torch>=2.0",
        "sudachipy>=0.6",
        "sudachidict-core>=2024.1",
        "fugashi>=1.3",
        "unidic-lite>=1.0",
    ]


def test_module_entrypoint_returns_success() -> None:
    output = StringIO()

    assert main((), stdout=output) == 0

    result = output.getvalue()
    assert "jp-learning-platform 1.0.0" in result
    assert "Version 1.0 subtitle pipeline" in result
    assert "Subtitle Writer" in result


def test_module_entrypoint_reports_version() -> None:
    output = StringIO()

    assert main(("--version",), stdout=output) == 0
    assert output.getvalue() == "jp-learning-platform 1.0.0\n"


def test_transcribe_command_defaults_output_directory() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3"))

    assert args.output_dir == Path("output")
    assert not args.export_srt
    assert args.asr_backend == "kotoba-whisper"
    assert args.asr_model == DEFAULT_KOTOBA_WHISPER_MODEL_ID
    assert args.model_size == "large-v3"
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.asr_chunk_length_seconds == DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS
    assert args.asr_chunk_overlap_seconds == DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS
    assert args.asr_batch_size == DEFAULT_KOTOBA_WHISPER_BATCH_SIZE
    assert args.enable_whisperx
    assert not args.disable_word_normalization
    assert not args.disable_sentence_boundaries
    assert not args.enable_homophone_resolver
    assert args.homophone_model_id == DEFAULT_HOMOPHONE_MODEL_ID
    assert args.homophone_top_k == DEFAULT_HOMOPHONE_TOP_K
    assert args.homophone_score_margin == DEFAULT_HOMOPHONE_SCORE_MARGIN
    assert not args.enable_qwen
    assert args.qwen_model_path is None
    assert not args.disable_qwen


def test_transcribe_command_accepts_optional_srt_export() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3", "--export-srt"))

    assert args.export_srt


def test_transcribe_command_accepts_asr_model_options() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--asr-backend",
            "faster-whisper",
            "--model-size",
            "small",
            "--device",
            "cuda",
            "--compute-type",
            "float16",
        )
    )

    assert args.model_size == "small"
    assert args.device == "cuda"
    assert args.compute_type == "float16"


def test_transcribe_command_accepts_kotoba_asr_options() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--asr-model",
            "kotoba-whisper-v2.0",
            "--device",
            "cuda",
            "--asr-chunk-length-seconds",
            "30",
            "--asr-batch-size",
            "4",
        )
    )

    assert args.asr_backend == "kotoba-whisper"
    assert args.asr_model == "kotoba-whisper-v2.0"
    assert args.device == "cuda"
    assert args.asr_chunk_length_seconds == 30
    assert args.asr_batch_size == 4


def test_transcribe_command_accepts_reazon_speech_backend() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--asr-backend",
            "reazon-speech",
            "--device",
            "cuda",
            "--asr-batch-size",
            "2",
            "--asr-chunk-overlap-seconds",
            "3",
        )
    )

    assert args.asr_backend == "reazon-speech"
    assert args.device == "cuda"
    assert args.asr_batch_size == 2
    assert args.asr_chunk_overlap_seconds == 3


def test_transcribe_command_uses_kotoba_whisper_by_default() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3"))

    transcriber = _build_transcriber(args)

    assert isinstance(transcriber, KotobaWhisperTranscriber)
    assert transcriber.model_id == DEFAULT_KOTOBA_WHISPER_MODEL_ID
    assert not transcriber.trust_remote_code


def test_transcribe_command_can_use_kotoba_whisper_v2_0() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--asr-model",
            "kotoba-whisper-v2.0",
        )
    )

    transcriber = _build_transcriber(args)

    assert isinstance(transcriber, KotobaWhisperTranscriber)
    assert transcriber.model_id == KOTOBA_WHISPER_V2_0_MODEL_ID


def test_transcribe_command_can_fallback_to_faster_whisper() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--asr-backend",
            "faster-whisper",
            "--model-size",
            "small",
        )
    )

    transcriber = _build_transcriber(args)

    assert isinstance(transcriber, FasterWhisperTranscriber)
    assert transcriber.model_size == "small"


def test_transcribe_command_can_use_reazon_speech() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--asr-backend",
            "reazon-speech",
        )
    )

    transcriber = _build_transcriber(args)

    assert isinstance(transcriber, ReazonSpeechTranscriber)
    assert transcriber.model_id == DEFAULT_REAZON_SPEECH_MODEL_ID
    assert transcriber.chunk_length_seconds == DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS
    assert (
        transcriber.chunk_overlap_seconds
        == DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS
    )


def test_transcribe_command_can_use_custom_reazon_model() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--asr-backend",
            "reazon-speech",
            "--asr-model",
            "custom/reazon",
        )
    )

    transcriber = _build_transcriber(args)

    assert isinstance(transcriber, ReazonSpeechTranscriber)
    assert transcriber.model_id == "custom/reazon"


def test_transcribe_command_accepts_quality_stage_options() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--enable-whisperx",
            "--whisperx-language",
            "ja",
            "--enable-qwen",
        )
    )

    assert args.enable_whisperx
    assert args.whisperx_language == "ja"
    assert args.enable_qwen


def test_transcribe_command_qwen_model_path_enables_repair() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--qwen-model-path",
            "models/qwen.gguf",
        )
    )

    assert args.qwen_model_path == Path("models/qwen.gguf")


def test_transcribe_command_can_disable_whisperx_alignment() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3", "--disable-whisperx"))

    assert not args.enable_whisperx


def test_transcribe_command_can_disable_qwen_repair() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3", "--disable-qwen"))

    assert args.disable_qwen


def test_transcribe_command_uses_disabled_qwen_repair_by_default() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3"))

    assert isinstance(_build_repairer(args), DisabledQwenRepairer)


def test_transcribe_command_can_enable_qwen_repair_with_default_model() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3", "--enable-qwen"))

    repairer = _build_repairer(args)

    assert isinstance(repairer, LlamaCppQwenRepairer)
    assert repairer.model_path == DEFAULT_QWEN_MODEL_PATH


def test_transcribe_command_uses_custom_qwen_model_when_path_is_provided() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--qwen-model-path",
            "models/qwen.gguf",
        )
    )

    repairer = _build_repairer(args)

    assert isinstance(repairer, LlamaCppQwenRepairer)
    assert repairer.model_path == Path("models/qwen.gguf")


def test_transcribe_command_rejects_conflicting_qwen_options() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            (
                "transcribe",
                "audio.mp3",
                "--disable-qwen",
                "--qwen-model-path",
                "models/qwen.gguf",
            )
        )


def test_transcribe_command_can_disable_sentence_boundaries() -> None:
    args = build_parser().parse_args(
        ("transcribe", "audio.mp3", "--disable-sentence-boundaries")
    )

    assert args.disable_sentence_boundaries


def test_transcribe_command_can_disable_word_normalization() -> None:
    args = build_parser().parse_args(
        ("transcribe", "audio.mp3", "--disable-word-normalization")
    )

    assert args.disable_word_normalization


def test_transcribe_command_accepts_homophone_resolver_options() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--enable-homophone-resolver",
            "--homophone-model-id",
            "custom/japanese-bert",
            "--homophone-top-k",
            "40",
            "--homophone-score-margin",
            "0.05",
        )
    )

    assert args.enable_homophone_resolver
    assert args.homophone_model_id == "custom/japanese-bert"
    assert args.homophone_top_k == 40
    assert args.homophone_score_margin == 0.05


def test_transcribe_command_uses_no_homophone_resolver_by_default() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3"))

    assert _build_homophone_resolver(args) is None


def test_transcribe_command_can_enable_homophone_resolver() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--enable-homophone-resolver",
            "--homophone-model-id",
            "custom/japanese-bert",
            "--homophone-top-k",
            "40",
            "--homophone-score-margin",
            "0.05",
        )
    )

    resolver = _build_homophone_resolver(args)

    assert isinstance(resolver, BertHomophoneResolver)
    assert resolver.model_id == "custom/japanese-bert"
    assert resolver.top_k == 40
    assert resolver.score_margin == 0.05


def test_transcribe_command_accepts_diarization_options() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--enable-diarization",
            "--hf-token",
            "hf-token",
        )
    )

    assert args.enable_diarization
    assert args.hf_token == "hf-token"

from __future__ import annotations

from io import StringIO
from pathlib import Path
import tomllib

import jp_learning_platform
from jp_learning_platform.__main__ import _build_homophone_resolver, build_parser, main
from jp_learning_platform.infrastructure import (
    BertHomophoneResolver,
    DEFAULT_HOMOPHONE_MODEL_ID,
    DEFAULT_HOMOPHONE_SCORE_MARGIN,
    DEFAULT_HOMOPHONE_TOP_K,
)


def test_package_exposes_release_version() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.0.0"
    assert jp_learning_platform.__version__ == pyproject["project"]["version"]
    assert pyproject["project"]["optional-dependencies"]["diarization"] == [
        "pyannote.audio>=3.1"
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
    assert args.model_size == "large-v3"
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert not args.enable_homophone_resolver
    assert args.homophone_model_id == DEFAULT_HOMOPHONE_MODEL_ID
    assert args.homophone_top_k == DEFAULT_HOMOPHONE_TOP_K
    assert args.homophone_score_margin == DEFAULT_HOMOPHONE_SCORE_MARGIN


def test_transcribe_command_accepts_optional_srt_export() -> None:
    args = build_parser().parse_args(("transcribe", "audio.mp3", "--export-srt"))

    assert args.export_srt


def test_transcribe_command_accepts_asr_model_options() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
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


def test_transcribe_command_accepts_quality_stage_options() -> None:
    args = build_parser().parse_args(
        (
            "transcribe",
            "audio.mp3",
            "--enable-whisperx",
            "--whisperx-language",
            "ja",
            "--qwen-model-path",
            "models/qwen.gguf",
        )
    )

    assert args.enable_whisperx
    assert args.whisperx_language == "ja"
    assert args.qwen_model_path == Path("models/qwen.gguf")


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

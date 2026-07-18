from __future__ import annotations

from io import StringIO
from pathlib import Path
import tomllib

import jp_learning_platform
from jp_learning_platform.__main__ import build_parser, main


def test_package_exposes_release_version() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.0.0"
    assert jp_learning_platform.__version__ == pyproject["project"]["version"]


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

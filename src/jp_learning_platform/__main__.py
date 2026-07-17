"""Command line entrypoint for JP Learning Platform."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import TextIO

from jp_learning_platform import __version__
from jp_learning_platform.application import (
    DEFAULT_OUTPUT_DIRECTORY,
    SubtitlePipelineInputError,
    SubtitlePipelineRequest,
)
from jp_learning_platform.infrastructure import (
    AudioLoader,
    AudioLoaderError,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_MODEL_SIZE,
    FasterWhisperDependencyError,
    FasterWhisperTranscriber,
    SrtSubtitleWriter,
    WordSubtitleBuilder,
)
from jp_learning_platform.workflow import (
    DuplicateSubtitleOutputError,
    SubtitlePipelineRunner,
    SubtitlePipelineRunnerError,
)

_PIPELINE_STAGES = (
    "Audio",
    "Whisper",
    "WhisperX Alignment",
    "Qwen Repair",
    "Subtitle Builder",
    "Subtitle Merger",
    "Readability Optimizer",
    "Subtitle Validator",
    "Subtitle Writer",
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="jp-learning-platform",
        description="Inspect the JP Learning Platform subtitle pipeline release.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the package version.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "status",
        help="Show Version 1.0 subtitle pipeline status.",
    )
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Generate SRT subtitles for an audio file or folder.",
    )
    transcribe_parser.add_argument(
        "input_path",
        type=Path,
        help="Audio file or folder to transcribe.",
    )
    transcribe_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIRECTORY,
        type=Path,
        help="Directory for generated SRT files. Defaults to output.",
    )
    transcribe_parser.add_argument(
        "--model-size",
        default=DEFAULT_WHISPER_MODEL_SIZE,
        help="faster-whisper model size. Defaults to large-v3.",
    )
    transcribe_parser.add_argument(
        "--device",
        default=DEFAULT_WHISPER_DEVICE,
        help="Device used by faster-whisper. Defaults to cpu.",
    )
    transcribe_parser.add_argument(
        "--compute-type",
        default=DEFAULT_WHISPER_COMPUTE_TYPE,
        help="faster-whisper compute type. Defaults to int8.",
    )

    return parser


def _write_version(output: TextIO) -> None:
    output.write(f"jp-learning-platform {__version__}\n")


def _write_status(output: TextIO) -> None:
    _write_version(output)
    output.write("Version 1.0 subtitle pipeline:\n")
    for index, stage in enumerate(_PIPELINE_STAGES, start=1):
        output.write(f"{index}. {stage}\n")
    output.write(
        "External SDK adapters are supplied through the tool registry and "
        "plugin system.\n"
    )


def _run_transcribe(args: Namespace, output: TextIO, error_output: TextIO) -> int:
    writer = SrtSubtitleWriter(output_directory=args.output_dir)
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=FasterWhisperTranscriber(
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
        ),
        builder=WordSubtitleBuilder(),
        writer=writer,
    )

    try:
        result = runner.run(
            SubtitlePipelineRequest(
                input_path=args.input_path,
                output_directory=args.output_dir,
            )
        )
    except (
        AudioLoaderError,
        DuplicateSubtitleOutputError,
        FasterWhisperDependencyError,
        RuntimeError,
        SubtitlePipelineInputError,
        SubtitlePipelineRunnerError,
        ValueError,
    ) as error:
        error_output.write(f"{error}\n")
        return 1

    for output_path in result.output_paths:
        output.write(f"{output_path}\n")

    return 0


def _run_command(args: Namespace, output: TextIO, error_output: TextIO) -> int:
    if args.version:
        _write_version(output)
        return 0

    if args.command == "transcribe":
        return _run_transcribe(args, output, error_output)

    if args.command in {None, "status"}:
        _write_status(output)
        return 0

    return 0


def main(
    argv: Sequence[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    error_output = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_command(args, output, error_output)


if __name__ == "__main__":
    raise SystemExit(main())

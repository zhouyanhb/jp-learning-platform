"""Command line entrypoint for JP Learning Platform."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
import sys
from typing import TextIO

from jp_learning_platform import __version__

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


def _run_command(args: Namespace, output: TextIO) -> int:
    if args.version:
        _write_version(output)
        return 0

    if args.command in {None, "status"}:
        _write_status(output)
        return 0

    return 0


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    output = stdout if stdout is not None else sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_command(args, output)


if __name__ == "__main__":
    raise SystemExit(main())

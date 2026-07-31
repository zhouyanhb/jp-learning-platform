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
    AuditableOverlapTextCleaner,
    AuditableRepeatedTextCleaner,
    CompositeSubtitleWriter,
    ConservativeSubtitleMerger,
    ConsoleProgressReporter,
    DEFAULT_HOMOPHONE_MODEL_ID,
    DEFAULT_HOMOPHONE_SCORE_MARGIN,
    DEFAULT_HOMOPHONE_TOP_K,
    DEFAULT_LISTENING_JSON_EXTENSION,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_MODEL_SIZE,
    DEFAULT_WHISPERX_LANGUAGE,
    BertHomophoneResolver,
    DomainSubtitleValidator,
    FasterWhisperDependencyError,
    FasterWhisperTranscriber,
    FFmpegAudioNormalizer,
    HomophoneResolverDependencyError,
    JapaneseLearningWordNormalizer,
    JapaneseSentenceBoundaryResolver,
    SudachiMorphologicalAnalyzer,
    ListeningJsonWriter,
    LocalPipelineContextCache,
    LocalReadabilityOptimizer,
    SrtSubtitleWriter,
    StageArtifactStore,
    WhisperXAlignerAdapter,
    WordSubtitleBuilder,
    WordNormalizerDependencyError,
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
    "Homophone Resolution (optional)",
    "Auditable Overlap Text Cleanup",
    "Auditable Repeated Text Cleanup",
    "Sentence Boundary Resolution",
    "Punctuation Attribution",
    "Learning Word Normalization (optional)",
    "Subtitle Builder",
    "Subtitle Display Normalization",
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
        help="Generate structured listening JSON for an audio/video file or folder.",
    )
    transcribe_parser.add_argument(
        "input_path",
        type=Path,
        help="Audio/video file or folder to transcribe.",
    )
    transcribe_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIRECTORY,
        type=Path,
        help=(
            "Directory for generated JSON files and optional SRT exports. "
            "Defaults to output."
        ),
    )
    transcribe_parser.add_argument(
        "--export-srt",
        action="store_true",
        help="Also export SRT subtitles beside the structured JSON output.",
    )
    transcribe_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable content-addressed result, stage, and normalized-audio reuse.",
    )
    transcribe_parser.add_argument(
        "--model-size",
        default=DEFAULT_WHISPER_MODEL_SIZE,
        help="faster-whisper model size. Defaults to turbo.",
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
    transcribe_parser.add_argument(
        "--enable-whisperx",
        action="store_true",
        default=True,
        help="Use WhisperX forced alignment after Whisper transcription (default).",
    )
    transcribe_parser.add_argument(
        "--disable-whisperx",
        action="store_false",
        dest="enable_whisperx",
        help="Disable WhisperX forced alignment.",
    )
    transcribe_parser.add_argument(
        "--whisperx-language",
        default=DEFAULT_WHISPERX_LANGUAGE,
        help="WhisperX alignment language code. Defaults to ja.",
    )
    transcribe_parser.add_argument(
        "--enable-homophone-resolver",
        action="store_true",
        help=(
            "Enable constrained same-reading semantic word correction with "
            "Sudachi and a Japanese masked language model. Disabled by default."
        ),
    )
    transcribe_parser.add_argument(
        "--enable-word-normalization",
        action="store_true",
        help="Normalize aligned Japanese tokens into learning words with Sudachi.",
    )
    transcribe_parser.add_argument(
        "--homophone-model-id",
        default=DEFAULT_HOMOPHONE_MODEL_ID,
        help=(
            "Masked language model id for homophone resolution. "
            f"Defaults to {DEFAULT_HOMOPHONE_MODEL_ID}."
        ),
    )
    transcribe_parser.add_argument(
        "--homophone-top-k",
        default=DEFAULT_HOMOPHONE_TOP_K,
        type=int,
        help=(
            "Number of language-model candidates to inspect for each token. "
            f"Defaults to {DEFAULT_HOMOPHONE_TOP_K}."
        ),
    )
    transcribe_parser.add_argument(
        "--homophone-score-margin",
        default=DEFAULT_HOMOPHONE_SCORE_MARGIN,
        type=float,
        help=(
            "Required score advantage over the original token before accepting "
            "a same-reading candidate. "
            f"Defaults to {DEFAULT_HOMOPHONE_SCORE_MARGIN}."
        ),
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
    writer = _build_writer(args)
    morphological_analyzer = (
        SudachiMorphologicalAnalyzer()
        if args.enable_word_normalization or args.enable_homophone_resolver
        else None
    )
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=FasterWhisperTranscriber(
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
        ),
        builder=WordSubtitleBuilder(),
        writer=writer,
        output_extension=DEFAULT_LISTENING_JSON_EXTENSION,
        aligner=_build_aligner(args),
        homophone_resolver=_build_homophone_resolver(args),
        overlap_text_cleaner=AuditableOverlapTextCleaner(
            morphological_analyzer=morphological_analyzer
        ),
        repeated_text_cleaner=AuditableRepeatedTextCleaner(
            morphological_analyzer=morphological_analyzer
        ),
        word_normalizer=_build_word_normalizer(args),
        sentence_boundary_resolver=JapaneseSentenceBoundaryResolver(
            morphological_analyzer=morphological_analyzer
        ),
        merger=ConservativeSubtitleMerger(),
        optimizer=LocalReadabilityOptimizer(),
        validator=DomainSubtitleValidator(),
        progress_reporter=ConsoleProgressReporter(output=error_output),
        artifact_recorder=StageArtifactStore(
            root_directory=args.output_dir / ".work",
        ),
        cache=(
            None
            if args.no_cache
            else LocalPipelineContextCache(args.output_dir / ".cache")
        ),
        audio_normalizer=FFmpegAudioNormalizer(),
        cache_namespace=(
            f"jp-learning-platform-{__version__}-sentence-provenance-v24"
        ),
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
        HomophoneResolverDependencyError,
        WordNormalizerDependencyError,
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


def _build_writer(args: Namespace) -> ListeningJsonWriter | CompositeSubtitleWriter:
    primary_writer = ListeningJsonWriter(output_directory=args.output_dir)
    if not args.export_srt:
        return primary_writer

    return CompositeSubtitleWriter(
        primary_writer=primary_writer,
        export_writers=(SrtSubtitleWriter(output_directory=args.output_dir),),
    )


def _build_aligner(
    args: Namespace,
) -> WhisperXAlignerAdapter | None:
    if args.enable_whisperx:
        base_aligner = WhisperXAlignerAdapter(
            device=args.device,
            language_code=args.whisperx_language,
        )
    else:
        base_aligner = None

    return base_aligner


def _build_homophone_resolver(args: Namespace) -> BertHomophoneResolver | None:
    if not args.enable_homophone_resolver:
        return None

    return BertHomophoneResolver(
        model_id=args.homophone_model_id,
        device=args.device,
        top_k=args.homophone_top_k,
        score_margin=args.homophone_score_margin,
    )


def _build_word_normalizer(args: Namespace) -> JapaneseLearningWordNormalizer | None:
    if not (args.enable_word_normalization or args.enable_homophone_resolver):
        return None
    return JapaneseLearningWordNormalizer()


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

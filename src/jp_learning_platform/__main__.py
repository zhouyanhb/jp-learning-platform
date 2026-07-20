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
    AcousticSentenceBoundaryResolver,
    CompositeSubtitleWriter,
    ConservativeSubtitleMerger,
    ConsoleProgressReporter,
    DEFAULT_KOTOBA_WHISPER_BATCH_SIZE,
    DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS,
    DEFAULT_KOTOBA_WHISPER_MODEL_ID,
    DEFAULT_LISTENING_JSON_EXTENSION,
    DEFAULT_HOMOPHONE_MODEL_ID,
    DEFAULT_HOMOPHONE_SCORE_MARGIN,
    DEFAULT_HOMOPHONE_TOP_K,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS,
    DEFAULT_REAZON_SPEECH_MODEL_ID,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_MODEL_SIZE,
    DEFAULT_WHISPERX_LANGUAGE,
    DisabledQwenRepairer,
    DiarizingWhisperXAligner,
    DomainSubtitleValidator,
    BertHomophoneResolver,
    FasterWhisperDependencyError,
    FasterWhisperTranscriber,
    HomophoneResolverDependencyError,
    JapaneseWordTimingNormalizer,
    KotobaWhisperDependencyError,
    KotobaWhisperTranscriber,
    LlamaCppQwenRepairer,
    ListeningJsonWriter,
    LocalReadabilityOptimizer,
    PassthroughQwenRepairer,
    PassthroughWhisperXAligner,
    PyannoteSpeakerDiarizer,
    ReazonSpeechDependencyError,
    ReazonSpeechTranscriber,
    SrtSubtitleWriter,
    StageArtifactStore,
    TorchVadSentenceBoundaryDetector,
    WhisperXAlignerAdapter,
    WordSubtitleBuilder,
)
from jp_learning_platform.workflow import (
    DuplicateSubtitleOutputError,
    SubtitlePipelineRunner,
    SubtitlePipelineRunnerError,
)

_PIPELINE_STAGES = (
    "Audio",
    "ASR Transcription",
    "WhisperX Alignment",
    "Sentence Boundary Detection",
    "Qwen Repair (disabled by default)",
    "Japanese Word Normalization",
    "Homophone Resolution (optional)",
    "Sentence Boundary Resolver",
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
        help="Generate structured listening JSON for an audio file or folder.",
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
        "--asr-backend",
        choices=("kotoba-whisper", "reazon-speech", "faster-whisper"),
        default="kotoba-whisper",
        help="ASR backend. Defaults to kotoba-whisper.",
    )
    transcribe_parser.add_argument(
        "--asr-model",
        default=DEFAULT_KOTOBA_WHISPER_MODEL_ID,
        help=(
            "ASR model id or alias. "
            "Use kotoba-whisper-v2.1 or kotoba-whisper-v2.0. "
            "With --asr-backend reazon-speech, defaults to "
            f"{DEFAULT_REAZON_SPEECH_MODEL_ID}. "
            f"Otherwise defaults to {DEFAULT_KOTOBA_WHISPER_MODEL_ID}."
        ),
    )
    transcribe_parser.add_argument(
        "--model-size",
        default=DEFAULT_WHISPER_MODEL_SIZE,
        help=(
            "faster-whisper model size. Used only with "
            "--asr-backend faster-whisper. Defaults to large-v3."
        ),
    )
    transcribe_parser.add_argument(
        "--device",
        default=DEFAULT_WHISPER_DEVICE,
        help="ASR device. Use cpu, cuda, cuda:0, or mps. Defaults to cpu.",
    )
    transcribe_parser.add_argument(
        "--compute-type",
        default=DEFAULT_WHISPER_COMPUTE_TYPE,
        help=(
            "ASR compute type for faster-whisper. Ignored by Kotoba Whisper "
            "and ReazonSpeech. "
            "Defaults to int8."
        ),
    )
    transcribe_parser.add_argument(
        "--asr-chunk-length-seconds",
        default=DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS,
        type=float,
        help=(
            "ASR chunk length in seconds for Kotoba Whisper and ReazonSpeech. "
            f"Defaults to {DEFAULT_KOTOBA_WHISPER_CHUNK_LENGTH_SECONDS}."
        ),
    )
    transcribe_parser.add_argument(
        "--asr-chunk-overlap-seconds",
        default=DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS,
        type=float,
        help=(
            "ReazonSpeech chunk overlap in seconds. "
            f"Defaults to {DEFAULT_REAZON_SPEECH_CHUNK_OVERLAP_SECONDS}."
        ),
    )
    transcribe_parser.add_argument(
        "--asr-batch-size",
        default=DEFAULT_KOTOBA_WHISPER_BATCH_SIZE,
        type=int,
        help=(
            "ASR batch size for Kotoba Whisper and ReazonSpeech NeMo fallback. "
            f"Defaults to {DEFAULT_KOTOBA_WHISPER_BATCH_SIZE}."
        ),
    )
    whisperx_group = transcribe_parser.add_mutually_exclusive_group()
    whisperx_group.add_argument(
        "--enable-whisperx",
        action="store_true",
        default=True,
        dest="enable_whisperx",
        help="Use WhisperX forced alignment after ASR transcription. Enabled by default.",
    )
    whisperx_group.add_argument(
        "--disable-whisperx",
        action="store_false",
        dest="enable_whisperx",
        help="Skip WhisperX forced alignment and keep Whisper timings.",
    )
    transcribe_parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Use pyannote.audio to assign speaker identifiers.",
    )
    transcribe_parser.add_argument(
        "--hf-token",
        help="Hugging Face token for pyannote.audio. Defaults to HF_TOKEN.",
    )
    transcribe_parser.add_argument(
        "--whisperx-language",
        default=DEFAULT_WHISPERX_LANGUAGE,
        help="WhisperX alignment language code. Defaults to ja.",
    )
    transcribe_parser.add_argument(
        "--disable-sentence-boundaries",
        action="store_true",
        help="Skip acoustic sentence boundary detection and final sentence splitting.",
    )
    transcribe_parser.add_argument(
        "--disable-word-normalization",
        action="store_true",
        help="Skip Japanese word timing normalization before sentence boundary detection.",
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
    qwen_group = transcribe_parser.add_mutually_exclusive_group()
    qwen_group.add_argument(
        "--enable-qwen",
        action="store_true",
        help=(
            "Enable local Qwen transcript repair. Disabled by default."
        ),
    )
    qwen_group.add_argument(
        "--disable-qwen",
        action="store_true",
        help="Keep local Qwen transcript repair disabled. This is the default.",
    )
    qwen_group.add_argument(
        "--qwen-model-path",
        default=None,
        type=Path,
        help=(
            "Local Qwen GGUF model path for transcript repair. "
            "Passing this option enables Qwen repair. "
            f"When --enable-qwen is used without this option, defaults to "
            f"{DEFAULT_QWEN_MODEL_PATH}."
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
    runner = SubtitlePipelineRunner(
        audio_loader=AudioLoader(),
        transcriber=_build_transcriber(args),
        builder=WordSubtitleBuilder(),
        writer=writer,
        output_extension=DEFAULT_LISTENING_JSON_EXTENSION,
        aligner=_build_aligner(args),
        word_normalizer=_build_word_normalizer(args),
        sentence_boundary_detector=_build_sentence_boundary_detector(args),
        repairer=_build_repairer(args),
        homophone_resolver=_build_homophone_resolver(args),
        sentence_boundary_resolver=_build_sentence_boundary_resolver(args),
        merger=ConservativeSubtitleMerger(),
        optimizer=LocalReadabilityOptimizer(),
        validator=DomainSubtitleValidator(),
        progress_reporter=ConsoleProgressReporter(output=error_output),
        artifact_recorder=StageArtifactStore(
            root_directory=args.output_dir / ".work",
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
        KotobaWhisperDependencyError,
        ReazonSpeechDependencyError,
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
) -> PassthroughWhisperXAligner | WhisperXAlignerAdapter | DiarizingWhisperXAligner:
    if args.enable_whisperx:
        base_aligner = WhisperXAlignerAdapter(
            device=args.device,
            language_code=args.whisperx_language,
        )
    else:
        base_aligner = PassthroughWhisperXAligner()

    if args.enable_diarization:
        return DiarizingWhisperXAligner(
            base_aligner=base_aligner,
            diarizer=PyannoteSpeakerDiarizer(auth_token=args.hf_token),
        )

    return base_aligner


def _build_transcriber(
    args: Namespace,
) -> KotobaWhisperTranscriber | ReazonSpeechTranscriber | FasterWhisperTranscriber:
    if args.asr_backend == "faster-whisper":
        return FasterWhisperTranscriber(
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
        )

    if args.asr_backend == "reazon-speech":
        model_id = args.asr_model
        if model_id == DEFAULT_KOTOBA_WHISPER_MODEL_ID:
            model_id = DEFAULT_REAZON_SPEECH_MODEL_ID
        return ReazonSpeechTranscriber(
            model_id=model_id,
            device=args.device,
            batch_size=args.asr_batch_size,
            chunk_length_seconds=args.asr_chunk_length_seconds,
            chunk_overlap_seconds=args.asr_chunk_overlap_seconds,
        )

    return KotobaWhisperTranscriber(
        model_id=args.asr_model,
        device=args.device,
        chunk_length_seconds=args.asr_chunk_length_seconds,
        batch_size=args.asr_batch_size,
    )


def _build_repairer(
    args: Namespace,
) -> DisabledQwenRepairer | PassthroughQwenRepairer | LlamaCppQwenRepairer:
    if args.disable_qwen:
        return DisabledQwenRepairer()

    if args.enable_qwen or args.qwen_model_path is not None:
        return LlamaCppQwenRepairer(
            model_path=args.qwen_model_path or DEFAULT_QWEN_MODEL_PATH,
        )

    return DisabledQwenRepairer()


def _build_sentence_boundary_detector(
    args: Namespace,
) -> TorchVadSentenceBoundaryDetector | None:
    if args.disable_sentence_boundaries:
        return None

    return TorchVadSentenceBoundaryDetector()


def _build_word_normalizer(args: Namespace) -> JapaneseWordTimingNormalizer | None:
    if args.disable_word_normalization:
        return None

    return JapaneseWordTimingNormalizer()


def _build_homophone_resolver(args: Namespace) -> BertHomophoneResolver | None:
    if not args.enable_homophone_resolver:
        return None

    return BertHomophoneResolver(
        model_id=args.homophone_model_id,
        device=args.device,
        top_k=args.homophone_top_k,
        score_margin=args.homophone_score_margin,
    )


def _build_sentence_boundary_resolver(
    args: Namespace,
) -> AcousticSentenceBoundaryResolver | None:
    if args.disable_sentence_boundaries:
        return None

    return AcousticSentenceBoundaryResolver()


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

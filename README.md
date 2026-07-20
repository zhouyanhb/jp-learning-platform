# JP Learning Platform

AI-powered Japanese learning platform focused on the Version 1.0 subtitle pipeline.

## Release

Current package version: `1.0.0`.

Release notes are maintained in `docs/release-1.0.md`.

## Scope

Version 1.0 is limited to the subtitle pipeline:

Audio -> Whisper -> WhisperX Alignment -> Sentence Boundary Detection -> Qwen Repair (disabled by default) -> Japanese Word Normalization -> Homophone Resolution (optional) -> Sentence Boundary Resolver -> Subtitle Builder -> Subtitle Merger -> Readability Optimizer -> Subtitle Validator -> Subtitle Writer

Features outside this pipeline are intentionally out of scope for Version 1.0.

## Requirements

- Python 3.12 or newer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

The default local transcription path uses Kotoba Whisper v2.1 and WhisperX.
Qwen repair code is kept available, but automatic Qwen text repair is disabled
by default.
If installing from extras instead of `requirements.txt`, install the needed
runtime support explicitly:

```bash
python -m pip install -e ".[asr]"
python -m pip install -e ".[align]"
python -m pip install -e ".[japanese]"
python -m pip install -e ".[homophone]"
python -m pip install -e ".[vad]"
python -m pip install -e ".[diarization]"
python -m pip install -e ".[qwen]"
```

## Run

```bash
python -m jp_learning_platform
python -m jp_learning_platform status
python -m jp_learning_platform --version
python -m jp_learning_platform transcribe audio.mp3
python -m jp_learning_platform transcribe ./audios
python -m jp_learning_platform transcribe audio.mp3 --export-srt
python -m jp_learning_platform transcribe audio.mp3 --asr-model kotoba-whisper-v2.0
python -m jp_learning_platform transcribe audio.mp3 --asr-backend reazon-speech
python -m jp_learning_platform transcribe audio.mp3 --asr-backend faster-whisper --model-size small --device cpu --compute-type int8
python -m jp_learning_platform transcribe audio.mp3 --disable-whisperx
python -m jp_learning_platform transcribe audio.mp3 --disable-word-normalization
python -m jp_learning_platform transcribe audio.mp3 --enable-homophone-resolver
python -m jp_learning_platform transcribe audio.mp3 --disable-sentence-boundaries
python -m jp_learning_platform transcribe audio.mp3 --enable-qwen
python -m jp_learning_platform transcribe audio.mp3 --disable-qwen
python -m jp_learning_platform transcribe audio.mp3 --enable-diarization
python -m jp_learning_platform transcribe audio.mp3 --qwen-model-path models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

The entrypoint reports the Version 1.0 subtitle pipeline status. External SDK
adapters are supplied through the tool registry and plugin system.

The `transcribe` command accepts either one audio file or a folder containing
audio files and writes structured `.json` files to `output/` by default. Use
`--export-srt` when an SRT file should be written beside the JSON output. Use
`--output-dir` only when a custom output directory is needed.
Kotoba Whisper v2.1 runs by default. Pass `--asr-model kotoba-whisper-v2.0`
when v2.0 should be used instead. Pass `--asr-backend reazon-speech` to use
`reazon-research/reazonspeech-nemo-v2`, or pass `--asr-backend faster-whisper`
only when the old faster-whisper adapter should be used. WhisperX forced
alignment runs by default; pass `--disable-whisperx` only when Whisper timings
should be kept without forced alignment.
The ReazonSpeech backend splits long audio into overlapping chunks before
transcription; tune this with `--asr-chunk-length-seconds` and
`--asr-chunk-overlap-seconds`.
Install the optional ReazonSpeech runtime before using that backend:

```bash
python -m pip install 'git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/nemo-asr'
```

The default Kotoba adapter uses the standard Transformers ASR pipeline and does
not load Kotoba's remote post-processing pipeline.
Acoustic sentence boundary detection runs after alignment and records pause
candidates by time. Japanese word timing normalization runs after the Qwen
repair workflow boundary, which is a no-op by default, using SudachiPy when
available to create final words such as `天気` from the current transcript text
and map them back to the aligned timing units. Pass
`--disable-word-normalization` only when raw ASR/aligner pieces should be kept.
Optional homophone resolution can run after word normalization with
`--enable-homophone-resolver`. It uses Sudachi readings/POS and a Japanese
masked language model to replace only same-reading words whose contextual score
beats the original token, so it is constrained to ASR-style homophone errors
instead of free rewriting. Final sentence splitting runs after word
normalization and optional homophone resolution; pass
`--disable-sentence-boundaries` to skip both boundary stages.
Local Qwen repair is disabled by default. Pass `--enable-qwen` to run it with
`models/Qwen2.5-14B-Instruct-Q4_K_M.gguf`, or pass `--qwen-model-path` to run
it with a different GGUF model. `--disable-qwen` is kept as an explicit no-op
for scripts that want to state the default.
Use `--enable-diarization` with a Hugging Face token from `HF_TOKEN` or
`--hf-token` when speaker identifiers should be assigned with pyannote.audio.

During transcription, the command reports the current file and pipeline stage
to stderr. Per-stage JSON artifacts are saved under
`output/.work/<run-name>/<audio-name>/`, while final listening JSON remains at
`output/<audio-name>.json`.

## Checks

```bash
python -m compileall src tests
python -m pytest
```

## Architecture

Version 1.0 architecture documentation is maintained in `docs/architecture.md`.
The source-level layer metadata is defined in `jp_learning_platform.architecture`.

## Domain Models

Core subtitle pipeline models are documented in `docs/domain-models.md`.

## Domain Services

Domain factories, validators, and repository boundaries are documented in
`docs/domain-services.md`.

## Workflow Runtime

Workflow orchestration primitives are documented in `docs/workflow-runtime.md`.

## Pipeline Configuration

Local pipeline configuration defaults are documented in
`docs/pipeline-configuration.md`.

## Local Audio Transcribe CLI

Local audio and folder transcription to structured JSON is documented in
`docs/local-audio-srt-cli.md`.

## Whisper Stage

Whisper transcription stage contracts are documented in `docs/whisper-stage.md`.

## WhisperX Alignment Stage

WhisperX alignment stage contracts are documented in
`docs/whisperx-alignment-stage.md`.

## Sentence Boundary Stages

Sentence boundary detection and resolution are documented in
`docs/sentence-boundary-stage.md`.

## Japanese Word Normalization

Japanese word timing normalization maps the current transcript text to final
Japanese word-level timings before sentence-boundary resolution. It is
documented in `docs/japanese-word-normalization-stage.md`.

## Qwen Repair Stage

Qwen repair stage contracts are documented in `docs/qwen-repair-stage.md`.

## Subtitle Builder Stage

Subtitle builder stage contracts are documented in
`docs/subtitle-builder-stage.md`.

## Subtitle Merger Stage

Subtitle merger stage contracts are documented in
`docs/subtitle-merger-stage.md`.

## Readability Optimizer Stage

Readability optimizer stage contracts are documented in
`docs/readability-optimizer-stage.md`.

## Subtitle Validator Stage

Subtitle validator stage contracts are documented in
`docs/subtitle-validator-stage.md`.

## Subtitle Writer Stage

Subtitle writer stage contracts are documented in
`docs/subtitle-writer-stage.md`.

## Tool Registry

External tool adapter resolution is documented in `docs/tool-registry.md`.

## Plugin System

Optional capability registration is documented in `docs/plugin-system.md`.

## Audio Loader

Local audio loading is documented in `docs/audio-loader.md`.

## Roadmap

The implementation roadmap is maintained in `.ai/ROADMAP.md`.

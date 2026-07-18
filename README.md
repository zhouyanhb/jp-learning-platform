# JP Learning Platform

AI-powered Japanese learning platform focused on the Version 1.0 subtitle pipeline.

## Release

Current package version: `1.0.0`.

Release notes are maintained in `docs/release-1.0.md`.

## Scope

Version 1.0 is limited to the subtitle pipeline:

Audio -> Whisper -> WhisperX Alignment -> Qwen Repair -> Subtitle Builder -> Subtitle Merger -> Readability Optimizer -> Subtitle Validator -> Subtitle Writer

Features outside this pipeline are intentionally out of scope for Version 1.0.

## Requirements

- Python 3.12 or newer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install ASR support when generating subtitles from audio:

```bash
python -m pip install -e ".[asr]"
python -m pip install -e ".[align]"
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
python -m jp_learning_platform transcribe audio.mp3 --model-size small --device cpu --compute-type int8
python -m jp_learning_platform transcribe audio.mp3 --enable-whisperx
python -m jp_learning_platform transcribe audio.mp3 --qwen-model-path models/qwen.gguf
```

The entrypoint reports the Version 1.0 subtitle pipeline status. External SDK
adapters are supplied through the tool registry and plugin system.

The `transcribe` command accepts either one audio file or a folder containing
audio files and writes structured `.json` files to `output/` by default. Use
`--export-srt` when an SRT file should be written beside the JSON output. Use
`--output-dir` only when a custom output directory is needed.

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

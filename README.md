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

## Run

```bash
python -m jp_learning_platform
```

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

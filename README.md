# JP Learning Platform

AI-powered Japanese learning platform focused on the Version 1.0 subtitle pipeline.

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

## Tool Registry

External tool adapter resolution is documented in `docs/tool-registry.md`.

## Roadmap

The implementation roadmap is maintained in `.ai/ROADMAP.md`.

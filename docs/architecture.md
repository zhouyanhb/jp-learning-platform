# Architecture

Version 1.0 uses Domain Driven Design, Clean Architecture, a workflow engine,
a tool registry, and a plugin system. The scope is the subtitle pipeline only.

## Runtime Flow

The subtitle pipeline flows through these responsibilities:

```text
Application
  -> Workflow
  -> Domain
  -> Infrastructure
  -> External SDKs
```

Application code starts use cases. Workflow code coordinates stages. Domain code
owns business rules. Infrastructure code adapts external tools. External SDKs
remain behind infrastructure or plugin boundaries.

## Dependency Direction

Project dependencies point inward toward the domain:

- Domain depends on no outer project layer.
- Application may depend on Domain.
- Workflow may depend on Application and Domain contracts.
- Infrastructure may depend on Workflow, Application, and Domain contracts.
- Plugins may depend on public contracts from the project layers they extend.

The workflow layer must never instantiate external SDKs directly. All external
tools are resolved through infrastructure adapters and, once implemented, the
tool registry.

## Domain Driven Design

Domain models and services represent the subtitle pipeline language. Core
business objects are introduced by the domain roadmap items and live under the
domain layer. Domain code must remain independent from infrastructure, workflow,
plugin, and SDK concerns.

## Workflow

Workflow code orchestrates stage execution. A stage has one responsibility and
communicates through the pipeline context once the domain model is introduced.
Workflow code coordinates execution order; validation and transformation rules
belong in the domain layer.

The workflow runtime primitives are documented in `docs/workflow-runtime.md`.
Runtime observers expose stage progress without coupling stages to console
output or file persistence.
The local audio SRT CLI runner is documented in `docs/local-audio-srt-cli.md`.
The Whisper transcription stage is documented in `docs/whisper-stage.md`.
The WhisperX alignment stage is documented in
`docs/whisperx-alignment-stage.md`.
The Qwen repair stage is documented in `docs/qwen-repair-stage.md`.
The subtitle builder stage is documented in `docs/subtitle-builder-stage.md`.
The subtitle merger stage is documented in `docs/subtitle-merger-stage.md`.
The readability optimizer stage is documented in
`docs/readability-optimizer-stage.md`.
The subtitle validator stage is documented in `docs/subtitle-validator-stage.md`.
The subtitle writer stage is documented in `docs/subtitle-writer-stage.md`.

## Infrastructure

Infrastructure wraps third-party libraries and processes such as Whisper,
WhisperX, Qwen, and FFmpeg. Adapters translate external SDK inputs and outputs
into project contracts without leaking third-party APIs into the domain.
Tool adapter resolution is handled by the registry documented in
`docs/tool-registry.md`.
Local audio loading is documented in `docs/audio-loader.md`.
Local SRT writing is implemented by the SRT subtitle writer adapter.
Local CLI quality adapters provide optional WhisperX alignment, optional Qwen
repair, subtitle merging, readability optimization, and final domain
validation while preserving the workflow stage contracts. Local CLI progress
reporting and JSON stage artifact storage are infrastructure adapters wired by
the entrypoint and runner.
Speaker metadata remains internal domain metadata for preserving dialogue
boundaries; SRT writing does not add speaker labels.

## Plugins

Plugins are optional capabilities. They register implementations through public
interfaces and must not change core workflow behavior by side effect. The
workflow layer depends on contracts, not concrete plugin implementations.
Plugin registration is documented in `docs/plugin-system.md`.

## Repository Layout

```text
docs/      Architecture and design documentation.
src/       Production Python package.
tests/     Deterministic automated tests.
scripts/   Maintenance and developer scripts.
diagrams/  Architecture diagrams.
```

The architecture boundary metadata is encoded in
`jp_learning_platform.architecture` so tests can verify the expected layer
packages and dependency rules.

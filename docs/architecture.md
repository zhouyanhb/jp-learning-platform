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

## Infrastructure

Infrastructure wraps third-party libraries and processes such as Whisper,
WhisperX, Qwen, and FFmpeg. Adapters translate external SDK inputs and outputs
into project contracts without leaking third-party APIs into the domain.
Tool adapter resolution is handled by the registry documented in
`docs/tool-registry.md`.

## Plugins

Plugins are optional capabilities. They register implementations through public
interfaces and must not change core workflow behavior by side effect. The
workflow layer depends on contracts, not concrete plugin implementations.

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

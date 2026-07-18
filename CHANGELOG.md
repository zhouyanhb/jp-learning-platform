# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Added local audio transcription CLI support for single audio files and
  folders with `output/` as the default output directory.
- Added CLI options for faster-whisper model size, device, and compute type.
- Added full local CLI quality workflow wiring for WhisperX alignment, Qwen
  repair, subtitle merging, readability optimization, and validation stages.
- Added per-file CLI progress logging for local audio transcription.
- Added per-stage JSON artifact persistence under `output/.work/` for local
  audio transcription.
- Added a conservative Qwen repair safety policy that rejects likely content
  additions or deletions before subtitle construction.
- Added internal speaker metadata preservation so subtitle merging keeps
  different speakers in separate SRT cues without displaying speaker labels.
- Added structured intensive-listening JSON as the default local CLI output,
  with SRT available through the optional `--export-srt` flag.
- Added centralized local pipeline configuration defaults for Whisper,
  WhisperX, Qwen repair, subtitle merging, and readability adapters.
- Added optional pyannote.audio speaker diarization for assigning speaker
  identifiers and splitting mixed-speaker subtitle runs.

### Fixed

- Made the package entrypoint report release status and version output when run
  directly.
- Passed pyannote diarization tokens with `use_auth_token` first while keeping
  a `token` fallback for newer pyannote APIs.

## [1.0.0] - 2026-07-17

### Changed

- Released Version 1.0.0 and synchronized package metadata.

### Added

- Added workflow subtitle writer stage contracts and orchestration tests.
- Added workflow subtitle validator stage contracts and orchestration tests.
- Added workflow readability optimizer stage contracts and orchestration tests.
- Added workflow subtitle merger stage contracts and orchestration tests.
- Added workflow subtitle builder stage contracts and orchestration tests.
- Added workflow Qwen repair stage contracts and orchestration tests.
- Added workflow WhisperX alignment stage contracts and orchestration tests.
- Added workflow Whisper stage contracts and orchestration tests.
- Added infrastructure audio loader for supported local audio files.
- Added audio loader tests and documentation.
- Added plugin system contracts for plugin metadata, activation, registration,
  context, and registry errors.
- Added plugin system tests and documentation.
- Added infrastructure tool registry for resolving external tool adapters.
- Added tool registry tests and documentation.
- Added workflow runtime primitives for stages, pipelines, workflows, stage
  results, and ordered execution.
- Added workflow runtime tests and documentation.
- Added domain services for model construction, document validation, and the
  document repository boundary.
- Added domain service tests and documentation.
- Added immutable subtitle pipeline domain models for words, sentences,
  segments, subtitles, documents, and pipeline context.
- Added domain model validation tests and documentation.
- Added Version 1.0 architecture documentation for DDD, Clean Architecture,
  workflow, infrastructure, and plugin boundaries.
- Added source-level architecture boundary metadata and package namespaces.
- Added tests for architecture package registration and dependency rules.
- Initialized the Python package structure for the JP Learning Platform.
- Added project metadata and pytest configuration.
- Added a minimal runnable package entrypoint.
- Added CI checks for compilation and tests.

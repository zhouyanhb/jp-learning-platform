# Release 1.0.0

Version 1.0.0 completes the frozen Version 1.0 subtitle pipeline scope.

## Scope

The release includes the full subtitle pipeline:

Audio -> Whisper -> WhisperX Alignment -> Qwen Repair -> Subtitle Builder -> Subtitle Merger -> Readability Optimizer -> Subtitle Validator -> Subtitle Writer

Features outside the subtitle pipeline remain out of scope for this release.

## Package Version

The package metadata declares version `1.0.0`.

The runtime package exposes the same version through
`jp_learning_platform.__version__`, including source-tree execution before the
package is installed.

## Validation

Release validation covers:

- Python compilation for source and tests
- Package module entrypoint execution
- Pytest test suite
- Git whitespace checks

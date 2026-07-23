# Session Log

> Append only.
> Never rewrite history.

---

# Session 001

Date

2026-07-16

Roadmap

Commit0001 Repository Initialize

Summary

Initialized repository foundation.

Changes

- Python project configuration
- pyproject.toml
- Package entrypoint
- README
- CHANGELOG
- GitHub Actions CI
- Initial tests

Validation

- compileall ✔
- package entry ✔
- pytest (2 passed)

Commit

d96937b

```
chore(repository): initialize project foundation
```

---

# Session 002

Date

2026-07-16

Roadmap

Commit0002 Architecture

Summary

Established project architecture and package boundaries.

Changes

- Architecture documentation
- Layer namespaces
- Architecture metadata
- Architecture tests

Validation

- compileall ✔
- package entry ✔
- pytest (11 passed)

Commit

6101aba

```
docs(architecture): define project boundaries
```

---

# Session 003

Date

2026-07-16

Roadmap

Commit0003 Domain Models

Summary

Implemented immutable domain models for the subtitle pipeline.

Changes

Added

- TimeRange
- Word
- Sentence
- Subtitle
- Segment
- Document
- PipelineContext

Documentation

- docs/domain-models.md

Tests

- test_domain_models.py

Validation

- compileall ✔
- package entry ✔
- pytest (21 passed)

Commit

fd0176a

```
feat(domain): add core subtitle models
```

---

# Session 004

Date

2026-07-16

Roadmap

Commit0004 Domain Services

Summary

Implemented domain services for model construction, document validation, and
the document repository boundary.

Changes

Added

- DomainModelFactory
- DocumentValidator
- ValidationResult
- ValidationIssue
- ValidationCode
- DomainValidationError
- DocumentRepository

Documentation

- docs/domain-services.md

Tests

- test_domain_services.py

Validation

- compileall ✔
- package entry ✔
- pytest (28 passed)

Commit

created by this commit

```
feat(domain): implement domain services
```

---

# Session 005

Date

2026-07-16

Roadmap

Commit0005 Workflow Runtime

Summary

Implemented the workflow runtime for ordered pipeline stage execution.

Changes

Added

- Stage
- StageResult
- Pipeline
- Workflow
- ExecutionEngine
- create_pipeline

Documentation

- docs/workflow-runtime.md

Tests

- test_workflow_runtime.py

Validation

- compileall ✔
- package entry ✔
- pytest (36 passed)

Commit

created by this commit

```
feat(workflow): add workflow runtime
```

---

# Session 006

Date

2026-07-17

Roadmap

Commit0006 Tool Registry

Summary

Implemented the infrastructure tool registry for resolving external tool
adapters by name.

Changes

Added

- RegisteredTool
- ToolRegistry
- ToolRegistryError
- DuplicateToolError
- ToolNotFoundError

Documentation

- docs/tool-registry.md

Tests

- test_tool_registry.py

Validation

- compileall ✔
- package entry ✔
- pytest (43 passed)

Commit

created by this commit

```
feat(infrastructure): add tool registry
```

---

# Session 007

Date

2026-07-17

Roadmap

Commit0007 Plugin System

Summary

Implemented the plugin system for optional capability registration and
activation.

Changes

Added

- PluginMetadata
- PluginRegistration
- PluginContext
- Plugin
- PluginRegistry
- PluginRegistryError
- DuplicatePluginError
- PluginNotFoundError

Documentation

- docs/plugin-system.md

Tests

- test_plugin_system.py

Validation

- compileall ✔
- package entry ✔
- pytest (52 passed)

Commit

created by this commit

```
feat(plugins): add plugin system
```

---

# Session 008

Date

2026-07-17

Roadmap

Commit0008 Audio Loader

Summary

Implemented the infrastructure audio loader for supported local audio files.

Changes

Added

- AudioFormat
- LoadedAudio
- AudioLoader
- AudioLoaderError
- AudioFileNotFoundError
- UnsupportedAudioFormatError
- EmptyAudioFileError

Documentation

- docs/audio-loader.md

Tests

- test_audio_loader.py

Validation

- compileall ✔
- package entry ✔
- pytest (60 passed)

Commit

created by this commit

```
feat(infrastructure): add audio loader
```

---

# Session 009

Date

2026-07-17

Roadmap

Commit0009 Whisper Stage

Summary

Implemented the workflow Whisper stage for coordinating transcription through
an injected transcriber contract.

Changes

Added

- WhisperTranscriptionRequest
- WhisperTranscript
- WhisperTranscriber
- WhisperStage
- WhisperStageError
- InvalidWhisperTranscriberError
- InvalidWhisperTranscriptError

Documentation

- docs/whisper-stage.md

Tests

- test_whisper_stage.py

Validation

- compileall ✔
- package entry ✔
- pytest (68 passed)

Commit

created by this commit

```
feat(workflow): add whisper stage
```

---

# Session 010

Date

2026-07-17

Roadmap

Commit0010 WhisperX Alignment

Summary

Implemented the workflow WhisperX alignment stage for coordinating segment
alignment through an injected aligner contract.

Changes

Added

- WhisperXAlignmentRequest
- WhisperXAlignment
- WhisperXAligner
- WhisperXAlignmentStage
- WhisperXAlignmentStageError
- InvalidWhisperXAlignerError
- MissingWhisperSegmentsError
- InvalidWhisperXAlignmentError

Documentation

- docs/whisperx-alignment-stage.md

Tests

- test_whisperx_alignment_stage.py

Validation

- compileall ✔
- package entry ✔
- pytest (77 passed)

Commit

created by this commit

```
feat(workflow): add whisperx alignment stage
```

---

# Session 011

Date

2026-07-17

Roadmap

Commit0011 Qwen Repair

Summary

Implemented the workflow Qwen repair stage for coordinating aligned transcript
repair through an injected repairer contract.

Changes

Added

- QwenRepairRequest
- QwenRepair
- QwenRepairer
- QwenRepairStage
- QwenRepairStageError
- InvalidQwenRepairerError
- MissingAlignedSegmentsError
- InvalidQwenRepairError

Documentation

- docs/qwen-repair-stage.md

Tests

- test_qwen_repair_stage.py

Validation

- compileall ✔
- package entry ✔
- pytest (86 passed)

Commit

created by this commit

```
feat(workflow): add qwen repair stage
```

---

# Session 012

Date

2026-07-17

Roadmap

Commit0012 Subtitle Builder

Summary

Implemented the workflow subtitle builder stage for coordinating subtitle
construction from repaired transcript segments through an injected builder
contract.

Changes

Added

- SubtitleBuildRequest
- SubtitleBuild
- SubtitleBuilder
- SubtitleBuilderStage
- SubtitleBuilderStageError
- InvalidSubtitleBuilderError
- MissingSubtitleBuildSegmentsError
- InvalidSubtitleBuildError

Documentation

- docs/subtitle-builder-stage.md

Tests

- test_subtitle_builder_stage.py

Validation

- compileall ✔
- package entry ✔
- pytest (95 passed)

Commit

created by this commit

```
feat(workflow): add subtitle builder stage
```

---

# Session 013

Date

2026-07-17

Roadmap

Commit0013 Subtitle Merger

Summary

Implemented the workflow subtitle merger stage for coordinating built subtitle
merging through an injected merger contract.

Changes

Added

- SubtitleMergeRequest
- SubtitleMerge
- SubtitleMerger
- SubtitleMergerStage
- SubtitleMergerStageError
- InvalidSubtitleMergerError
- MissingSubtitlesToMergeError
- InvalidSubtitleMergeError

Documentation

- docs/subtitle-merger-stage.md

Tests

- test_subtitle_merger_stage.py

Validation

- compileall ✔
- package entry ✔
- pytest (104 passed)

Commit

created by this commit

```
feat(workflow): add subtitle merger stage
```

---

# Session 014

Date

2026-07-17

Roadmap

Commit0014 Readability Optimizer

Summary

Implemented the workflow readability optimizer stage for coordinating subtitle
readability optimization through an injected optimizer contract.

Changes

Added

- ReadabilityOptimizationRequest
- ReadabilityOptimization
- ReadabilityOptimizer
- ReadabilityOptimizerStage
- ReadabilityOptimizerStageError
- InvalidReadabilityOptimizerError
- MissingSubtitlesToOptimizeError
- InvalidReadabilityOptimizationError

Documentation

- docs/readability-optimizer-stage.md

Tests

- test_readability_optimizer_stage.py

Validation

- compileall ✔
- package entry ✔
- pytest (113 passed)

Commit

created by this commit

```
feat(workflow): add readability optimizer stage
```

---

# Session 015

Date

2026-07-17

Roadmap

Commit0015 Subtitle Validator

Summary

Implemented the workflow subtitle validator stage for coordinating optimized
subtitle validation through an injected validator contract.

Changes

Added

- SubtitleValidationRequest
- SubtitleValidation
- SubtitleValidator
- SubtitleValidatorStage
- SubtitleValidatorStageError
- InvalidSubtitleValidatorError
- MissingSubtitlesToValidateError
- InvalidSubtitleValidationError
- SubtitleValidationFailedError

Documentation

- docs/subtitle-validator-stage.md

Tests

- test_subtitle_validator_stage.py

Validation

- compileall ✔
- package entry ✔
- pytest (122 passed)

Commit

created by this commit

```
feat(workflow): add subtitle validator stage
```

---

# Session 016

Date

2026-07-17

Roadmap

Commit0016 Subtitle Writer

Summary

Implemented the workflow subtitle writer stage for coordinating validated
subtitle output through an injected writer contract.

Changes

Added

- SubtitleWriteRequest
- SubtitleWrite
- SubtitleWriter
- SubtitleWriterStage
- SubtitleWriterStageError
- InvalidSubtitleWriterError
- MissingSubtitlesToWriteError
- InvalidSubtitleWriteError

Documentation

- docs/subtitle-writer-stage.md

Tests

- test_subtitle_writer_stage.py

Validation

- compileall ✔
- package entry ✔
- pytest (130 passed)

Commit

created by this commit

```
feat(workflow): add subtitle writer stage
```

---

# Session 017

Date

2026-07-17

Roadmap

Commit0017 Release Version 1.0

Summary

Prepared the repository for the Version 1.0.0 subtitle pipeline release.

Changes

Updated

- Package metadata version
- Runtime package version fallback
- Changelog release section
- README release documentation
- Project state and roadmap completion state

Documentation

- docs/release-1.0.md

Tests

- test_package.py

Validation

- compileall ✔
- package entry ✔
- pytest (130 passed)

Commit

created by this commit

```
chore(release): prepare version 1.0.0
```

---

# Session 018

Date

2026-07-17

Roadmap

Maintenance CLI Runnable Entrypoint

Summary

Made the package entrypoint visibly runnable by reporting release status and
version output from the command line.

Changes

Updated

- Package command line entrypoint
- README run instructions
- Changelog unreleased fixes
- Project state

Tests

- test_package.py

Validation

- compileall ✔
- package entry ✔
- pytest (131 passed)

Commit

created by this commit

```
fix(cli): make package entrypoint runnable
```

---

# Session 019

Date

2026-07-17

Roadmap

Maintenance Local Audio SRT CLI

Summary

Implemented the first-stage local command line flow for generating SRT files
from a single audio file or a folder of audio files.

Changes

Added

- Local audio input discovery
- Subtitle pipeline request and result contracts
- Local audio-to-SRT workflow runner
- faster-whisper transcription adapter
- Word-aware subtitle builder adapter
- UTF-8 SRT subtitle writer adapter
- `transcribe` CLI command with `output/` as the default output directory

Documentation

- docs/local-audio-srt-cli.md
- README.md
- docs/architecture.md

Tests

- test_srt_subtitle_writer.py
- test_word_subtitle_builder.py
- test_subtitle_pipeline_runner.py
- test_package.py

Validation

- compileall ✔
- package entry ✔
- pytest (141 passed)

Commit

created by this commit

```
feat(cli): add local audio srt generation
```

---

# Session 020

Date

2026-07-17

Roadmap

Maintenance ASR CLI Model Options

Summary

Added command line options for configuring faster-whisper model size, device,
and compute type during local audio SRT generation.

Changes

Added

- `--model-size`
- `--device`
- `--compute-type`

Documentation

- docs/local-audio-srt-cli.md
- README.md

Tests

- test_package.py

Validation

- compileall ✔
- package entry ✔
- pytest (142 passed)

Commit

created by this commit

```
feat(cli): add asr model options
```

---

# Session 021

Date

2026-07-17

Roadmap

Maintenance Subtitle Quality CLI Stages

Summary

Connected the local audio SRT CLI to the full subtitle quality workflow while
keeping external WhisperX and Qwen model integrations optional.

Changes

Added

- WhisperX alignment adapter and pass-through alignment adapter
- Qwen repair adapter and pass-through repair adapter
- Local subtitle merger adapter
- Local readability optimizer adapter
- Domain subtitle validator adapter
- CLI options for WhisperX alignment and Qwen repair model selection
- Full quality stage wiring in the local subtitle pipeline runner

Documentation

- docs/local-audio-srt-cli.md
- docs/architecture.md
- README.md

Tests

- test_subtitle_pipeline_runner.py
- test_subtitle_quality_adapters.py
- test_package.py

Validation

- compileall ✔
- package entry ✔
- pytest (149 passed)

Commit

created by this commit

```
feat(cli): wire subtitle quality stages
```

---

# Session 022

Date

2026-07-17

Roadmap

Maintenance CLI Progress Artifacts

Summary

Added per-file progress logging and per-stage JSON artifact persistence for
local audio SRT generation.

Changes

Added

- Workflow stage execution observer events
- Workflow progress and stage artifact recording contracts
- Console progress reporter for local CLI transcription
- JSON stage artifact store with ordered stage filenames
- Audio-loader and workflow stage progress recording in the subtitle runner
- CLI wiring for progress logs and `output/.work/` artifacts

Documentation

- docs/local-audio-srt-cli.md
- docs/workflow-runtime.md
- docs/architecture.md
- README.md
- CHANGELOG.md

Tests

- test_workflow_runtime.py
- test_pipeline_progress_artifacts.py
- test_subtitle_pipeline_runner.py

Validation

- compileall ✔
- package entry ✔
- pytest (156 passed)

Commit

created by this commit

```
feat(cli): add progress artifacts
```

---

# Session 023

Date

2026-07-18

Roadmap

Maintenance Qwen Repair Safety

Summary

Added a conservative safety layer for Qwen transcript repair so model output
cannot silently add or remove spoken content before subtitle construction.

Changes

Added

- Qwen repair safety decision model
- Qwen repair safety policy
- Candidate repair fallback to original aligned text when safety limits fail
- Prompt constraints against adding, deleting, rewriting, summarizing, or
  explaining content

Documentation

- docs/qwen-repair-stage.md
- docs/local-audio-srt-cli.md
- CHANGELOG.md

Tests

- test_qwen_repair_safety.py

Validation

- compileall ✔
- package entry ✔
- pytest (161 passed)

Commit

created by this commit

```
feat(qwen): add repair safety policy
```

---

# Session 024

Date

2026-07-18

Roadmap

Maintenance Speaker Boundary Subtitles

Summary

Preserved internal speaker metadata through subtitle construction and quality
stages so short dialogue turns from different speakers remain separate while
SRT output stays label-free.

Changes

Added

- Optional speaker identifiers on word, sentence, segment, and subtitle domain
  models
- Speaker metadata propagation through Whisper, WhisperX, Qwen repair, subtitle
  building, merging, and readability optimization adapters
- Cross-speaker merge prevention in the conservative subtitle merger
- SRT writer coverage confirming speaker labels are not emitted

Documentation

- docs/domain-models.md
- docs/subtitle-builder-stage.md
- docs/subtitle-merger-stage.md
- docs/subtitle-writer-stage.md
- docs/local-audio-srt-cli.md
- docs/architecture.md
- CHANGELOG.md

Tests

- test_domain_models.py
- test_word_subtitle_builder.py
- test_subtitle_quality_adapters.py
- test_srt_subtitle_writer.py
- test_whisperx_alignment_stage.py
- test_qwen_repair_safety.py

Validation

- compileall ✔
- package entry ✔
- pytest (166 passed)

Commit

created by this commit

```
feat(subtitles): preserve speaker boundaries
```

---

# Session 025

Date

2026-07-18

Roadmap

Maintenance Listening JSON Default Output

Summary

Changed the local transcribe CLI so structured intensive-listening JSON is the
default output while SRT is generated only when explicitly requested.

Changes

Added

- Listening JSON writer adapter for segment, sentence, word, and subtitle
  timing output
- Composite subtitle writer adapter for optional export artifacts
- `--export-srt` CLI flag for writing SRT beside the primary JSON output
- Configurable primary output extension on the local subtitle pipeline runner

Documentation

- docs/local-audio-srt-cli.md
- docs/subtitle-writer-stage.md
- docs/architecture.md
- README.md
- CHANGELOG.md

Tests

- test_listening_json_writer.py
- test_subtitle_pipeline_runner.py
- test_package.py

Validation

- compileall ✔
- package entry ✔
- pytest (170 passed)

Commit

created by this commit

```
feat(cli): default to listening json
```

---

# Session 026

Date

2026-07-18

Roadmap

Maintenance Pipeline Configuration Defaults

Summary

Consolidated local pipeline runtime defaults after reviewing the earlier
`jp_project_backend` configuration and wired the current adapters to use the
new typed configuration objects.

Changes

Added

- Typed pipeline configuration defaults for Whisper, WhisperX, Qwen repair,
  Qwen repair safety, subtitle merging, and readability
- faster-whisper default options for best-of decoding, temperature, word
  timestamps, VAD, previous-text conditioning, and hallucination silence
  filtering
- Documentation for maintaining local pipeline defaults
- `output/` ignore rule for generated CLI artifacts

Updated

- Whisper, WhisperX, Qwen, subtitle quality, and readability adapters now
  derive their backwards-compatible constants from centralized config objects

Documentation

- docs/pipeline-configuration.md
- docs/local-audio-srt-cli.md
- docs/architecture.md
- README.md
- CHANGELOG.md
- .ai/ARCHITECTURE.md

Tests

- test_pipeline_config.py
- test_faster_whisper_transcriber.py

Validation

- compileall ✔
- pytest (173 passed)

Commit

created by this commit

```
feat(config): centralize pipeline defaults
```

---

# Session 027

Date

2026-07-18

Roadmap

Maintenance Pyannote Speaker Diarization

Summary

Added optional pyannote.audio speaker diarization so local transcription can
assign speaker identifiers from audio and split mixed-speaker word runs before
subtitle building.

Changes

Added

- pyannote.audio optional dependency group
- Pyannote speaker diarizer infrastructure adapter
- Diarizing WhisperX aligner wrapper
- Speaker turn model for diarization intervals
- CLI options for `--enable-diarization` and `--hf-token`
- Pyannote diarization defaults in centralized pipeline configuration

Documentation

- docs/local-audio-srt-cli.md
- docs/whisperx-alignment-stage.md
- docs/pipeline-configuration.md
- docs/architecture.md
- README.md
- CHANGELOG.md
- .ai/ARCHITECTURE.md

Tests

- test_pyannote_diarizer.py
- test_pipeline_config.py
- test_package.py

Validation

- compileall ✔
- package help ✔
- pytest (180 passed)

Commit

created by this commit

```
feat(diarization): add pyannote speaker labels
```

---

# Session 028

Date

2026-07-18

Roadmap

Maintenance Pyannote Auth Token Parameter

Summary

Updated pyannote pipeline loading to pass Hugging Face tokens with
`use_auth_token` first while retaining a `token` fallback for pyannote versions
that require the newer parameter name.

Changes

Updated

- Pyannote pipeline loading authentication parameter order
- Changelog and project state maintenance records

Tests

- test_pyannote_diarizer.py

Validation

- compileall ✔
- package entry ✔
- pytest (182 passed)

Commit

created by this commit

```
fix(diarization): pass pyannote token as use_auth_token
```

---

# Session 029

Date

2026-07-23

Roadmap

Maintenance Homophone Candidate Prefiltering and Benchmark

Summary

Reduced contextual homophone scoring work by risk-ranking same-reading targets
and limiting full candidate evaluation to three suspicious words per sentence.

Changes

- Kept Sudachi analysis enabled for every sentence.
- Filtered targets without same-reading lexical candidates.
- Added batched original-token contextual probability scoring.
- Ranked targets using contextual probability, ASR confidence, tokenizer
  vocabulary-rank frequency proxy, and lexical candidate count.
- Limited full contextual candidate scoring to three targets per sentence.
- Added end-to-end pipeline timing output and a reproducible benchmark report.

Benchmark

- homophone-resolution: 16.8859 s to 9.2291 s (45.35% reduction)
- pipeline total: 131.33 s to 118.60 s (9.69% reduction)
- contextual decisions: 189 to 81
- accepted corrections: 6 before and 6 after

Documentation

- docs/homophone-prefilter-benchmark.md
- docs/local-audio-srt-cli.md
- CHANGELOG.md

Tests

- test_homophone_resolver.py
- test_pipeline_config.py

Validation

- compileall ✔
- package entry ✔
- pytest (212 passed)
- before/after local audio benchmark ✔

Commit

created by this commit

```
perf(homophones): prefilter contextual candidates
```

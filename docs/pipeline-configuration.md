# Pipeline Configuration

Local subtitle pipeline defaults live in
`jp_learning_platform.infrastructure.pipeline_config`.

The module keeps runtime tuning values close to the infrastructure adapters
that use them while avoiding duplicated constants across Whisper, WhisperX,
Qwen repair, subtitle quality, and readability code.

## Configuration Groups

- `WhisperTranscriptionConfig` stores faster-whisper model and decoding
  defaults, including Japanese language selection, beam search, word timing,
  VAD, and hallucination silence filtering.
- `WhisperXAlignmentConfig` stores forced-alignment language defaults.
- `PyannoteDiarizationConfig` stores the speaker diarization model name and the
  Hugging Face token environment variable used by pyannote.audio.
- `QwenRepairConfig` stores llama.cpp generation defaults for local Qwen
  repair.
- `QwenRepairSafetyConfig` stores conservative acceptance thresholds for model
  repairs.
- `SubtitleMergeConfig` stores conservative subtitle merge timing, length, and
  Japanese terminal-mark defaults.
- `ReadabilityConfig` stores Japanese punctuation normalization defaults.

## Source

The defaults were consolidated after reviewing the earlier local
`jp_project_backend` configuration. Values already relied on by the current
pipeline were moved into typed config objects first so the repository remains
runnable and behavior changes stay explicit.

Adapter modules may continue exporting backwards-compatible `DEFAULT_*`
constants, but those constants should be derived from this configuration module.

# Pipeline Configuration

Local subtitle pipeline defaults live in
`jp_learning_platform.infrastructure.pipeline_config`.

The module keeps runtime tuning values close to the infrastructure adapters
that use them while avoiding duplicated constants across Whisper, WhisperX,
Qwen repair, subtitle quality, and readability code.

## Configuration Groups

- `WhisperTranscriptionConfig` stores ASR backend defaults. Kotoba Whisper v2.1
  is the default first-pass transcriber, with v2.0 selectable by model id.
  The same config keeps the ReazonSpeech NeMo v2 model id and faster-whisper
  fallback settings such as beam search, word timing, VAD, and hallucination
  silence filtering.
- `WhisperXAlignmentConfig` stores forced-alignment language defaults.
- `PyannoteDiarizationConfig` stores the speaker diarization model name and the
  Hugging Face token environment variable used by pyannote.audio.
- `QwenRepairConfig` stores the project-local GGUF model path and llama.cpp
  generation defaults used when local Qwen repair is explicitly enabled.
- `QwenRepairSafetyConfig` stores conservative acceptance thresholds for model
  repairs.
- `SentenceBoundaryDetectionConfig` stores acoustic pause detection defaults,
  including the minimum aligned word gap, VAD silence duration, sample rate,
  frame size, and energy threshold ratio.
- `SentenceBoundaryResolutionConfig` stores final sentence split
  thresholds for candidate confidence and minimum readable sentence size.
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

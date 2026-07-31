# Local Audio Transcribe CLI

Homophone candidate prefilter behavior and measured before/after runtime are
documented in `docs/homophone-prefilter-benchmark.md`.

Japanese sentence-boundary and cross-segment dependent-continuation behavior
is documented in `docs/japanese-sentence-boundary-resolution.md`.

The local transcribe CLI generates structured intensive-listening JSON from a
single audio/video file or a folder containing supported media. SRT output is
an optional export.

## Usage

```bash
python -m jp_learning_platform transcribe audio.mp3
python -m jp_learning_platform transcribe lesson.mp4
python -m jp_learning_platform transcribe ./audios
```

Supported video containers are AVI, M4V, MKV, MOV, MP4, and WebM. For video
input, FFmpeg extracts the first audio stream as deterministic mono 16 kHz PCM
before the existing audio loader and transcription stages run. A video without
a decodable audio stream fails with the FFmpeg error instead of producing an
empty transcription.

Generated `.json` files are written to `output/` by default. A custom output
directory can be supplied when needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --output-dir subtitles
```

Export SRT beside the structured JSON when a subtitle file is needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --export-srt
```

The command reports per-file stage progress while it runs. Progress is written
to stderr so stdout can continue to list the final generated JSON paths.

## Content-addressed Reuse and Audio Normalization

Caching is enabled by default under `<output-dir>/.cache`. A run identifies the
audio by SHA-256 content rather than its filename, and identifies processing by
the effective stage configuration and implementation. Reuse follows this
order:

1. For video, return the complete result keyed by the original video content.
2. Wait for an already-running identical video task when necessary.
3. On a video miss, reuse or create its deterministic extracted PCM WAV.
4. Hash that PCM and return or resume compatible audio-stage results, including
   results created from a different video container with the same extracted
   audio.
5. Run new model work only after both video-result and extracted-audio reuse
   miss.

Audio inputs use the same stage-level content cache directly, without the
video lookup and extraction steps.

The writer always materializes an output for the current input filename, but a
complete cache hit does not call the audio loader, FFmpeg, Whisper, or later
analysis stages. Whisper, pass-through alignment, and standard WhisperX retain
the original source because they can decode supported compressed audio
directly. Video is the exception: its audio is extracted before loading and
the extracted PCM is reused by Whisper and WhisperX. Both context and
audio writes are atomic, and failed stages are not recorded as successful cache
entries.

Use a cold run for benchmarking or troubleshooting when needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --no-cache
```

`--no-cache` disables cross-run result, stage, and normalized-audio reuse.

Progress includes separate `pipeline-cache`, `video-audio-extraction`,
`audio-content-cache`, `audio-loader`, `audio-normalization`, individual
model/quality stage, and `pipeline-total` durations. This makes cached and
uncached runs directly comparable.

For a multi-user service, place cache lookup behind the same authorization and
tenant boundary as the uploaded audio and generated result. Content identity
may safely deduplicate computation only when service policy permits it; a hash
match by itself is not authorization to expose another user's result. Cache
retention and capacity cleanup are deployment responsibilities for the local
`<output-dir>/.cache` directory.

ASR model settings can be supplied from the CLI:

```bash
python -m jp_learning_platform transcribe audio.mp3 --model-size small --device cpu --compute-type int8
```

Defaults are:

- `--model-size turbo`
- `--device cpu`
- `--compute-type int8`

Lower-level transcription defaults such as beam size, word timestamps, VAD,
and hallucination silence filtering are centralized in
`docs/pipeline-configuration.md`.

## Quality Stages

The CLI now runs the full subtitle quality workflow:

```text
Video audio extraction (video input only)
-> AudioLoader
-> WhisperStage
-> WhisperXAlignmentStage
-> HomophoneResolutionStage (optional)
-> SentenceBoundaryResolutionStage
-> WordNormalizationStage (optional)
-> SubtitleBuilderStage
-> SubtitleMergerStage
-> ReadabilityOptimizerStage
-> SubtitleValidatorStage
-> SubtitleWriterStage
```

WhisperX forced alignment is enabled by default. It can be disabled explicitly
when the alignment dependency is unavailable:

```bash
python -m jp_learning_platform transcribe audio.mp3 --disable-whisperx
```

Install the optional alignment dependency first:

```bash
python -m pip install -e ".[align]"
```

## Confidence-Gated Retries

The first Whisper pass is intentionally neutral: it fixes the language to
Japanese but does not use a domain prompt or carry previous decoded text across
windows. After that pass, only a bounded number of low-confidence segments are
transcribed again over their original time ranges. A retry may use the nearest
preceding high-confidence text from the same audio as context. Its output
replaces the first pass only when word confidence improves by the configured
margin; otherwise the original result is retained.

## ASR Dependency

The command uses the faster-whisper infrastructure adapter for speech
recognition. Install the optional ASR dependencies before running transcription:

```bash
python -m pip install -e ".[asr]"
```

## Pipeline

The first-stage local CLI uses the existing workflow contracts:

```text
AudioLoader
-> WhisperStage
-> WhisperXAlignmentStage
-> HomophoneResolutionStage (optional)
-> SentenceBoundaryResolutionStage
-> WordNormalizationStage (optional)
-> WordSubtitleBuilder
-> ConservativeSubtitleMerger
-> LocalReadabilityOptimizer
-> DomainSubtitleValidator
-> SubtitleWriterStage
-> ListeningJsonWriter
```

The generated subtitles preserve aligned-word timing through the domain
`Sentence` and `Word` objects. Sudachi writes separate `learning_words` after
sentence boundaries are resolved. The structured JSON schema is version 1.4
and contains both collections so study views never need to reinterpret the
authoritative audio timeline.

Optional SRT export writes the same text and timing without additional metadata.

## Progress and Stage Artifacts

Each processed audio file emits one-line progress events for every stage:

```text
[1/2] lesson.mp3 audio-loader started
[1/2] lesson.mp3 audio-loader done 0.01s -> output/.work/20260717_153012_123456/lesson/00_audio_load.json
[1/2] lesson.mp3 whisper started
[1/2] lesson.mp3 whisper done 12.48s -> output/.work/20260717_153012_123456/lesson/01_whisper.json
```

The final structured JSON remains at `output/<audio-name>.json`. When
`--export-srt` is supplied, the optional SRT export is written beside it as
`output/<audio-name>.srt`. Stage artifacts are saved under the output directory:

```text
output/.work/<run-name>/<audio-name>/manifest.json
output/.work/<run-name>/<audio-name>/00_pipeline_cache.json
output/.work/<run-name>/<audio-name>/00_audio_load.json
output/.work/<run-name>/<audio-name>/00_audio_normalization.json
output/.work/<run-name>/<audio-name>/01_whisper.json
output/.work/<run-name>/<audio-name>/02_align.json
output/.work/<run-name>/<audio-name>/02a_forced_alignment.json
output/.work/<run-name>/<audio-name>/04_homophone_resolution.json
output/.work/<run-name>/<audio-name>/05_sentence_boundary_resolution.json
output/.work/<run-name>/<audio-name>/05a_punctuation_attribution.json
output/.work/<run-name>/<audio-name>/06_word_normalization.json
output/.work/<run-name>/<audio-name>/07_build.json
output/.work/<run-name>/<audio-name>/07a_subtitle_display_normalization.json
output/.work/<run-name>/<audio-name>/08_merge.json
output/.work/<run-name>/<audio-name>/09_readability.json
output/.work/<run-name>/<audio-name>/10_validate.json
output/.work/<run-name>/<audio-name>/11_write.json
```

Artifacts contain the source path, output path, file index, stage status,
elapsed time, recorded timestamp, pipeline context, and stage data when the
stage exposes additional data. The manifest records the current stage and the
latest stage artifact path for that audio file.

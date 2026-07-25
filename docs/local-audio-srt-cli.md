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

1. Return the complete cached analysis for identical content and configuration.
2. Wait for an already-running identical task, then read the result it created.
3. Resume after the latest compatible cached stage when only later processing
   changed.
4. Reuse video-extracted or normalized deterministic PCM WAV when an audio
   consumer still needs it.
5. Normalize and run the remaining stages only when no compatible entry exists.

The writer always materializes an output for the current input filename, but a
complete cache hit does not call the audio loader, FFmpeg, Whisper, or later
analysis stages. Whisper, pass-through alignment, and standard WhisperX retain
the original source because they can decode supported compressed audio
directly. FFmpeg conversion to mono 16 kHz PCM is deferred until an adapter
explicitly requires deterministic sample addressing, currently pyannote
diarization. Video is the exception: its audio is extracted before loading and
the extracted PCM is reused by Whisper, WhisperX, and pyannote. Whisper
completes and is cached before any later conversion, so a
conversion or diarization retry does not repeat transcription. Both context and
audio writes are atomic, and failed stages are not recorded as successful cache
entries.

Use a cold run for benchmarking or troubleshooting when needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --no-cache
```

`--no-cache` disables cross-run result, stage, and normalized-audio reuse. It
does not disable compatibility processing: when pyannote diarization is
enabled, the cold run still converts the source after Whisper and reports the
`audio-normalization` duration before alignment/diarization begins.

Progress includes separate `pipeline-cache`, `video-audio-extraction`,
`audio-loader`, `audio-normalization`, individual model/quality stage, and
`pipeline-total` durations. This makes cached and uncached runs directly
comparable.

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

- `--model-size large-v3`
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
-> QwenRepairStage
-> SubtitleBuilderStage
-> SubtitleMergerStage
-> ReadabilityOptimizerStage
-> SubtitleValidatorStage
-> SubtitleWriterStage
```

WhisperX and Qwen are external-model stages. By default, their pass-through
adapters keep the pipeline runnable without additional model files. Enable real
WhisperX alignment with:

```bash
python -m jp_learning_platform transcribe audio.mp3 --enable-whisperx
```

Install the optional alignment dependency first:

```bash
python -m pip install -e ".[align]"
```

Enable pyannote.audio speaker diarization when speaker identifiers should be
assigned automatically:

```bash
python -m jp_learning_platform transcribe audio.mp3 --enable-diarization
```

Install the optional diarization dependency and provide a Hugging Face token
accepted for the pyannote speaker diarization model:

```bash
python -m pip install -e ".[diarization]"
HF_TOKEN=hf_... python -m jp_learning_platform transcribe audio.mp3 --enable-diarization
```

The token can also be passed with `--hf-token`. Diarization runs inside the
existing WhisperX alignment workflow boundary: the configured aligner produces
timed segments first, then pyannote speaker turns are matched to words by time
overlap. When a sentence contains multiple speakers, it is split into
speaker-specific segment runs before subtitle building.

Enable local Qwen repair by passing a GGUF model path:

```bash
python -m jp_learning_platform transcribe audio.mp3 --qwen-model-path models/qwen.gguf
```

Local Qwen repair uses a conservative safety policy. If a model output appears
to add or remove spoken content, the repairer keeps the original aligned text
so subtitle timing and word timing remain authoritative.

Install the optional Qwen dependency first:

```bash
python -m pip install -e ".[qwen]"
```

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
-> QwenRepairStage
-> WordSubtitleBuilder
-> ConservativeSubtitleMerger
-> LocalReadabilityOptimizer
-> DomainSubtitleValidator
-> SubtitleWriterStage
-> ListeningJsonWriter
```

The generated subtitles preserve word-derived timing through the domain
`Sentence` and `Word` objects before writing the final structured JSON. The
JSON output contains segment, sentence, word, and subtitle timing so downstream
intensive-listening views can query unfamiliar words without parsing SRT text.

When upstream alignment data includes speaker identifiers, the pipeline keeps
different speakers in separate subtitle cues and prevents cross-speaker merging.
With `--enable-diarization`, speaker identifiers can be produced from the audio
itself by pyannote.audio. Speaker identifiers remain structured metadata.
Optional SRT export does not display speaker labels.

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
output/.work/<run-name>/<audio-name>/02b_pyannote_diarization.json
output/.work/<run-name>/<audio-name>/02c_speaker_assignment.json
output/.work/<run-name>/<audio-name>/03_repair.json
output/.work/<run-name>/<audio-name>/04_homophone_resolution.json
output/.work/<run-name>/<audio-name>/05_word_normalization.json
output/.work/<run-name>/<audio-name>/06_sentence_boundary_resolution.json
output/.work/<run-name>/<audio-name>/07_build.json
output/.work/<run-name>/<audio-name>/08_merge.json
output/.work/<run-name>/<audio-name>/09_readability.json
output/.work/<run-name>/<audio-name>/10_validate.json
output/.work/<run-name>/<audio-name>/11_write.json
```

Artifacts contain the source path, output path, file index, stage status,
elapsed time, recorded timestamp, pipeline context, and stage data when the
stage exposes additional data. The manifest records the current stage and the
latest stage artifact path for that audio file.

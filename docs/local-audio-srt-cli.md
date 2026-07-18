# Local Audio Transcribe CLI

The local audio transcribe CLI generates structured intensive-listening JSON
from a single audio file or a folder of audio files. SRT output is an optional
export.

## Usage

```bash
python -m jp_learning_platform transcribe audio.mp3
python -m jp_learning_platform transcribe ./audios
```

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
AudioLoader
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
Speaker identifiers remain structured metadata. Optional SRT export does not
display speaker labels.

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
output/.work/<run-name>/<audio-name>/00_audio_load.json
output/.work/<run-name>/<audio-name>/01_whisper.json
output/.work/<run-name>/<audio-name>/02_align.json
output/.work/<run-name>/<audio-name>/03_repair.json
output/.work/<run-name>/<audio-name>/04_build.json
output/.work/<run-name>/<audio-name>/05_merge.json
output/.work/<run-name>/<audio-name>/06_readability.json
output/.work/<run-name>/<audio-name>/07_validate.json
output/.work/<run-name>/<audio-name>/08_write.json
```

Artifacts contain the source path, output path, file index, stage status,
elapsed time, recorded timestamp, pipeline context, and stage data when the
stage exposes additional data. The manifest records the current stage and the
latest stage artifact path for that audio file.

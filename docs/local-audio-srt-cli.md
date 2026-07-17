# Local Audio SRT CLI

The local audio SRT CLI generates subtitle files from a single audio file or a
folder of audio files.

## Usage

```bash
python -m jp_learning_platform transcribe audio.mp3
python -m jp_learning_platform transcribe ./audios
```

Generated `.srt` files are written to `output/` by default. A custom output
directory can be supplied when needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --output-dir subtitles
```

ASR model settings can be supplied from the CLI:

```bash
python -m jp_learning_platform transcribe audio.mp3 --model-size small --device cpu --compute-type int8
```

Defaults are:

- `--model-size large-v3`
- `--device cpu`
- `--compute-type int8`

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
-> SrtSubtitleWriter
```

The generated subtitles preserve word-derived timing through the domain
`Sentence` and `Word` objects before writing the final SRT text.

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
-> WordSubtitleBuilder
-> SubtitleWriterStage
-> SrtSubtitleWriter
```

The generated subtitles preserve word-derived timing through the domain
`Sentence` and `Word` objects before writing the final SRT text.

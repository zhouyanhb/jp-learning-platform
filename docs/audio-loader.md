# Audio Loader

The audio loader is the first infrastructure boundary in the subtitle pipeline.
It loads a local audio file into an immutable value object that later pipeline
stages can pass to transcription adapters.

## Supported Formats

The loader recognizes audio format from the file extension. Supported formats
are:

- AAC
- FLAC
- M4A
- MP3
- OGG
- OPUS
- WAV

## Loaded Audio

`LoadedAudio` contains the source path, detected `AudioFormat`, and file bytes.
It does not decode, resample, transcribe, or call external tools. Those
responsibilities belong to later roadmap items.

## Failure Modes

The loader fails fast when:

- the file does not exist
- the path is not a file
- the extension is unsupported
- the file is empty

Typed exceptions make failures explicit for workflow stages and callers.

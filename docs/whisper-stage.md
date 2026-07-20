# Whisper Stage

The Whisper stage is the workflow boundary for the first transcription step in
the subtitle pipeline. It coordinates transcription without importing Whisper
SDKs or constructing external adapters directly.

## Stage Contract

`WhisperStage` accepts a configured `WhisperTranscriber`. The transcriber is a
protocol implemented by infrastructure or plugin adapters and receives a
`WhisperTranscriptionRequest` containing:

- the source audio path from the current document
- the pipeline working directory
- the pipeline run identifier

The transcriber returns a `WhisperTranscript`.

## Transcript

`WhisperTranscript` contains the source path and normalized domain `Segment`
objects. The stage validates that the returned source path matches the request,
then writes the transcript segments into a new immutable `Document` on the next
`PipelineContext`.

Existing subtitles are preserved so later stages can decide how to merge or
replace subtitle output.

## Failure Modes

The stage fails fast when:

- the configured transcriber has no callable `transcribe()` method
- the transcriber returns anything other than `WhisperTranscript`
- the returned transcript source path does not match the requested source path
- the pipeline context is not a `PipelineContext`

The stage does not decode audio, call external SDKs, or resolve tools by itself.
Those responsibilities remain outside the workflow layer.

The local CLI wires this contract to `KotobaWhisperTranscriber` by default,
using `kotoba-tech/kotoba-whisper-v2.1`. Pass
`--asr-model kotoba-whisper-v2.0` to use Kotoba Whisper v2.0, or
`--asr-backend reazon-speech` to use
`reazon-research/reazonspeech-nemo-v2`. Pass `--asr-backend faster-whisper` to
use the older faster-whisper adapter.

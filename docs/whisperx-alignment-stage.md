# WhisperX Alignment Stage

The WhisperX alignment stage is the workflow boundary for aligning Whisper
transcription segments into richer domain segments with sentence and word
timing.

## Stage Contract

`WhisperXAlignmentStage` accepts a configured `WhisperXAligner`. The aligner is
a protocol implemented by infrastructure or plugin adapters and receives a
`WhisperXAlignmentRequest` containing:

- the source audio path from the current document
- the pipeline working directory
- the pipeline run identifier
- the current document segments produced by the Whisper stage

The aligner returns a `WhisperXAlignment`.

## Alignment Output

`WhisperXAlignment` contains the source path and aligned domain `Segment`
objects. Aligned segments may include nested `Sentence` and `Word` values with
word-level timing and confidence.

The stage validates that:

- the document already has Whisper segments to align
- the aligner returns `WhisperXAlignment`
- the returned source path matches the request source path

After validation, the stage writes the aligned segments into a new immutable
`Document` on the next `PipelineContext`. Existing subtitles are preserved for
later subtitle-building and merge stages.

## Boundary

The workflow stage does not import WhisperX, call external SDKs, load models,
or resolve tools directly. Those responsibilities belong to infrastructure or
plugin adapters that implement the `WhisperXAligner` contract.

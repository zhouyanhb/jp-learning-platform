# Subtitle Writer Stage

The subtitle writer stage is the workflow boundary for producing a subtitle
artifact after validation.

## Stage Contract

`SubtitleWriterStage` accepts a configured `SubtitleWriter`. The writer is a
protocol implemented by infrastructure, plugin, or application adapters and
receives a `SubtitleWriteRequest` containing:

- the source audio path from the current document
- the pipeline working directory
- the pipeline run identifier
- the current transcript segments
- the validated subtitles

The writer returns a `SubtitleWrite`.

## Write Output

`SubtitleWrite` contains the source path and output path produced by the writer.
The local CLI uses `ListeningJsonWriter` as the primary writer so the default
artifact contains structured segment, sentence, word, and subtitle timing for
intensive-listening workflows. `CompositeSubtitleWriter` can attach optional
exports such as `SrtSubtitleWriter` while still returning the primary JSON
output path.

Speaker identifiers remain internal metadata. Structured JSON keeps the speaker
identifier fields for downstream decisions, while the local SRT writer emits
only subtitle text and timing, without speaker labels.
The stage validates that:

- the document already has subtitles to write
- the writer returns `SubtitleWrite`
- the returned source path matches the request source path

A successful write leaves the `PipelineContext` unchanged. The output artifact
path is owned by the writer adapter because the current domain context only
tracks transcript and subtitle state.

## Boundary

The workflow stage does not format SRT, VTT, ASS, or other subtitle files and
does not write to disk itself. Formatting, file naming, and filesystem behavior
belong to adapters that implement the `SubtitleWriter` contract.

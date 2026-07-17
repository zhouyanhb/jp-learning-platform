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

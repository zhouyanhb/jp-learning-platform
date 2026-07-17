# Subtitle Builder Stage

The subtitle builder stage is the workflow boundary for turning repaired
transcript segments into subtitle entries.

## Stage Contract

`SubtitleBuilderStage` accepts a configured `SubtitleBuilder`. The builder is a
protocol implemented by infrastructure, plugin, or application adapters and
receives a `SubtitleBuildRequest` containing:

- the source audio path from the current document
- the pipeline working directory
- the pipeline run identifier
- the repaired document segments

The builder returns a `SubtitleBuild`.

## Build Output

`SubtitleBuild` contains the source path and built domain `Subtitle` objects.
The stage validates that:

- the document already has repaired segments to build from
- the builder returns `SubtitleBuild`
- the returned source path matches the request source path
- at least one subtitle is produced

After validation, the stage writes the built subtitles into a new immutable
`Document` on the next `PipelineContext`. Existing segments are preserved so
later stages can merge, optimize, validate, and write subtitles with access to
the repaired transcript context.

## Boundary

The workflow stage does not decide subtitle splitting rules, readability rules,
or output formatting. Those responsibilities belong to the configured
`SubtitleBuilder` and later dedicated pipeline stages.

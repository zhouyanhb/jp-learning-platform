# Subtitle Merger Stage

The subtitle merger stage is the workflow boundary for combining built
subtitles before readability optimization.

## Stage Contract

`SubtitleMergerStage` accepts a configured `SubtitleMerger`. The merger is a
protocol implemented by infrastructure, plugin, or application adapters and
receives a `SubtitleMergeRequest` containing:

- the source audio path from the current document
- the pipeline working directory
- the pipeline run identifier
- the current repaired segments
- the current built subtitles

The merger returns a `SubtitleMerge`.

## Merge Output

`SubtitleMerge` contains the source path and merged domain `Subtitle` objects.
Local conservative merging never merges adjacent subtitles when their speaker
identifiers differ, even when both cues are short and close together.
It also uses Japanese morphological analysis to treat complete predicate
endings as hard boundaries even when Qwen did not add a final period, so a
complete sentence is not joined to the next prompt or heading.
The stage validates that:

- the document already has subtitles to merge
- the merger returns `SubtitleMerge`
- the returned source path matches the request source path
- at least one subtitle is produced

After validation, the stage writes the merged subtitles into a new immutable
`Document` on the next `PipelineContext`. Existing segments are preserved so
later stages can optimize and validate subtitles with access to the transcript
context.

## Boundary

The workflow stage does not decide merge heuristics, timing thresholds, or
readability constraints. Those responsibilities belong to the configured
`SubtitleMerger` and later dedicated pipeline stages.

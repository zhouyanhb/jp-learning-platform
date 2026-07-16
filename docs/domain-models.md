# Domain Models

The domain layer contains immutable business objects for the subtitle pipeline.
These models describe the language shared by workflow stages and infrastructure
adapters without depending on either layer.

## TimeRange

`TimeRange` represents a media interval in seconds. Start and end times must be
finite, non-negative values, and the end time must not be earlier than the start
time.

## Word

`Word` represents a recognized token from speech recognition or alignment. It
stores normalized text, a time range, and an optional confidence score from
`0.0` to `1.0`.

## Sentence

`Sentence` groups words into readable text. When words are present, each word
must fall within the sentence time range.

## Segment

`Segment` represents an ordered transcript interval. It contains transcript text
and may contain sentence objects produced by upstream processing.

## Subtitle

`Subtitle` represents a subtitle cue ready for validation or writing. Subtitle
indexes are one-based to match subtitle file conventions.

## Document

`Document` represents the pipeline document being processed. It owns the source
path plus the current segment and subtitle collections.

## PipelineContext

`PipelineContext` is the immutable value passed between workflow stages. It
identifies a pipeline run, the document being processed, and the working
directory available to the run.

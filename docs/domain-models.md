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
`0.0` to `1.0`. These aligned words remain the authoritative audio timeline and are not
replaced by learning-word normalization.

## LearningWord

`LearningWord` represents a learning-oriented unit created by Sudachi after
sentence boundaries are resolved. It stores its character span in sentence
text, the contiguous aligned-word indexes that support it, a derived time
range, and whether that timing required character-level estimation.

## Sentence

`Sentence` groups aligned words and learning words into readable text. Both
collections must fall within the sentence time range. Aligned `words` support
timing and boundary decisions; `learning_words` support lookup and study UI.
ASR text boundaries are retained as indexes of the aligned word that starts
after each boundary, so later text reconstruction cannot erase boundary evidence.

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

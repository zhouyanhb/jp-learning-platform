# Subtitle Validator Stage

The subtitle validator stage is the workflow boundary for checking optimized
subtitles before they are written to disk.

## Stage Contract

`SubtitleValidatorStage` accepts a configured `SubtitleValidator`. The validator
is a protocol implemented by domain, infrastructure, plugin, or application
adapters and receives a `SubtitleValidationRequest` containing:

- the source audio path from the current document
- the pipeline working directory
- the pipeline run identifier
- the current transcript segments
- the current optimized subtitles

The validator returns a `SubtitleValidation`.

## Validation Output

`SubtitleValidation` contains the source path and a domain `ValidationResult`.
The stage validates that:

- the document already has subtitles to validate
- the validator returns `SubtitleValidation`
- the returned source path matches the request source path

When the validation result contains issues, the stage raises
`SubtitleValidationFailedError` and exposes the reported `ValidationIssue`
values. A valid result leaves the `PipelineContext` unchanged and allows the
workflow to continue to subtitle writing.

## Boundary

The workflow stage does not own validation rules. Rule checks belong to domain
validators or configured adapters that implement the `SubtitleValidator`
contract.

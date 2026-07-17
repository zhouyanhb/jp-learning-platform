# Readability Optimizer Stage

The readability optimizer stage is the workflow boundary for making merged
subtitles easier to read before validation.

## Stage Contract

`ReadabilityOptimizerStage` accepts a configured `ReadabilityOptimizer`. The
optimizer is a protocol implemented by infrastructure, plugin, or application
adapters and receives a `ReadabilityOptimizationRequest` containing:

- the source audio path from the current document
- the pipeline working directory
- the pipeline run identifier
- the current transcript segments
- the current merged subtitles

The optimizer returns a `ReadabilityOptimization`.

## Optimization Output

`ReadabilityOptimization` contains the source path and optimized domain
`Subtitle` objects. The stage validates that:

- the document already has subtitles to optimize
- the optimizer returns `ReadabilityOptimization`
- the returned source path matches the request source path
- at least one subtitle is produced

After validation, the stage writes the optimized subtitles into a new immutable
`Document` on the next `PipelineContext`. Existing segments are preserved so
later validation can compare subtitle output against transcript context.

## Boundary

The workflow stage does not decide reading-speed thresholds, line-length
limits, or subtitle splitting heuristics. Those responsibilities belong to the
configured `ReadabilityOptimizer` and the later validation stage.

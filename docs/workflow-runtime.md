# Workflow Runtime

The workflow runtime coordinates subtitle pipeline stages without owning
business rules. Domain validation remains in the domain layer, while external
tools remain behind infrastructure and plugin boundaries.

## Stage

`Stage` is a protocol for one unit of workflow work. A stage has a name and a
`run()` method that accepts a `PipelineContext` and returns a `StageResult`.

## StageResult

`StageResult` records the stage name and the context produced by that stage.
The execution engine uses the result context as the input to the next stage.

## Pipeline

`Pipeline` is an immutable ordered collection of stages. It validates that each
stage has a non-empty name and a callable `run()` method.

## Workflow

`Workflow` names a runnable pipeline. It does not execute by itself; execution
is handled by the engine so orchestration behavior stays in one place.

## Execution Engine

`ExecutionEngine` runs every stage in order. It propagates stage exceptions
without swallowing them and fails fast if a stage returns anything other than a
`StageResult`.

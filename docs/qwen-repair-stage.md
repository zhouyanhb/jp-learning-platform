# Qwen Repair Stage

The Qwen repair stage is the workflow boundary for repairing aligned transcript
segments before subtitle construction. It coordinates the repair step without
calling Qwen APIs or importing external SDKs directly.

## Stage Contract

`QwenRepairStage` accepts a configured `QwenRepairer`. The repairer is a
protocol implemented by infrastructure or plugin adapters and receives a
`QwenRepairRequest` containing:

- the source audio path from the current document
- the pipeline working directory
- the pipeline run identifier
- the current aligned document segments

The repairer returns a `QwenRepair`.

## Repair Output

`QwenRepair` contains the source path and repaired domain `Segment` objects.
Repair adapters may normalize transcript text while preserving timing carried
by segments, sentences, and words.

## Repair Safety

Infrastructure Qwen adapters apply a conservative repair safety policy before
accepting model output. The policy allows low-risk changes such as punctuation,
whitespace cleanup, and small typo corrections, but rejects candidate text that
appears to add or delete spoken content.

When a candidate repair is rejected, the adapter falls back to the original
segment text. Segment, sentence, and word timing remain anchored to the aligned
audio timeline, so Qwen repair cannot silently introduce words that have no
corresponding timing data.

The stage validates that:

- the document already has aligned segments to repair
- the repairer returns `QwenRepair`
- the returned source path matches the request source path

After validation, the stage writes the repaired segments into a new immutable
`Document` on the next `PipelineContext`. Existing subtitles are preserved for
later subtitle-building and merge stages.

## Boundary

The workflow stage does not call Qwen, manage prompts, resolve tools, or handle
credentials. Those responsibilities belong to infrastructure or plugin adapters
that implement the `QwenRepairer` contract.

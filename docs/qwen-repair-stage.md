# Qwen Repair Stage

The Qwen repair stage is the workflow boundary for repairing aligned transcript
segments before subtitle construction. It coordinates the repair step without
calling Qwen APIs or importing external SDKs directly.

The local CLI keeps this boundary in the pipeline but wires a disabled no-op
repairer by default. In the default path, no Qwen model is loaded and transcript
text is not changed by this stage. Use `--enable-qwen` or `--qwen-model-path`
only when local Qwen repair should participate.

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

`QwenRepair` contains the source path, repaired domain `Segment` objects, and
optional diagnostic decisions for each model candidate. Stage artifacts record
these decisions under `data.decisions`, including raw model text, the applied
candidate text, selected text, whether the safety policy accepted it, and the
rejection or acceptance reason.
When the local CLI uses the disabled adapter, each decision records
`reason=qwen_disabled`, empty `raw_text`, and unchanged candidate and selected
text.
Repair adapters may normalize transcript text while preserving timing carried
by segments, sentences, and words.
Local Qwen repair explicitly allows meaning-preserving Japanese punctuation so
the later sentence boundary resolver can use repaired text as a semantic signal
without moving word timing.
The local prompt only includes the current segment's transcript text and asks
the model for structured JSON edit candidates, not a rewritten full subtitle
line. It does not include complete previous or next segment text; any context is
limited to controlled metadata so model output cannot copy neighboring raw
transcript content from the prompt.
Before prompting or passing through repair output, infrastructure repairers also
remove tightly adjacent duplicate function words at segment boundaries, such as
a trailing `です` being repeated at the start of the next segment.

## Repair Safety

Infrastructure Qwen adapters apply a conservative repair safety policy before
accepting model output. The policy allows low-risk changes such as punctuation,
whitespace cleanup, and small typo corrections, but rejects candidate text that
appears to add or delete spoken content.
It also rejects non-phonetic expression rewrites: a candidate may replace a
suspect phrase only when the replacement still looks like an ASR correction by
surface similarity or Japanese reading, not when it is merely a more natural
synonym or paraphrase.
Short deletions are also rejected when they remove meaningful Japanese text,
letters, or numbers. A suspicious phrase should be replaced with a
pronunciation- and context-compatible phrase rather than removed.
The adapter does not hard-code domain-specific replacements. If the model
proposes a replacement, the safety policy validates it with generic checks such
as length, deletion risk, surface similarity, and Japanese reading similarity.
If the model deletes a suspicious word instead of proposing a safe replacement,
the adapter keeps the original text.
Structured repair accepts only replacement edits and punctuation insertions.
Delete, insert, and rewrite operations are invalid. Each replacement must point
to an exact substring of the current segment. A replacement edit may include
multiple candidates with declared readings; candidates are tried in order and
each accepted replacement is validated before it is applied. If a candidate is
rejected, the adapter can still try later candidates for the same source text.
The safety policy also protects compact information units such as ASCII letters
and digits, so level names, numbers, and similar identifiers are not silently
rewritten into ordinary words. The adapter does not ship an internal candidate
vocabulary or replacement list; candidates must come from model output or a
future explicit retrieval provider, and every candidate must pass the same
generic safety checks. If an edit is rejected but punctuation entries can still
be applied without changing spoken content, the adapter records the raw JSON
output and keeps only the safe punctuation in the selected text.

When a structured replacement edit is rejected, the adapter can still apply
separate punctuation entries from the same JSON response if those entries do not
change spoken content. Diagnostic records keep the raw JSON in `raw_text`, the
applied candidate in `candidate_text`, and the final selected text in
`selected_text`. If no safe edit can be isolated, the adapter falls back to the
original segment text. Segment, sentence, and word timing remain anchored to the
aligned audio timeline, so Qwen repair cannot silently introduce words that have
no corresponding timing data.

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

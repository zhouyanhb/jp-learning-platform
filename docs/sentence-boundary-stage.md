# Sentence Boundary Stages

Sentence boundary handling is split into two workflow stages so acoustic
evidence and the current transcript text can both contribute without moving
subtitle timing off aligned word boundaries.

## Detection

`SentenceBoundaryDetectionStage` runs after WhisperX alignment. Its detector
receives aligned domain segments and returns `SentenceBoundaryCandidate`
objects. Each candidate records:

- the segment position
- the word index after which a sentence may end
- the pause time range between adjacent words
- a boundary timestamp inside that pause
- an acoustic confidence score
- the detector source

The local detector uses torch/torchaudio waveform energy to confirm that an
aligned word gap contains low-energy non-speech. If the audio backend cannot
decode the file, it can fall back to aligned word-gap candidates and marks the
candidate source accordingly.

## Resolution

`SentenceBoundaryResolverStage` runs after the optional Qwen repair boundary.
Its resolver applies high-confidence acoustic candidates to the current
segments, using punctuation when it cleanly matches the candidate count and
otherwise falling back to aligned word text. Final sentence timing remains
anchored to the first and last word in each resolved sentence.

Acoustic pauses are treated as candidates, not mandatory cuts. Before applying
a candidate, the local resolver checks whether the text on the left side is a
complete Japanese sentence. It uses SudachiPy part-of-speech information when
available, with a conservative word-level fallback, so pauses after dependent
condition clauses, topic particles, or selection ranges remain attached to the
following predicate.
The resolver also rejects candidates that would split a fixed connection
expression across the boundary, such as `それ` / `から`, because the pause is
inside one discourse connector rather than between two sentences.

This order lets audio propose candidate sentence boundaries before optional
model repair while letting punctuation and cleaner text make the final sentence
split easier to read when such text is available.

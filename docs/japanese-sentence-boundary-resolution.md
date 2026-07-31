# Japanese Sentence Boundary Resolution

The local Japanese sentence boundary resolver runs before subtitle building.
It uses aligned word timing, terminal punctuation, sentence-final expressions,
timing evidence and Japanese dependent continuations to preserve grammatical
sentence units.

## Alignment-held silence

Some forced aligners represent a pause by extending the preceding word until
the next word starts. The resolver recognizes unusually long per-character
word durations as a boundary signal. It caps only the boundary word's sentence
time to a conservative speech duration; the original segment time remains
available, and the following sentence retains its original start time.

An aligned `か` is treated as a question boundary only when timing supplies
independent evidence. An abnormally extended alignment is sufficient; a
normal held pause is sufficient only when the preceding clause already ends
in a configured complete sentence-final form. This structural rule avoids
lexical exclusion lists and preserves indefinite, listing, modal, and embedded
question constructions. A confirmed boundary receives a question mark when
the aligned transcript omitted one. Both Japanese and ASCII question and
exclamation marks protect that boundary from later subtitle merging.

## Connective clauses

An adjacent clause ending in the connective forms `て` or `で` is joined to
its following main clause when the gap and speaker agree. A perceptible pause
is represented with `、`; tightly contiguous morphology such as `話して` plus
`います` is joined without punctuation.

## Cross-segment dependent continuations

Adjacent segments are merged when all of these conditions hold:

- the next segment begins with a configured dependent continuation such as
  `とき`, `場合`, or `ため`;
- the gap is non-negative and no greater than the configured maximum;
- the preceding sentence has no terminal punctuation;
- both sentences belong to the same speaker, including the unlabeled case.

The merge combines sentence text, aligned words, and time ranges before
subtitle construction. Remaining segment positions are reindexed from zero.
The resolver does not merge across terminal punctuation, speaker changes, or
long pauses.

For example:

```text
学生は授業を休んだ
とき、どのように宿題を確認しますか?
```

becomes one sentence with the union of both time ranges:

```text
学生は授業を休んだとき、どのように宿題を確認しますか?
```

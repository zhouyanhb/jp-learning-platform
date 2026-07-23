# Japanese Sentence Boundary Resolution

The local Japanese sentence boundary resolver runs before subtitle building.
It uses aligned word timing, terminal punctuation, sentence-final expressions,
speaker metadata, and Japanese dependent continuations to preserve grammatical
sentence units.

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

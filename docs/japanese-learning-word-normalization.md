# Japanese Learning Word Normalization

The optional `word-normalization` stage runs after homophone correction and
before sentence-boundary resolution. Enable it with
`--enable-word-normalization`; enabling the homophone resolver also enables
this stage because both use Sudachi.

It converts morphology into learning-oriented units without sentence-specific
replacement rules. A connective `て/で` stays with the preceding inflected
verb (`聞いて`, `話して`), a non-independent verb and its auxiliary stay
together (`います`), and `でも` is separated from its host (`メール / でも`,
`いつ / でも`). Sudachi SplitMode C keeps compound nouns such as `学生` and
`授業` together.

When a source alignment token is divided, its time range is distributed by
character position. When fragments are joined, their ranges are joined and the
minimum available confidence is retained. Speaker metadata is preserved.

Install only this optional capability with:

```console
python3.12 -m pip install -e '.[word-normalization]'
```

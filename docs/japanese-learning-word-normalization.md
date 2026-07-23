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

Inflectional auxiliaries remain attached to their verb or adjective, including
negative, polite, past, passive, causative, and volitional forms. Examples
include `聞こえない`, `行きました`, and `高くない`. A noun marked by
Sudachi as sahen-capable joins an inflection whose dictionary form is `する`,
so `散歩しましょう` is one unit. A following auxiliary verb still starts a
new learning unit: `話して / います`, `確認して / ください`.

These decisions use morphological metadata rather than vocabulary or sentence
replacement tables. A standalone predicate remains separate, for example
`問題 / が / ない`, and a copula after an ordinary noun remains separate,
for example `便利 / です`.

When a source alignment token is divided, its time range is distributed by
character position. When fragments are joined, their ranges are joined and the
minimum available confidence is retained. Speaker metadata is preserved.

Install only this optional capability with:

```console
python3.12 -m pip install -e '.[word-normalization]'
```

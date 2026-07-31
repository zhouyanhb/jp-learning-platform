# Japanese Learning Word Normalization

The optional `word-normalization` stage runs after homophone correction and
sentence-boundary resolution. Enable it with
`--enable-word-normalization`; enabling the homophone resolver also enables
this stage because both use Sudachi.

It converts morphology into learning-oriented units without sentence-specific
replacement rules. A connective `て/で` stays with the preceding inflected
verb (`聞いて`, `話して`), a non-independent verb and its auxiliary stay
together (`います`), and `でも` is separated from its host (`メール / でも`,
`いつ / でも`). Sudachi SplitMode C keeps compound nouns such as `学生` and
`授業` together.

Some compounds are split only because they appear after another noun. For
example, Sudachi may analyze `回答用紙` as `回答 / 用 / 紙` even though it
recognizes standalone `用紙` as one noun. The normalizer therefore reanalyzes
contiguous noun and nominal-suffix windows of two or three morphemes, longest
first. It merges a window only when standalone Sudachi analysis returns one
complete noun with the same surface. This restores `回答 / 用紙` without a
word list or sentence-specific replacement and leaves combinations that remain
multi-morpheme unchanged.

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

The same structural pass normalizes units that Sudachi intentionally leaves
open-ended: contiguous ASCII letter-number identifiers (`N2`), a number plus a
counter-capable noun (`2番`), and adjacent katakana nouns
(`ポイントカード`). These rules use Unicode character classes and Sudachi
part-of-speech features rather than identifier, counter, or compound word
lists. Copular auxiliaries attach only to morphologically compatible
predicates, preventing a new `では` clause from being absorbed into a preceding
verb such as `休みましょう / で / は`.

The stage writes separate `learning_words` and never replaces the sentence's
aligned `words`. A learning word records its sentence character span and the
contiguous aligned-word indexes that support it. When a source alignment token
is divided, only the learning-word time is estimated by character position and
marked as estimated. When complete aligned tokens are joined, their time ranges
are joined without changing the source tokens.

Install only this optional capability with:

```console
python3.12 -m pip install -e '.[word-normalization]'
```

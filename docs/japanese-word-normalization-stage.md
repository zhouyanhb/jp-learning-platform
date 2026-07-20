# Japanese Word Normalization Stage

Japanese word normalization runs after the optional Qwen repair boundary and
before sentence-boundary resolution.

Whisper and WhisperX expose a field named `word`, but for Japanese that value
can be a character or token piece rather than a linguistic word. For example,
`天気` may arrive as `天` and `気`. If optional Qwen repair is enabled, it may
also change the current transcript text before this stage. The normalizer treats
the current segment text as final for word timing, tokenizes it into Japanese
words, and maps those words back onto the aligned timing units.

The default infrastructure adapter uses SudachiPy when it is installed. It then
applies a conservative learning-word merge pass on top of Sudachi morphemes:

- verb inflection chains such as `聞い / て` become `聞いて`;
- sahen verb chains such as `散歩 / し / ましょう` become `散歩しましょう`;
- `ください` stays separate, so `挙げてください` becomes `挙げて / ください`;
- stable inner noun pieces may merge, for example `問題 / 用 / 紙` becomes
  `問題 / 用紙`;
- generic number counters such as `2021 / 年` and `第 / 2 / 回` become
  `2021年` and `第2回`;
- compound particle units keep the left word separate, for example
  `いつ / で / も` becomes `いつ / でも`, and `これ / に / つい / て`
  becomes `これ / について`.

The merge pass intentionally does not use fixed JLPT or exam-term dictionaries.
A small heuristic tokenizer is used as a conservative fallback so the pipeline
can still run in minimal environments.

The stage writes `05_word_normalization.json` in progress artifact output.

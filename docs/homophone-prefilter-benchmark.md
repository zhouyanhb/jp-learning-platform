# Homophone Candidate Prefilter Benchmark

The homophone resolver keeps Sudachi sentence analysis exhaustive while
limiting contextual masked-language-model candidate scoring to at most three
suspicious targets per sentence.

## Prefilter signals

Targets must first have at least one same-reading lexical candidate. Remaining
targets are ranked by:

1. batched contextual probability of the original token;
2. aligned ASR word confidence;
3. normalized tokenizer vocabulary rank as a stable frequency proxy;
4. number of same-reading lexical candidates.

The contextual probability for every eligible target in one sentence is
computed in one model batch. Only the highest-risk targets proceed to full
candidate generation and replacement scoring. No source-to-replacement word
mapping is embedded in the resolver.

## Conservative acceptance

Target prefiltering decides which suspicious words are worth scoring; it does
not authorize a replacement. Automatic replacement requires all of these
independent signals:

- aligned ASR confidence is at most `0.9`;
- candidate probability is at least `0.0001`;
- candidate probability is at least `20` times the original probability;
- source and candidate retain the same kanji, hiragana, and katakana profile;
- reading and compatible part of speech still match.

Missing ASR confidence fails closed. Rejections remain in the stage artifact
with reasons such as `asr_confidence_too_high`, `candidate_score_too_low`, and
`candidate_score_ratio_too_low`.

An artifact audit of run `20260723_185826_324291` reduced automatic changes
from 76 to 2 without another model inference. It retains `懲戒 → 聴解` where
ASR confidence is low and `買える → 変える`; observed corruptions including
`寄付 → 記譜`, `学生 → 学政`, `もらえる → 貰る`, and `なかなか → 中中`
are rejected.

## Document consistency propagation

Resolution runs in two passes within one document. The first pass uses every
strict acceptance gate above. If those accepted decisions establish exactly
one replacement for an original surface, the second pass may propagate that
mapping to occurrences rejected only because ASR confidence was high.

Propagation also covers occurrences rejected only by the strict `20x` score
ratio. In that case the locally scored candidate must still be above the
minimum absolute score and strictly better than the local original. This lets
the confirmed `懲戒 → 聴解` mapping repair the final announcement, whose local
candidate is about `2.35x` the original, without admitting candidates that are
locally worse.

A propagated occurrence must already contain the same candidate in its local
scored candidates, and that candidate must still exceed the configured minimum
score and beat the local original score. Conflicting strict mappings disable
propagation for that surface. Exact target spans are recorded so all accepted
changes are reapplied from original sentence offsets without cascading edits.

Replaying decisions from run `20260724_070959_640324` with this policy uses the
strict title correction to confirm `懲戒 → 聴解`; the same unambiguous mapping
then repairs both announcements while unrelated rejected candidates remain
unchanged.

## Reproducible benchmark

Date: 2026-07-23

Host configuration:

- Python 3.12
- CPU execution
- `tohoku-nlp/bert-base-japanese-v3`
- input: `input/2021_12_start.mp3`
- homophone resolver enabled
- model files cached locally before both measured homophone stages

Command:

```bash
python3.12 -m jp_learning_platform transcribe \
  input/2021_12_start.mp3 \
  --output-dir <isolated-output-directory> \
  --export-srt \
  --enable-homophone-resolver
```

Measured stage times:

| Stage | Before | After |
| --- | ---: | ---: |
| audio-loader | 0.0003 s | 0.0003 s |
| whisper | 114.3064 s | 109.2281 s |
| whisperx-alignment | 0.00005 s | 0.00005 s |
| qwen-repair | 0.00004 s | 0.00006 s |
| homophone-resolution | 16.8859 s | 9.2291 s |
| sentence-boundary-resolution | 0.0036 s | 0.0036 s |
| subtitle-builder | 0.00008 s | 0.00009 s |
| subtitle-merger | 0.00012 s | 0.00012 s |
| readability-optimizer | 0.0010 s | 0.0011 s |
| subtitle-validator | 0.00007 s | 0.00006 s |
| subtitle-writer | 0.0050 s | 0.0038 s |
| pipeline total | 131.33 s | 118.60 s |

The homophone stage decreased by 45.35%, while end-to-end runtime decreased by
9.69%. Contextual homophone decisions decreased from 189 to 81. Both runs
accepted six corrections, including the two observed same-reading listening
exam corrections.

End-to-end results include normal run-to-run Whisper variation. The homophone
stage comparison is the primary measurement for this optimization.

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

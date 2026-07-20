# Local Audio Transcribe CLI

The local audio transcribe CLI generates structured intensive-listening JSON
from a single audio file or a folder of audio files. SRT output is an optional
export.

## Usage

```bash
python -m jp_learning_platform transcribe audio.mp3
python -m jp_learning_platform transcribe ./audios
```

Generated `.json` files are written to `output/` by default. A custom output
directory can be supplied when needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --output-dir subtitles
```

Export SRT beside the structured JSON when a subtitle file is needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --export-srt
```

The command reports per-file stage progress while it runs. Progress is written
to stderr so stdout can continue to list the final generated JSON paths.

ASR model settings can be supplied from the CLI. Kotoba Whisper v2.1 is the
default ASR backend:

```bash
python -m jp_learning_platform transcribe audio.mp3
python -m jp_learning_platform transcribe audio.mp3 --asr-model kotoba-whisper-v2.0
python -m jp_learning_platform transcribe audio.mp3 --asr-backend reazon-speech
python -m jp_learning_platform transcribe audio.mp3 --asr-backend faster-whisper --model-size small --device cpu --compute-type int8
```

Defaults are:

- `--asr-backend kotoba-whisper`
- `--asr-model kotoba-tech/kotoba-whisper-v2.1`
- `--asr-model reazon-research/reazonspeech-nemo-v2` for
  `--asr-backend reazon-speech`
- `--device cpu`
- `--asr-chunk-length-seconds 15.0`
- `--asr-chunk-overlap-seconds 2.0` for `--asr-backend reazon-speech`
- `--asr-batch-size 16`
- `--model-size large-v3` for `--asr-backend faster-whisper`
- `--compute-type int8` for `--asr-backend faster-whisper`

The default Kotoba adapter uses the standard Transformers ASR pipeline and does
not load Kotoba's remote post-processing pipeline, so the main runtime
requirements do not include `stable-ts`, `punctuators`, or `openai-whisper`.
When the v2.1 weights are used through this standard path, the adapter reuses
the v2.0 processor files because the v2.1 repository is primarily the model
weight and remote-pipeline wrapper.

The optional ReazonSpeech backend uses the official `reazonspeech.nemo.asr`
adapter when the default model id is selected. Long audio is split into
overlapping chunks before being sent to ReazonSpeech, then chunk-local
timestamps are shifted back onto the original audio timeline. Install it
separately before running `--asr-backend reazon-speech`:

```bash
python -m pip install 'git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/nemo-asr'
```

Lower-level transcription defaults such as beam size, word timestamps, VAD,
and hallucination silence filtering are centralized in
`docs/pipeline-configuration.md`.

## Quality Stages

The CLI now runs the full subtitle quality workflow:

```text
AudioLoader
-> WhisperStage
-> WhisperXAlignmentStage
-> SentenceBoundaryDetectionStage
-> QwenRepairStage (disabled by default)
-> JapaneseWordNormalizationStage
-> HomophoneResolutionStage (optional)
-> SentenceBoundaryResolverStage
-> SubtitleBuilderStage
-> SubtitleMergerStage
-> ReadabilityOptimizerStage
-> SubtitleValidatorStage
-> SubtitleWriterStage
```

WhisperX and Qwen are external-model integrations, but only WhisperX runs by
default. The CLI runs real WhisperX alignment after ASR transcription. Skip
forced alignment only when the original Whisper timings should be kept:

```bash
python -m jp_learning_platform transcribe audio.mp3 --disable-whisperx
```

Install the alignment dependency first:

```bash
python -m pip install -e ".[align]"
```

After alignment, the CLI uses torch/torchaudio waveform energy around aligned
word gaps to record acoustic sentence-boundary candidates. Candidates are kept
by time so they can still be applied if optional Qwen repair changes word
boundaries. The final sentence split is resolved later, after the current text
and punctuation are available. Skip both boundary stages only when the original
segment-level sentence grouping should be kept:

```bash
python -m jp_learning_platform transcribe audio.mp3 --disable-sentence-boundaries
```

Install the VAD dependency first:

```bash
python -m pip install -e ".[vad]"
```

After the optional Qwen repair workflow boundary, Japanese word timing
normalization maps the current transcript text back onto aligned timing units.
SudachiPy is used when available, so final text such as `天気` becomes the final
word even if the pre-normalization ASR pieces were `天` and `気`:

```bash
python -m pip install -e ".[japanese]"
```

Skip this only when the raw ASR/aligner word pieces should be preserved in the
final JSON:

```bash
python -m jp_learning_platform transcribe audio.mp3 --disable-word-normalization
```

Homophone semantic resolution is available after word normalization, but is
disabled by default because it loads a Japanese masked language model. It is
not a free text repair step: candidates must have the same Sudachi reading and
compatible part of speech, and the language model only decides whether a
same-reading candidate is more likely in the current sentence context:

```bash
python -m jp_learning_platform transcribe audio.mp3 --enable-homophone-resolver
```

Tune the masked language model and candidate search breadth when needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --enable-homophone-resolver --homophone-model-id tohoku-nlp/bert-base-japanese-v3 --homophone-top-k 120 --homophone-score-margin 0.05
```

Install the optional tokenizer/runtime support first:

```bash
python -m pip install -e ".[homophone]"
```

Enable pyannote.audio speaker diarization when speaker identifiers should be
assigned automatically:

```bash
python -m jp_learning_platform transcribe audio.mp3 --enable-diarization
```

Install the optional diarization dependency and provide a Hugging Face token
accepted for the pyannote speaker diarization model:

```bash
python -m pip install -e ".[diarization]"
HF_TOKEN=hf_... python -m jp_learning_platform transcribe audio.mp3 --enable-diarization
```

The token can also be passed with `--hf-token`. Diarization runs inside the
existing WhisperX alignment workflow boundary: the configured aligner produces
timed segments first, then pyannote speaker turns are matched to words by time
overlap. When a sentence contains multiple speakers, it is split into
speaker-specific segment runs before subtitle building.

Local Qwen repair is disabled by default. The default command does not load
llama.cpp, call the local Qwen model, or apply Qwen text edits:

```bash
python -m jp_learning_platform transcribe audio.mp3
```

Local Qwen repair uses a conservative safety policy. If a model output appears
to add or remove spoken content, the repairer keeps the original aligned text
so subtitle timing and word timing remain authoritative.

Enable repair or override the model path when needed:

```bash
python -m jp_learning_platform transcribe audio.mp3 --enable-qwen
python -m jp_learning_platform transcribe audio.mp3 --qwen-model-path models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

`--disable-qwen` is still accepted as an explicit no-op for scripts that want
to state the default.

Install the Qwen dependency first:

```bash
python -m pip install -e ".[qwen]"
```

## ASR Dependency

The command uses the Kotoba Whisper infrastructure adapter for speech
recognition by default. Install the optional ASR dependencies before running
Kotoba or faster-whisper transcription:

```bash
python -m pip install -e ".[asr]"
```

Install the ReazonSpeech backend separately:

```bash
python -m pip install 'git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/nemo-asr'
```

## Pipeline

The first-stage local CLI uses the existing workflow contracts:

```text
AudioLoader
-> WhisperStage
-> WhisperXAlignmentStage
-> SentenceBoundaryDetectionStage
-> QwenRepairStage (disabled by default)
-> JapaneseWordNormalizationStage
-> HomophoneResolutionStage (optional)
-> SentenceBoundaryResolverStage
-> WordSubtitleBuilder
-> ConservativeSubtitleMerger
-> LocalReadabilityOptimizer
-> DomainSubtitleValidator
-> SubtitleWriterStage
-> ListeningJsonWriter
```

The generated subtitles preserve word-derived timing through the domain
`Sentence` and `Word` objects before writing the final structured JSON. The
JSON output contains segment, sentence, word, and subtitle timing so downstream
intensive-listening views can query unfamiliar words without parsing SRT text.

When upstream alignment data includes speaker identifiers, the pipeline keeps
different speakers in separate subtitle cues and prevents cross-speaker merging.
With `--enable-diarization`, speaker identifiers can be produced from the audio
itself by pyannote.audio. Speaker identifiers remain structured metadata.
Optional SRT export does not display speaker labels.

## Progress and Stage Artifacts

Each processed audio file emits one-line progress events for every stage:

```text
[1/2] lesson.mp3 audio-loader started
[1/2] lesson.mp3 audio-loader done 0.01s -> output/.work/20260717_153012_123456/lesson/00_audio_load.json
[1/2] lesson.mp3 whisper started
[1/2] lesson.mp3 whisper done 12.48s -> output/.work/20260717_153012_123456/lesson/01_whisper.json
```

The final structured JSON remains at `output/<audio-name>.json`. When
`--export-srt` is supplied, the optional SRT export is written beside it as
`output/<audio-name>.srt`. Stage artifacts are saved under the output directory:

```text
output/.work/<run-name>/<audio-name>/manifest.json
output/.work/<run-name>/<audio-name>/00_audio_load.json
output/.work/<run-name>/<audio-name>/01_whisper.json
output/.work/<run-name>/<audio-name>/02_align.json
output/.work/<run-name>/<audio-name>/03_sentence_boundary_candidates.json
output/.work/<run-name>/<audio-name>/04_repair.json
output/.work/<run-name>/<audio-name>/05_word_normalization.json
output/.work/<run-name>/<audio-name>/06_sentence_boundary_resolution.json
output/.work/<run-name>/<audio-name>/07_build.json
output/.work/<run-name>/<audio-name>/08_merge.json
output/.work/<run-name>/<audio-name>/09_readability.json
output/.work/<run-name>/<audio-name>/10_validate.json
output/.work/<run-name>/<audio-name>/11_write.json
```

Artifacts contain the source path, output path, file index, stage status,
elapsed time, recorded timestamp, pipeline context, and stage data when the
stage exposes additional data. The manifest records the current stage and the
latest stage artifact path for that audio file.

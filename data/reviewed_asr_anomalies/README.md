# Reviewed ASR anomaly labels

These fixed, manually reviewed time ranges evaluate transcript-content anomalies
independently from language sentence boundaries.

Run the reviewed baseline against a new pipeline run:

```bash
python3.12 -m jp_learning_platform.reviewed_asr_anomaly_evaluation \
  data/reviewed_asr_anomalies/20260827/reviewed.json \
  output/evaluations/reviewed_asr_anomalies_latest.json \
  --artifact-root output/.work/RUN_ID
```

The report contains overall precision/recall, per-anomaly metrics, per-content
category metrics, and explicit false-positive/false-negative details. Samples
may include negative guards for content that is present and must not be reported
as omitted.

Local text substitutions remain in `data/local_asr_regressions`; they are
evaluated separately and do not authorize automatic transcript replacement.

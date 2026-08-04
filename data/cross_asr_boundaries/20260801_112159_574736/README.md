# Cross-ASR boundary review set

This directory contains review candidates built from the eight media files in
run `20260801_112159_574736`.

Review order:

1. Review every sample whose `predicted_label` is `merge`. These samples
   determine merge precision and expose false merges.
2. Review a representative sample of predicted `keep` boundaries from N2,
   drama, Vlog/Podcast, and news files to estimate merge recall.
3. Set `gold_label` to `merge`, `keep`, or `uncertain` and set
   `review_status` to `reviewed`.
4. Do not copy `predicted_label` into `gold_label` without listening to the
   audio or checking a trusted transcript.

Labels:

- `merge`: both sides belong to one language sentence.
- `keep`: the language sentence boundary must remain.
- `uncertain`: the reviewer cannot confidently decide.

The evaluator ignores every sample that is not marked `reviewed`.

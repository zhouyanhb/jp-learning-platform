# Local ASR regression datasets

These reviewed samples keep local ASR repair and homophone resolution separate
from language-sentence boundary rules.

Each sample records the reference, stage artifacts, target span, expected text,
and the layer where the current pipeline succeeds or fails. Unresolved samples
are baseline failures. Resolved samples are regression guards and must remain
correct.

Do not use reference text as a runtime replacement dictionary. It is evaluation
evidence only.

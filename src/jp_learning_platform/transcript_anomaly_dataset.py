"""Build reviewable datasets for transcript-content anomaly isolation."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from jp_learning_platform.workflow.transcript_anomaly_stage import (
    ISOLATED_CONTENT_ANOMALY_KINDS,
)


def build_transcript_anomaly_dataset(
    artifact_paths: tuple[Path, ...],
    *,
    context_radius: int = 0,
) -> dict[str, object]:
    if context_radius < 0:
        raise ValueError("context_radius must be non-negative.")
    documents: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    for document_index, artifact_path in enumerate(artifact_paths, start=1):
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        document_id = f"document-{document_index:03d}"
        document = ((artifact.get("context") or {}).get("document") or {})
        segments = {
            int(segment["position"]): segment
            for segment in document.get("segments") or ()
        }
        predictions: dict[tuple[int, int], set[str]] = {}
        for candidate in (artifact.get("data") or {}).get("candidates") or ():
            kind = str(candidate.get("kind") or "")
            if kind not in ISOLATED_CONTENT_ANOMALY_KINDS:
                continue
            for position in candidate.get("segment_positions") or ():
                segment = segments.get(int(position))
                if segment is None:
                    continue
                sentence_indexes = candidate.get("sentence_indexes") or range(
                    len(segment.get("sentences") or ())
                )
                for sentence_index in sentence_indexes:
                    key = (int(position), int(sentence_index))
                    predictions.setdefault(key, set()).add(kind)
        documents.append(
            {
                "id": document_id,
                "artifact": str(artifact_path),
                "source_path": str(document.get("source_path") or ""),
            }
        )
        sentence_keys = tuple(
            (position, sentence_index)
            for position, segment in sorted(segments.items())
            for sentence_index, _sentence in enumerate(
                segment.get("sentences") or ()
            )
        )
        key_indexes = {key: index for index, key in enumerate(sentence_keys)}
        selected_keys = set(predictions)
        for key in predictions:
            center = key_indexes[key]
            selected_keys.update(
                sentence_keys[
                    max(0, center - context_radius) : center + context_radius + 1
                ]
            )
        for position, sentence_index in sorted(selected_keys):
            segment = segments[position]
            sentence = (segment.get("sentences") or ())[sentence_index]
            predicted_kinds = sorted(predictions.get((position, sentence_index), ()))
            samples.append(
                {
                    "id": f"anomaly-sample-{len(samples) + 1:05d}",
                    "document_id": document_id,
                    "segment_position": position,
                    "sentence_index": sentence_index,
                    "time_range": sentence["time_range"],
                    "text": str(sentence.get("text") or ""),
                    "sample_kind": (
                        "prediction" if predicted_kinds else "context_negative_candidate"
                    ),
                    "predicted_anomaly_kinds": predicted_kinds,
                    "gold_anomaly_kinds": [],
                    "review_status": "needs_review",
                    "review_note": "",
                }
            )
    return {
        "schema_version": 1,
        "language": "ja",
        "annotation_status": "needs_review",
        "anomaly_kinds": sorted(ISOLATED_CONTENT_ANOMALY_KINDS),
        "documents": documents,
        "samples": samples,
    }


def write_transcript_anomaly_dataset(
    artifact_paths: tuple[Path, ...],
    output_path: Path,
    *,
    context_radius: int = 0,
) -> dict[str, object]:
    dataset = build_transcript_anomaly_dataset(
        artifact_paths,
        context_radius=context_radius,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(dataset, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return dataset


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-radius", type=int, default=0)
    args = parser.parse_args()
    dataset = write_transcript_anomaly_dataset(
        tuple(args.artifacts),
        args.output,
        context_radius=args.context_radius,
    )
    print(f"documents={len(dataset['documents'])}")
    print(f"samples={len(dataset['samples'])}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

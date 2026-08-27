"""Evaluate shadow homophone candidate recall without changing transcripts."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import unicodedata


def evaluate_homophone_shadow_candidates(
    dataset_path: Path,
    artifact_root: Path,
) -> dict[str, object]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    artifacts = _artifacts_by_source(Path(artifact_root))
    results: list[dict[str, object]] = []
    for sample in dataset.get("samples") or ():
        artifact_directory = artifacts.get(_source_key(str(sample["source_path"])))
        results.append(_evaluate_sample(sample, artifact_directory))

    counts = Counter(str(item["status"]) for item in results)
    evaluated = len(results) - counts["missing_artifact"]
    generated = counts["generated"]
    positive_evaluated = sum(
        item["label"] == "positive" and item["status"] != "missing_artifact"
        for item in results
    )
    negative_evaluated = sum(
        item["label"] == "negative" and item["status"] != "missing_artifact"
        for item in results
    )
    top_1_correct = sum(item.get("expected_rank") == 1 for item in results)
    negative_passed = counts["negative_passed"]
    top_1_decision_correct = top_1_correct + negative_passed
    acceptance_true_positive = sum(
        item["label"] == "positive"
        and _normalize(str(item.get("accepted_candidate") or ""))
        == _normalize(str(item.get("expected_candidate") or ""))
        for item in results
    )
    acceptance_decisions = sum(
        item.get("relative_acceptance_status") == "accepted" for item in results
    )
    acceptance_false_positive = acceptance_decisions - acceptance_true_positive
    acceptance_evaluated = sum(
        item.get("relative_acceptance_status") in {"accepted", "rejected"}
        for item in results
    )
    acceptance_positive_evaluated = sum(
        item["label"] == "positive"
        and item.get("relative_acceptance_status") in {"accepted", "rejected"}
        for item in results
    )
    acceptance_negative_evaluated = sum(
        item["label"] == "negative"
        and item.get("relative_acceptance_status") in {"accepted", "rejected"}
        for item in results
    )
    margins = [
        float(item["top_score_margin"])
        for item in results
        if item.get("top_score_margin") is not None
    ]
    ratios = [
        float(item["top_score_ratio_vs_original"])
        for item in results
        if item.get("top_score_ratio_vs_original") is not None
    ]
    return {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "artifact_root": str(artifact_root),
        "metrics": {
            "total": len(results),
            "evaluated": evaluated,
            "positive_evaluated": positive_evaluated,
            "negative_evaluated": negative_evaluated,
            "generated": generated,
            "candidate_missing": counts["candidate_missing"],
            "target_missing": counts["target_missing"],
            "missing_artifact": counts["missing_artifact"],
            "target_recall": _ratio(generated, positive_evaluated),
            "top_1_correct": top_1_correct,
            "top_1_accuracy": _ratio(top_1_correct, positive_evaluated),
            "top_1_decision_correct": top_1_decision_correct,
            "top_1_decision_accuracy": _ratio(
                top_1_decision_correct,
                positive_evaluated + negative_evaluated,
            ),
            "acceptance_true_positive": acceptance_true_positive,
            "acceptance_false_positive": acceptance_false_positive,
            "acceptance_evaluated": acceptance_evaluated,
            "acceptance_positive_evaluated": acceptance_positive_evaluated,
            "acceptance_negative_evaluated": acceptance_negative_evaluated,
            "acceptance_not_evaluated": sum(
                item.get("relative_acceptance_status")
                in {"audit_only", "not_evaluated"}
                for item in results
            ),
            "acceptance_precision": _ratio(
                acceptance_true_positive,
                acceptance_decisions,
            ),
            "acceptance_recall": _ratio(
                acceptance_true_positive,
                acceptance_positive_evaluated,
            ),
            "negative_passed": negative_passed,
            "false_positive": counts["false_positive"],
            "false_positive_rate": _ratio(
                counts["false_positive"],
                negative_evaluated,
            ),
            "score_missing": counts["score_missing"],
            "mean_top_1_margin": _mean(margins),
            "mean_top_score_ratio_vs_original": _mean(ratios),
            "candidate_count": sum(int(item["candidate_count"]) for item in results),
        },
        "metrics_by_strategy": _metrics_by_strategy(results),
        "results": results,
        "false_negatives": [
            item
            for item in results
            if item["label"] == "positive"
            and item["status"] in {"candidate_missing", "target_missing"}
        ],
        "misranked": [
            item
            for item in results
            if item["status"] == "generated" and item.get("expected_rank") != 1
        ],
        "false_positives": [
            item for item in results if item["status"] == "false_positive"
        ],
    }


def write_homophone_shadow_evaluation(
    dataset_path: Path,
    artifact_root: Path,
    output_path: Path,
) -> dict[str, object]:
    report = evaluate_homophone_shadow_candidates(dataset_path, artifact_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def _evaluate_sample(
    sample: dict[str, object],
    artifact_directory: Path | None,
) -> dict[str, object]:
    base = {
        "id": sample["id"],
        "label": sample.get("label", "positive"),
        "strategy": sample["strategy"],
        "surface": sample["surface"],
        "expected_candidate": sample.get("expected_candidate"),
    }
    if artifact_directory is None:
        return {**base, "status": "missing_artifact", "candidate_count": 0}
    artifact_path = artifact_directory / "04_homophone_resolution.json"
    if not artifact_path.is_file():
        return {**base, "status": "missing_artifact", "candidate_count": 0}

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    shadows = (artifact.get("data") or {}).get("shadow_candidates") or ()
    positions = _positions_in_time_range(artifact, sample["time_range"])
    matching = [
        item
        for item in shadows
        if item.get("segment_position") in positions
        and item.get("strategy") == sample["strategy"]
        and _normalize(str(item.get("surface") or ""))
        == _normalize(str(sample["surface"]))
    ]
    candidates = tuple(
        dict.fromkeys(
            str(candidate)
            for item in matching
            for candidate in item.get("candidates") or ()
        )
    )
    ranked_candidates = _combined_ranking(matching)
    expected_normalized = _normalize(str(sample.get("expected_candidate") or ""))
    expected_rank = next(
        (
            int(candidate["rank"])
            for candidate in ranked_candidates
            if _normalize(str(candidate["text"])) == expected_normalized
        ),
        None,
    )
    top_candidate = (
        str(ranked_candidates[0]["text"]) if ranked_candidates else None
    )
    top_score_margin = _ranking_margin(ranked_candidates)
    original_score = next(
        (
            float(item["original_score"])
            for item in matching
            if item.get("original_score") is not None
        ),
        None,
    )
    top_score = (
        float(ranked_candidates[0]["score"])
        if ranked_candidates and ranked_candidates[0].get("score") is not None
        else None
    )
    acceptance_statuses = {
        str(item.get("relative_acceptance_status"))
        for item in matching
        if item.get("relative_acceptance_status")
        in {"accepted", "rejected", "audit_only", "not_evaluated"}
    }
    relative_acceptance_status = (
        "accepted"
        if "accepted" in acceptance_statuses
        else "rejected"
        if "rejected" in acceptance_statuses
        else "audit_only"
        if "audit_only" in acceptance_statuses
        else "not_evaluated"
        if "not_evaluated" in acceptance_statuses
        else "legacy_score_fallback"
    )
    relative_acceptance_reason = next(
        (
            str(item.get("relative_acceptance_reason") or "")
            for item in matching
            if item.get("relative_acceptance_status")
            == relative_acceptance_status
        ),
        "",
    )
    accepted_candidate = next(
        (
            str(item.get("accepted_candidate"))
            for item in matching
            if item.get("accepted_candidate")
        ),
        None,
    )
    label = str(sample.get("label") or "positive")
    if not matching:
        status = "target_missing"
    elif label == "negative" and not candidates:
        status = "negative_passed"
    elif label == "negative" and relative_acceptance_status == "accepted":
        status = "false_positive"
    elif label == "negative" and relative_acceptance_status == "rejected":
        status = "negative_passed"
    elif label == "negative" and relative_acceptance_status in {
        "audit_only",
        "not_evaluated",
    }:
        status = "acceptance_not_evaluated"
    elif label == "negative" and (top_score is None or original_score is None):
        status = "score_missing"
    elif label == "negative" and top_score <= original_score:
        status = "negative_passed"
    elif label == "negative":
        status = "false_positive"
    elif expected_normalized in {
        _normalize(candidate) for candidate in candidates
    }:
        status = "generated"
    else:
        status = "candidate_missing"
    return {
        **base,
        "status": status,
        "artifact_directory": str(artifact_directory),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ranked_candidates": ranked_candidates,
        "expected_rank": expected_rank,
        "top_candidate": top_candidate,
        "top_score_margin": top_score_margin,
        "original_score": original_score,
        "top_score_ratio_vs_original": _score_ratio(top_score, original_score),
        "relative_acceptance_status": relative_acceptance_status,
        "relative_acceptance_reason": relative_acceptance_reason,
        "accepted_candidate": accepted_candidate,
        "score_method": next(
            (str(item.get("score_method")) for item in matching if item.get("score_method")),
            "legacy_independent_score",
        ),
        "original_token_count": next(
            (item.get("original_token_count") for item in matching if item.get("original_token_count") is not None),
            None,
        ),
        "top_candidate_token_count": next(
            (item.get("top_candidate_token_count") for item in matching if item.get("top_candidate_token_count") is not None),
            None,
        ),
    }


def _combined_ranking(
    matching: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    best_scores: dict[str, float | None] = {}
    for item in matching:
        scored = item.get("ranked_candidates") or item.get("candidate_scores") or ()
        for candidate in scored:
            text = str(candidate.get("text") or "")
            if not text:
                continue
            score = candidate.get("score")
            numeric_score = float(score) if score is not None else None
            previous = best_scores.get(text)
            if previous is None or (
                numeric_score is not None and numeric_score > previous
            ):
                best_scores[text] = numeric_score
    ordered = sorted(
        best_scores.items(),
        key=lambda item: (item[1] is None, -(item[1] or 0.0), item[0]),
    )
    return tuple(
        {"text": text, "score": score, "rank": rank}
        for rank, (text, score) in enumerate(ordered, start=1)
    )


def _ranking_margin(ranked: tuple[dict[str, object], ...]) -> float | None:
    if len(ranked) < 2:
        return None
    first = ranked[0].get("score")
    second = ranked[1].get("score")
    if first is None or second is None:
        return None
    return float(first) - float(second)


def _score_ratio(score: float | None, original: float | None) -> float | None:
    if score is None or original is None:
        return None
    return score / max(original, 1e-12)


def _positions_in_time_range(
    artifact: dict[str, object],
    time_range: dict[str, object],
) -> set[int]:
    document = ((artifact.get("context") or {}).get("document") or {})
    start = float(time_range["start_seconds"])
    end = float(time_range["end_seconds"])
    positions: set[int] = set()
    for segment in document.get("segments") or ():
        segment_range = segment.get("time_range") or {}
        segment_start = float(segment_range.get("start_seconds", 0.0))
        segment_end = float(segment_range.get("end_seconds", 0.0))
        if segment_end >= start - 0.5 and segment_start <= end + 0.5:
            positions.add(int(segment["position"]))
    return positions


def _artifacts_by_source(root: Path) -> dict[str, Path]:
    artifacts: dict[str, tuple[str, Path]] = {}
    if not root.is_dir():
        return {}
    for manifest_path in root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = str(manifest.get("source_path") or "")
        status = manifest.get("status")
        if not source_path or (status is not None and status != "succeeded"):
            continue
        key = _source_key(source_path)
        updated_at = str(manifest.get("updated_at") or "")
        current = artifacts.get(key)
        if current is None or updated_at > current[0]:
            artifacts[key] = (updated_at, manifest_path.parent)
    return {key: value[1] for key, value in artifacts.items()}


def _source_key(value: str) -> str:
    stem = Path(unicodedata.normalize("NFKC", value)).stem
    return "".join(character for character in stem if character.isalnum()).lower()


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _metrics_by_strategy(
    results: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in results:
        grouped.setdefault(str(item["strategy"]), []).append(item)
    return {
        strategy: {
            "total": len(items),
            "positive": sum(item["label"] == "positive" for item in items),
            "negative": sum(item["label"] == "negative" for item in items),
            "generated": sum(item["status"] == "generated" for item in items),
            "target_recall": _ratio(
                sum(item["status"] == "generated" for item in items),
                sum(
                    item["label"] == "positive"
                    and item["status"] != "missing_artifact"
                    for item in items
                ),
            ),
            "top_1_correct": sum(
                item.get("expected_rank") == 1 for item in items
            ),
            "top_1_accuracy": _ratio(
                sum(item.get("expected_rank") == 1 for item in items),
                sum(
                    item["label"] == "positive"
                    and item["status"] != "missing_artifact"
                    for item in items
                ),
            ),
            "top_1_decision_correct": sum(
                item.get("expected_rank") == 1
                or item["status"] == "negative_passed"
                for item in items
            ),
            "top_1_decision_accuracy": _ratio(
                sum(
                    item.get("expected_rank") == 1
                    or item["status"] == "negative_passed"
                    for item in items
                ),
                sum(item["status"] != "missing_artifact" for item in items),
            ),
            "acceptance_true_positive": sum(
                item["label"] == "positive"
                and _normalize(str(item.get("accepted_candidate") or ""))
                == _normalize(str(item.get("expected_candidate") or ""))
                for item in items
            ),
            "acceptance_false_positive": sum(
                item.get("relative_acceptance_status") == "accepted"
                for item in items
            ) - sum(
                item["label"] == "positive"
                and _normalize(str(item.get("accepted_candidate") or ""))
                == _normalize(str(item.get("expected_candidate") or ""))
                for item in items
            ),
            "acceptance_evaluated": sum(
                item.get("relative_acceptance_status") in {"accepted", "rejected"}
                for item in items
            ),
            "acceptance_not_evaluated": sum(
                item.get("relative_acceptance_status")
                in {"audit_only", "not_evaluated"}
                for item in items
            ),
            "acceptance_precision": _ratio(
                sum(
                    item["label"] == "positive"
                    and _normalize(str(item.get("accepted_candidate") or ""))
                    == _normalize(str(item.get("expected_candidate") or ""))
                    for item in items
                ),
                sum(
                    item.get("relative_acceptance_status") == "accepted"
                    for item in items
                ),
            ),
            "acceptance_recall": _ratio(
                sum(
                    item["label"] == "positive"
                    and _normalize(str(item.get("accepted_candidate") or ""))
                    == _normalize(str(item.get("expected_candidate") or ""))
                    for item in items
                ),
                sum(
                    item["label"] == "positive"
                    and item.get("relative_acceptance_status")
                    in {"accepted", "rejected"}
                    for item in items
                ),
            ),
            "negative_passed": sum(
                item["status"] == "negative_passed" for item in items
            ),
            "false_positive": sum(
                item["status"] == "false_positive" for item in items
            ),
        }
        for strategy, items in sorted(grouped.items())
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = write_homophone_shadow_evaluation(
        args.dataset,
        args.artifact_root,
        args.output,
    )
    metrics = report["metrics"]
    print(
        f"generated={metrics['generated']} evaluated={metrics['evaluated']} "
        f"target_recall={metrics['target_recall']} "
        f"top_1_accuracy={metrics['top_1_accuracy']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.question_punctuation_evaluation import (
    evaluate_question_punctuation,
)


def test_evaluates_question_mark_precision_recall_and_error_details(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:00,000 --> 00:00:01,000
元気ですか？

2
00:00:02,000 --> 00:00:03,000
今日は晴れです。

3
00:00:04,000 --> 00:00:05,000
何をしますか？

4
00:00:04,050 --> 00:00:05,050
何をしますか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "context": {
                    "document": {
                        "segments": [
                            {
                                "position": 0,
                                "sentences": [
                                    {
                                        "text": "元気ですか?",
                                        "is_question": True,
                                        "time_range": {
                                            "start_seconds": 0.0,
                                            "end_seconds": 1.0,
                                        },
                                    },
                                    {
                                        "text": "今日は晴れです?",
                                        "is_question": True,
                                        "time_range": {
                                            "start_seconds": 2.0,
                                            "end_seconds": 3.0,
                                        },
                                    },
                                    {
                                        "text": "何をしますか",
                                        "is_question": False,
                                        "time_range": {
                                            "start_seconds": 4.0,
                                            "end_seconds": 5.0,
                                        },
                                    },
                                ],
                            }
                        ]
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(reference, artifact)

    assert report["coverage"] == {
        "reference_questions": 2,
        "evaluated_reference_questions": 2,
        "excluded_reference_questions": 0,
        "language_sentences": 3,
        "predicted_questions": 2,
        "evaluated_predicted_questions": 2,
        "ignored_predictions": 0,
    }
    assert report["metrics"] == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "precision_target_met": False,
    }
    assert report["errors"]["false_positive"][0]["text"] == "今日は晴れです?"
    missed = report["errors"]["false_negative"][0]
    assert missed["text"] == "何をしますか？"
    assert missed["overlapping_language_sentences"][0]["text"] == "何をしますか"


def test_evaluates_candidate_artifact_without_requiring_visible_question_mark(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:02,000
何を？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "05b_question_punctuation_candidates.json"
    artifact.write_text(
        json.dumps(
            {
                "context": {
                    "document": {
                        "segments": [
                            {
                                "position": 4,
                                "sentences": [
                                    {
                                        "text": "何を",
                                        "time_range": {
                                            "start_seconds": 1.0,
                                            "end_seconds": 2.0,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                },
                "data": {
                    "candidates": [
                        {
                            "segment_position": 4,
                            "sentence_index": 0,
                            "text": "何を",
                            "time_range": {
                                "start_seconds": 1.0,
                                "end_seconds": 2.0,
                            },
                            "confidence": 0.95,
                            "evidence": [
                                "short_pronominal_case_phrase",
                                "following_independent_response",
                            ],
                        }
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(reference, artifact)

    assert report["prediction_source"] == "question_punctuation_candidates"
    assert report["metrics"]["precision"] == 1.0
    assert report["metrics"]["recall"] == 1.0
    assert report["metrics"]["precision_target_met"]
    assert report["matches"][0]["prediction"]["confidence"] == 0.95


def test_does_not_match_sentence_terminal_to_embedded_reference_question(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:05,000
「トイレ借りていい？」って聞くのに似ています。
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "トイレ借りていいって聞くのに似ていますか",
                            "time_range": {
                                "start_seconds": 1.0,
                                "end_seconds": 5.0,
                            },
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(reference, artifact)

    assert report["metrics"]["true_positive"] == 0
    assert report["metrics"]["false_positive"] == 1
    assert report["metrics"]["false_negative"] == 1


def test_matches_short_question_at_end_of_combined_reference_cue(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:28:46,026 --> 00:28:50,464
今世をやり直すかですよね。ん？どういうことですか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "どういうことですか",
                            "time_range": {
                                "start_seconds": 1730.447,
                                "end_seconds": 1731.060,
                            },
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(
        reference,
        artifact,
        reference_time_offset_seconds=0.0,
    )

    assert report["metrics"]["true_positive"] == 1
    assert report["metrics"]["false_positive"] == 0


def test_evaluates_each_question_inside_one_reference_cue(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:05,000
今日はどこへ行きますか？電車で行きますか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "今日はどこへ行きますか",
                            "time_range": {
                                "start_seconds": 1.0,
                                "end_seconds": 3.0,
                            },
                        },
                        {
                            "segment_position": 0,
                            "sentence_index": 1,
                            "text": "電車で行きますか",
                            "time_range": {
                                "start_seconds": 3.0,
                                "end_seconds": 5.0,
                            },
                        },
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(
        reference,
        artifact,
        reference_time_offset_seconds=0.0,
    )

    assert report["coverage"]["reference_questions"] == 2
    assert report["metrics"]["true_positive"] == 2
    assert report["metrics"]["false_positive"] == 0
    assert report["metrics"]["false_negative"] == 0
    assert {
        match["reference"]["text"] for match in report["matches"]
    } == {"今日はどこへ行きますか？", "電車で行きますか？"}


def test_reports_unmatched_question_inside_multi_question_reference_cue(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:05,000
最初の質問ですか？次の質問ですか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "次の質問ですか",
                            "time_range": {
                                "start_seconds": 3.0,
                                "end_seconds": 5.0,
                            },
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(
        reference,
        artifact,
        reference_time_offset_seconds=0.0,
    )

    assert report["metrics"]["true_positive"] == 1
    assert report["metrics"]["false_negative"] == 1
    assert report["errors"]["false_negative"][0]["text"] == "最初の質問ですか？"


def test_terminal_endpoint_exception_requires_exact_reference_suffix(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:05,000
前の発言です。どういうことですか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "次の話は何ですか",
                            "time_range": {
                                "start_seconds": 5.05,
                                "end_seconds": 6.0,
                            },
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(
        reference,
        artifact,
        reference_time_offset_seconds=0.0,
    )

    assert report["metrics"]["true_positive"] == 0
    assert report["metrics"]["false_positive"] == 1


def test_reports_metrics_separately_by_candidate_type(tmp_path: Path) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:02,000
何を？

2
00:00:03,000 --> 00:00:04,000
トイレ借りていい？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "何を",
                            "time_range": {
                                "start_seconds": 1.0,
                                "end_seconds": 2.0,
                            },
                            "evidence": ["short_pronominal_case_phrase"],
                        },
                        {
                            "segment_position": 1,
                            "sentence_index": 0,
                            "text": "トイレ借りていい",
                            "time_range": {
                                "start_seconds": 3.0,
                                "end_seconds": 4.0,
                            },
                            "evidence": ["embedded_quoted_question"],
                        },
                        {
                            "segment_position": 2,
                            "sentence_index": 0,
                            "text": "今日は晴れですか",
                            "time_range": {
                                "start_seconds": 5.0,
                                "end_seconds": 6.0,
                            },
                            "evidence": ["semantic_question_boundary"],
                        },
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(reference, artifact)

    grouped = report["metrics_by_candidate_type"]
    assert grouped["elliptical_question"]["precision"] == 1.0
    assert grouped["embedded_quoted_question"]["precision"] == 1.0
    assert grouped["sentence_terminal_question"]["precision"] == 0.0
    assert grouped["sentence_terminal_question"]["false_positive"] == 1


def test_embedded_question_uses_prediction_span_coverage(tmp_path: Path) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:05,000
友達の家で「トイレ借りていい？」って聞きました。
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "トイレ借りていい",
                            "time_range": {
                                "start_seconds": 2.0,
                                "end_seconds": 3.0,
                            },
                            "evidence": ["embedded_quoted_question"],
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(reference, artifact)

    assert report["metrics"]["true_positive"] == 1
    assert report["matches"][0]["terminal_text_similarity"] == 1.0
    assert report["metrics_by_candidate_type"]["embedded_quoted_question"][
        "precision"
    ] == 1.0


def test_zero_predictions_report_precision_as_not_applicable(tmp_path: Path) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:02,000
元気ですか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"data": {"candidates": []}}),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(reference, artifact)

    assert report["metrics"]["precision"] is None
    assert report["metrics"]["f1"] is None
    assert report["metrics"]["precision_target_met"] is None
    for metrics in report["metrics_by_candidate_type"].values():
        assert metrics["predicted_questions"] == 0
        assert metrics["precision"] is None
        assert metrics["precision_target_met"] is None


def test_reviewed_annotations_scope_false_negatives_by_owner(tmp_path: Path) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:02,000
候補規則ですか？

2
00:00:03,000 --> 00:00:04,000
断句の問題ですか？

3
00:00:05,000 --> 00:00:06,000
既に見えますか？

4
00:00:07,000 --> 00:00:08,000
字幕の重複ですか？

5
00:00:09,000 --> 00:00:10,000
音声認識の欠落ですか？

6
00:00:11,000 --> 00:00:12,000
料理を食べましたか？

7
00:00:13,000 --> 00:00:14,000
未確認ですか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"data": {"candidates": []}}),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.json"
    reasons = (
        "candidate_rule_missing",
        "language_sentence_error",
        "already_visible_question_mark",
        "reference_overlap_duplicate",
        "asr_omission_or_misalignment",
        "asr_error",
    )
    annotations.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_kind": "missed_reference",
                        "review_status": "reviewed",
                        "reference": {
                            "start_seconds": float(index * 2 + 1),
                            "end_seconds": float(index * 2 + 2),
                            "text": text,
                        },
                        "error_attribution": reason,
                    }
                    for index, (text, reason) in enumerate(
                        zip(
                            (
                                "候補規則ですか？",
                                "断句の問題ですか？",
                                "既に見えますか？",
                                "字幕の重複ですか？",
                                "音声認識の欠落ですか？",
                                "料理を食べましたか？",
                            ),
                            reasons,
                        )
                    )
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(
        reference,
        artifact,
        annotation_path=annotations,
    )

    assert report["evaluation_scope"] == "reviewed_candidate_rule"
    assert report["metrics"]["false_negative"] == 1
    assert report["raw_metrics"]["false_negative"] == 7
    assert report["error_attribution"]["candidate_rule_missing"] == 1
    assert report["error_attribution"]["language_sentence_error"] == 1
    assert report["error_attribution"]["excluded_false_negative"] == 4
    assert report["error_attribution"]["unreviewed"] == 1


def test_excluded_reference_cannot_be_true_positive_or_false_positive(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.srt"
    reference.write_text(
        """1
00:00:01,000 --> 00:00:02,000
人間じゃないっていうことですか？
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "segment_position": 0,
                            "sentence_index": 0,
                            "text": "人間じゃないっていうことですかな",
                            "time_range": {
                                "start_seconds": 1.0,
                                "end_seconds": 2.0,
                            },
                            "evidence": ["sentence_final_self_question"],
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_kind": "missed_reference",
                        "review_status": "reviewed",
                        "reference": {
                            "start_seconds": 1.0,
                            "end_seconds": 2.0,
                            "text": "人間じゃないっていうことですか？",
                        },
                        "error_attribution": "asr_error",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_question_punctuation(
        reference,
        artifact,
        annotation_path=annotations,
    )

    assert report["raw_metrics"]["true_positive"] == 1
    assert report["metrics"]["true_positive"] == 0
    assert report["metrics"]["false_positive"] == 0
    assert report["coverage"]["excluded_reference_questions"] == 1
    assert report["coverage"]["ignored_predictions"] == 1
    assert report["errors"]["ignored_prediction"][0]["text"].endswith("ですかな")

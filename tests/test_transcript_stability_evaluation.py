from __future__ import annotations

import json
from pathlib import Path

from jp_learning_platform.transcript_stability_evaluation import (
    evaluate_transcript_stability,
    write_transcript_stability_evaluation,
)


def _artifact(path: Path, source: str, segments: list[tuple[float, float, str]]) -> Path:
    payload = {
        "run_name": path.parent.name,
        "source_path": source,
        "context": {
            "document": {
                "source_path": source,
                "segments": [
                    {
                        "position": index,
                        "text": text,
                        "time_range": {
                            "start_seconds": start,
                            "end_seconds": end,
                        },
                        "sentences": [],
                    }
                    for index, (start, end, text) in enumerate(segments)
                ],
            }
        },
        "data": {"retry_decisions": []},
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_stability_evaluation_classifies_consensus_outlier_and_omission(
    tmp_path: Path,
) -> None:
    source = "input/lesson.mp3"
    inputs = (
        _artifact(
            tmp_path / "run-a" / "lesson" / "01_whisper.json",
            source,
            [(0, 1, "今日は晴れです。"), (2, 3, "駅に行きます。"), (4, 5, "またね。")],
        ),
        _artifact(
            tmp_path / "run-b" / "lesson" / "01_whisper.json",
            source,
            [(0, 1, "今日は晴れです"), (2, 3, "駅に行きます。")],
        ),
        _artifact(
            tmp_path / "run-c" / "lesson" / "01_whisper.json",
            source,
            [(0, 1, "今日は晴れです。"), (2, 3, "ご視聴ありがとうございました。"), (4, 5, "またね。")],
        ),
    )

    report = evaluate_transcript_stability(inputs)

    source_report = report["sources"][0]
    assert source_report["metrics"]["run_count"] == 3
    assert [item["classification"] for item in source_report["regions"]] == [
        "stable",
        "possible_hallucination",
        "possible_asr_omission",
    ]
    hallucination = source_report["regions"][1]
    assert hallucination["consensus_text"] == "駅に行きます。"
    assert hallucination["candidates"][0]["support_count"] == 2


def test_two_runs_keep_disagreement_as_unstable_text(tmp_path: Path) -> None:
    source = "input/lesson.mp3"
    inputs = (
        _artifact(
            tmp_path / "run-a" / "lesson" / "01_whisper.json",
            source,
            [(0, 1, "学校へ行きます。")],
        ),
        _artifact(
            tmp_path / "run-b" / "lesson" / "01_whisper.json",
            source,
            [(0, 1, "会社へ行きます。")],
        ),
    )

    report = evaluate_transcript_stability(inputs)

    assert report["sources"][0]["regions"][0]["classification"] == "unstable_text"


def test_writer_discovers_whisper_artifacts_in_run_directories(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _artifact(
        run_a / "lesson" / "01_whisper.json",
        str(run_a / "lesson" / "video-audio" / "pcm-s16le-mono-16000-v1.wav"),
        [(0, 1, "同じです。")],
    )
    _artifact(
        run_b / "lesson" / "01_whisper.json",
        str(run_b / "lesson" / "video-audio" / "pcm-s16le-mono-16000-v1.wav"),
        [(0, 1, "同じです。")],
    )
    output = tmp_path / "evaluation" / "stability.json"

    report = write_transcript_stability_evaluation((run_a, run_b), output)

    assert output.exists()
    assert report["coverage"]["comparable_sources"] == 1
    assert report["sources"][0]["source_key"] == "lesson"
    assert report["summary"]["classifications"] == {"stable": 1}

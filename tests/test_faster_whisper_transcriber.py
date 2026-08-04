from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jp_learning_platform.infrastructure import FasterWhisperTranscriber
from jp_learning_platform.infrastructure.transcript_anomaly_detector import (
    ConservativeTranscriptAnomalyDetector,
)
from jp_learning_platform.workflow import WhisperTranscriptionRequest
from jp_learning_platform.workflow.transcript_anomaly_stage import (
    TranscriptAnomalyRequest,
)


class RecordingWhisperModel:
    def __init__(self) -> None:
        self.source_path = ""
        self.options: dict[str, object] = {}

    def transcribe(
        self,
        source_path: str,
        **options: object,
    ) -> tuple[tuple[object, ...], object]:
        self.source_path = source_path
        self.options = options
        return (), object()


def test_faster_whisper_transcriber_uses_centralized_default_options(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    model = RecordingWhisperModel()
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
        )
    )

    assert result.source_path == source_path
    assert result.segments == ()
    assert model.source_path == str(source_path)
    assert model.options == {
        "language": "ja",
        "initial_prompt": (
            "これは日本語学習教材の書き起こしです。"
            "自然な日本語の句読点を使用します。"
        ),
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 600},
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
    }


class RetryWhisperModel:
    def __init__(self, responses: list[tuple[object, ...]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def transcribe(self, source_path: str, **options: object) -> tuple[tuple[object, ...], object]:
        self.calls.append(options)
        return self.responses.pop(0), object()


def _external_segment(
    text: str,
    start: float,
    end: float,
    confidence: float,
) -> object:
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        words=(
            SimpleNamespace(
                word=text,
                start=start,
                end=end,
                probability=confidence,
            ),
        ),
    )


def test_retries_only_low_confidence_segments_with_reliable_audio_context(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    reliable = _external_segment("今日は", 0.0, 1.0, 0.95)
    uncertain = _external_segment("日本こ", 1.1, 2.0, 0.4)
    repaired = _external_segment("日本語", 1.1, 2.0, 0.9)
    model = RetryWhisperModel([(reliable, uncertain), (repaired,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert tuple(segment.text for segment in result.segments) == ("今日は", "日本語")
    assert len(model.calls) == 2
    assert model.calls[0]["initial_prompt"] == (
        "これは日本語学習教材の書き起こしです。"
        "自然な日本語の句読点を使用します。"
    )
    assert model.calls[1]["initial_prompt"] == "今日は"
    assert model.calls[1]["clip_timestamps"] == [1.1, 2.0]


def test_keeps_first_pass_when_retry_confidence_does_not_improve(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = _external_segment("日本語", 0.0, 1.0, 0.5)
    retry = _external_segment("日本こ", 0.0, 1.0, 0.52)
    model = RetryWhisperModel([(original,), (retry,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert tuple(segment.text for segment in result.segments) == ("日本語",)


def test_retries_internal_word_gap_without_vad_and_accepts_coverage_repair(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="ゆっくりになってなりました。",
        start=35.0,
        end=42.0,
        words=(
            SimpleNamespace(word="ゆっくりになって", start=35.0, end=37.9, probability=0.95),
            SimpleNamespace(word="なりました。", start=41.4, end=42.0, probability=0.4),
        ),
    )
    repaired = SimpleNamespace(
        text="ゆっくりになって、止まりました。",
        start=35.0,
        end=42.0,
        words=(
            SimpleNamespace(word="ゆっくりになって、", start=35.0, end=38.0, probability=0.9),
            SimpleNamespace(word="止まりました。", start=38.1, end=42.0, probability=0.9),
        ),
    )
    model = RetryWhisperModel([(original,), (repaired,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == "ゆっくりになって、止まりました。"
    assert model.calls[1]["clip_timestamps"] == [35.0, 42.0]
    assert model.calls[1]["vad_filter"] is False


def test_rejects_internal_gap_retry_that_changes_surrounding_text(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="電車がゆっくりになってなりました。",
        start=35.0,
        end=42.0,
        words=(
            SimpleNamespace(word="電車がゆっくりになって", start=35.0, end=37.9, probability=0.95),
            SimpleNamespace(word="なりました。", start=41.4, end=42.0, probability=0.4),
        ),
    )
    unrelated = _external_segment("駅で昼ご飯を食べました。", 35.0, 42.0, 0.9)
    model = RetryWhisperModel([(original,), (unrelated,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == original.text


@pytest.mark.parametrize(
    ("original_text", "degraded_text"),
    (
        (
            "今回はトーマス列車ではありませんでした。",
            "今回はトーマス列車ではませんでした。",
        ),
        (
            "大月に戻ってきた時は雨が降っていました。",
            "大月に戻ってきた時はが降っていました。",
        ),
    ),
)
def test_rejects_internal_gap_retry_that_introduces_grammatical_degradation(
    tmp_path: Path,
    original_text: str,
    degraded_text: str,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text=original_text,
        start=0.0,
        end=6.0,
        words=(
            SimpleNamespace(word=original_text[:5], start=0.0, end=1.0, probability=0.9),
            SimpleNamespace(word=original_text[5:], start=4.0, end=6.0, probability=0.4),
        ),
    )
    degraded = SimpleNamespace(
        text=degraded_text,
        start=0.0,
        end=6.0,
        words=(
            SimpleNamespace(word=degraded_text, start=0.0, end=6.0, probability=0.95),
        ),
    )
    model = RetryWhisperModel([(original,), (degraded,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == original_text


@pytest.mark.parametrize(
    ("original_text", "candidate_text"),
    (
        ("今日は雨が降りました。", "今日は降りました。"),
        ("私は駅で電車を待ちました。", "私は駅電車を待ちました。"),
        ("景色が面白いです。", "景色が白いです。"),
    ),
)
def test_internal_gap_retry_requires_confidence_gain_for_meaningful_deletion(
    tmp_path: Path,
    original_text: str,
    candidate_text: str,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text=original_text,
        start=0.0,
        end=6.0,
        words=(
            SimpleNamespace(word=original_text[:3], start=0.0, end=1.0, probability=0.8),
            SimpleNamespace(word=original_text[3:], start=4.0, end=6.0, probability=0.8),
        ),
    )
    candidate = SimpleNamespace(
        text=candidate_text,
        start=0.0,
        end=6.0,
        words=(
            SimpleNamespace(word=candidate_text, start=0.0, end=6.0, probability=0.82),
        ),
    )
    model = RetryWhisperModel([(original,), (candidate,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == original_text


def test_internal_gap_retry_accepts_meaningful_deletion_with_strong_confidence_gain(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="今日はあの雨が降りました。",
        start=0.0,
        end=6.0,
        words=(
            SimpleNamespace(word="今日はあの", start=0.0, end=1.0, probability=0.6),
            SimpleNamespace(word="雨が降りました。", start=4.0, end=6.0, probability=0.7),
        ),
    )
    candidate = SimpleNamespace(
        text="今日は雨が降りました。",
        start=0.0,
        end=6.0,
        words=(
            SimpleNamespace(word="今日は雨が降りました。", start=0.0, end=6.0, probability=0.9),
        ),
    )
    model = RetryWhisperModel([(original,), (candidate,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == candidate.text


def test_internal_gap_retry_rejects_low_original_character_coverage(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="駅ではたくさんの人が電車を待っていました。",
        start=0.0,
        end=6.0,
        words=(
            SimpleNamespace(word="駅ではたくさんの人が", start=0.0, end=1.0, probability=0.6),
            SimpleNamespace(word="電車を待っていました。", start=4.0, end=6.0, probability=0.6),
        ),
    )
    shortened = SimpleNamespace(
        text="駅で待っていました。",
        start=0.0,
        end=6.0,
        words=(
            SimpleNamespace(word="駅で待っていました。", start=0.0, end=6.0, probability=0.99),
        ),
    )
    model = RetryWhisperModel([(original,), (shortened,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == original.text
    assert len(result.retry_decisions) == 1
    decision = result.retry_decisions[0]
    assert not decision.accepted
    assert "low_original_character_coverage" in decision.reasons
    assert decision.original_character_coverage < 0.8

    anomalies = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(source_path, result.segments)
    )
    assert len(anomalies) == 1
    assert anomalies[0].kind == "possible_internal_asr_omission"
    assert anomalies[0].time_range == decision.time_range


def test_internal_gap_retry_rejects_language_model_score_regression(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="電車がゆっくりになってなりました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.2,
        words=(
            SimpleNamespace(word="電車がゆっくりになって", start=0.0, end=1.0, probability=0.6),
            SimpleNamespace(word="なりました。", start=4.0, end=6.0, probability=0.4),
        ),
    )
    candidate = SimpleNamespace(
        text="電車がゆっくりになって止まりました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.5,
        words=(
            SimpleNamespace(
                word="電車がゆっくりになって止まりました。",
                start=0.0,
                end=6.0,
                probability=0.95,
            ),
        ),
    )
    model = RetryWhisperModel([(original,), (candidate,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == original.text

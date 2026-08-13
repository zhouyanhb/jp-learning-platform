from pathlib import Path

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.transcript_anomaly_detector import (
    ConservativeTranscriptAnomalyDetector,
)
from jp_learning_platform.workflow.transcript_anomaly_stage import (
    TranscriptAnomalyRequest,
)


def _segment(position: int, text: str, start: float, end: float, confidence: float) -> Segment:
    time_range = TimeRange(start, end)
    sentence = Sentence(text, time_range, (Word(text, time_range, confidence),))
    return Segment(position, text, time_range, (sentence,))


def test_detects_anomalies_without_changing_or_inspecting_transcript_text() -> None:
    segments = (
        _segment(0, "甲", 0.0, 1.0, 0.9),
        _segment(1, "乙", 3.0, 4.0, 0.4),
        _segment(2, "丙", 4.2, 5.2, 0.9),
    )

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("audio.mp3"), segments)
    )

    assert {candidate.kind for candidate in candidates} == {
        "possible_asr_omission",
        "possible_background_speech",
    }
    assert segments[1].text == "乙"


def test_does_not_label_stable_silence_gap_as_missing_content() -> None:
    segments = (
        _segment(0, "前", 0.0, 1.0, 0.95),
        _segment(1, "後", 4.0, 5.0, 0.95),
    )

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("audio.mp3"), segments)
    )

    assert candidates == ()


def test_detects_low_confidence_gap_inside_one_asr_segment() -> None:
    time_range = TimeRange(35.0, 42.0)
    sentence = Sentence(
        "ゆっくりになってなりました",
        time_range,
        (
            Word("ゆっくりになって", TimeRange(35.0, 37.9), 0.95),
            Word("なりました", TimeRange(41.4, 42.0), 0.4),
        ),
    )
    segment = Segment(0, sentence.text, time_range, (sentence,))

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("audio.mp3"), (segment,))
    )

    assert len(candidates) == 1
    assert candidates[0].kind == "possible_internal_asr_omission"
    assert candidates[0].time_range == TimeRange(37.9, 41.4)


def test_preserves_rejected_morphological_asr_error_for_review() -> None:
    time_range = TimeRange(658.88, 669.57)
    sentence = Sentence(
        "多分、いいなっても悩まないことなんてないと思います。",
        time_range,
        (
            Word("多分、", TimeRange(658.88, 661.88), 0.95),
            Word("いい", TimeRange(662.12, 662.36), 0.226),
            Word(
                "なっても悩まないことなんてないと思います。",
                TimeRange(662.78, 669.57),
                0.95,
            ),
        ),
    )
    segment = Segment(151, sentence.text, time_range, (sentence,))

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("audio.mp3"), (segment,))
    )

    assert len(candidates) == 1
    assert candidates[0].kind == "possible_morphological_asr_error"
    assert candidates[0].time_range == time_range

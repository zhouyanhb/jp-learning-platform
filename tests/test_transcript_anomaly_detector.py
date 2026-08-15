from pathlib import Path

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    TimeRange,
    Word,
)
from jp_learning_platform.infrastructure.transcript_anomaly_detector import (
    ConservativeTranscriptAnomalyDetector,
)
from jp_learning_platform.workflow.transcript_anomaly_stage import (
    TranscriptAnomalyIsolationStage,
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


def test_detects_repeated_laughter_as_independent_anomaly() -> None:
    segment = _segment(7, "あははは", 12.0, 13.2, 0.96)

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("drama.mp4"), (segment,))
    )

    assert len(candidates) == 1
    assert candidates[0].kind == "possible_repeated_laughter"
    assert candidates[0].time_range == segment.time_range
    assert candidates[0].evidence == (
        "non_lexical_utterance",
        "repeated_laughter_syllables",
    )


def test_detects_explicit_background_sound_annotation() -> None:
    segment = _segment(3, "【拍手】", 8.0, 9.5, 0.99)

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("video.webm"), (segment,))
    )

    assert len(candidates) == 1
    assert candidates[0].kind == "possible_background_sound"
    assert candidates[0].time_range == segment.time_range


def test_does_not_treat_lexical_laughter_or_music_as_background_sound() -> None:
    segments = (
        _segment(0, "笑顔で話しています", 0.0, 1.0, 0.95),
        _segment(1, "音楽教室へ行きます", 1.1, 2.1, 0.95),
    )

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("speech.mp3"), segments)
    )

    assert candidates == ()


def test_does_not_treat_repeated_acknowledgement_as_laughter() -> None:
    segment = _segment(0, "あーはいはい", 0.0, 1.0, 0.95)

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("speech.mp3"), (segment,))
    )

    assert candidates == ()


def test_classifies_unaligned_repeated_vocalization_separately() -> None:
    time_range = TimeRange(10.0, 11.0)
    sentence = Sentence("ペッドペッドペッドペッド", time_range)
    segment = Segment(4, sentence.text, time_range, (sentence,))

    candidates = ConservativeTranscriptAnomalyDetector().detect(
        TranscriptAnomalyRequest(Path("drama.mp4"), (segment,))
    )

    assert {candidate.kind for candidate in candidates} == {
        "possible_alignment_failure",
        "possible_repeated_vocalization",
    }
    alignment = next(
        item for item in candidates if item.kind == "possible_alignment_failure"
    )
    assert alignment.sentence_indexes == (0,)


def test_isolation_keeps_text_but_suppresses_unreliable_learning_words(
    tmp_path: Path,
) -> None:
    time_range = TimeRange(10.0, 11.0)
    sentence = Sentence("ご視聴ありがとうございました", time_range)
    segment = Segment(4, sentence.text, time_range, (sentence,))
    context = PipelineContext(
        run_id="run-1",
        document=Document(Path("drama.mp4"), (segment,)),
        working_directory=tmp_path,
    )

    result = TranscriptAnomalyIsolationStage(
        ConservativeTranscriptAnomalyDetector()
    ).run(context)

    isolated = result.context.document.segments[0].sentences[0]
    assert isolated.text == sentence.text
    assert isolated.anomaly_kinds == ("possible_alignment_failure",)
    assert isolated.excluded_from_language_evaluation
    assert isolated.learning_words_suppressed
    assert isolated.learning_words == ()
    assert result.data["isolated_sentence_count"] == 1

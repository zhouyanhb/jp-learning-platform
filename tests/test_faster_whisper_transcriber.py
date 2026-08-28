from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure import FasterWhisperTranscriber
from jp_learning_platform.infrastructure.faster_whisper_transcriber import (
    _extract_external_retry_between_anchors,
    _extract_external_retry_window,
    _candidate_morpheme_disagreements,
    _has_ordered_text_anchors,
    _instantiate_whisper_model,
    _short_utterance_structure_penalty,
    _sentence_initial_uncertain_noun_sequence,
)
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    SudachiMorphologicalAnalyzer,
)
from jp_learning_platform.infrastructure.transcript_anomaly_detector import (
    ConservativeTranscriptAnomalyDetector,
)
from jp_learning_platform.workflow import WhisperTranscriptionRequest
from jp_learning_platform.workflow.transcript_anomaly_stage import (
    TranscriptAnomalyCandidate,
    TranscriptAnomalyRequest,
)
from jp_learning_platform.workflow.transcript_omission_shadow_stage import (
    TranscriptOmissionShadowRequest,
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


def test_whisper_model_loader_prefers_complete_local_cache() -> None:
    calls: list[bool] = []

    def model_class(model_name: str, **options: object) -> object:
        assert model_name == "turbo"
        calls.append(bool(options["local_files_only"]))
        return object()

    _instantiate_whisper_model(model_class, "turbo", device="auto")

    assert calls == [True]


def test_whisper_model_loader_downloads_only_when_local_snapshot_is_missing() -> None:
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[bool] = []

    def model_class(model_name: str, **options: object) -> object:
        del model_name
        local_only = bool(options["local_files_only"])
        calls.append(local_only)
        if local_only:
            raise LocalEntryNotFoundError("not cached")
        return object()

    _instantiate_whisper_model(model_class, "turbo", device="auto")

    assert calls == [True, False]


def test_whisper_model_loader_does_not_hide_invalid_local_model_error() -> None:
    def model_class(model_name: str, **options: object) -> object:
        del model_name, options
        raise ValueError("invalid model")

    with pytest.raises(ValueError, match="invalid model"):
        _instantiate_whisper_model(model_class, "turbo", device="auto")


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
        response = self.responses[0] if len(self.responses) == 1 else self.responses.pop(0)
        return response, object()


def test_omission_shadow_retries_without_vad_and_never_replaces_segments(
    tmp_path: Path,
) -> None:
    recovered = _external_segment(
        "陸は仕事に対してひたすら真面目です。",
        27.0,
        37.8,
        0.9,
    )
    model = RetryWhisperModel([(recovered,), (recovered,), (recovered,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model
    left = _domain_segment(0, "三頭がデビューしました。", 18.0, 26.5)
    right = _domain_segment(1, "三頭は各地で活躍します。", 38.1, 43.8)
    candidate = TranscriptAnomalyCandidate(
        kind="possible_asr_omission",
        time_range=TimeRange(26.5, 38.1),
        segment_positions=(0, 1),
        confidence=0.74,
        evidence=(
            "long_uncovered_time_range",
            "substantial_stable_context",
        ),
    )

    audits = transcriber.recognize_omission_candidates(
        TranscriptOmissionShadowRequest(
            source_path=tmp_path / "news.m4a",
            segments=(left, right),
            candidates=(candidate,),
        )
    )

    assert len(audits) == 1
    assert audits[0].candidate_consensus_count == 3
    assert audits[0].consensus_reached
    assert audits[0].candidate_consensus_text == recovered.text
    assert audits[0].recovered_time_coverage == (0.931, 0.931, 0.931)
    assert audits[0].validation_passed
    assert not audits[0].automatic_replacement_allowed
    assert all(call["vad_filter"] is False for call in model.calls)
    assert all(call["clip_timestamps"] == [25.0, 39.6] for call in model.calls)
    assert (left.text, right.text) == (
        "三頭がデビューしました。",
        "三頭は各地で活躍します。",
    )


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
        avg_logprob=-0.2,
        words=(
            SimpleNamespace(
                word=text,
                start=start,
                end=end,
                probability=confidence,
            ),
        ),
    )


def test_omission_shadow_ignores_punctuation_but_exposes_lexical_disagreement(
    tmp_path: Path,
) -> None:
    candidates = (
        _external_segment("一緒の状態は問題ありません。", 27.0, 37.8, 0.9),
        _external_segment("一緒の状態は問題ありません", 27.0, 37.8, 0.9),
        _external_segment("衣装の状態は問題ありません。", 27.0, 37.8, 0.9),
    )
    model = RetryWhisperModel([(item,) for item in candidates])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model
    request = TranscriptOmissionShadowRequest(
        source_path=tmp_path / "news.m4a",
        segments=(
            _domain_segment(0, "前の安定したニュース文脈です。", 18.0, 26.5),
            _domain_segment(1, "後の安定したニュース文脈です。", 38.1, 43.8),
        ),
        candidates=(
            TranscriptAnomalyCandidate(
                kind="possible_asr_omission",
                time_range=TimeRange(26.5, 38.1),
                segment_positions=(0, 1),
                confidence=0.8,
                evidence=(
                    "long_uncovered_time_range",
                    "substantial_stable_context",
                ),
            ),
        ),
    )

    audit = transcriber.recognize_omission_candidates(request)[0]

    assert audit.candidate_consensus_count == 2
    assert audit.normalized_candidate_texts[:2] == (
        "一緒の状態は問題ありません",
        "一緒の状態は問題ありません",
    )
    assert "状態" in audit.core_character_consensus
    assert "状態" in audit.core_morpheme_consensus
    assert any(
        {item.left_fragment, item.right_fragment} == {"一緒", "衣装"}
        for item in audit.candidate_disagreements
    )
    assert "lexical_candidate_disagreement" in audit.review_reasons
    assert not audit.validation_passed
    assert not audit.automatic_replacement_allowed


def test_morpheme_boundary_only_difference_is_not_a_lexical_disagreement() -> None:
    disagreements = _candidate_morpheme_disagreements(
        (("関し", "て", "は"), ("関し", "ては"))
    )

    assert disagreements == ()


def test_omission_shadow_requires_review_for_unlexicalized_initial_nouns(
    tmp_path: Path,
) -> None:
    recovered = _external_segment(
        "陸豪は非常に好奇心が強く仕事に対して真面目です。",
        27.0,
        37.8,
        0.95,
    )
    model = RetryWhisperModel([(recovered,), (recovered,), (recovered,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model
    request = TranscriptOmissionShadowRequest(
        source_path=tmp_path / "news.m4a",
        segments=(
            _domain_segment(0, "前の安定したニュース文脈です。", 18.0, 26.5),
            _domain_segment(1, "後の安定したニュース文脈です。", 38.1, 43.8),
        ),
        candidates=(
            TranscriptAnomalyCandidate(
                kind="possible_asr_omission",
                time_range=TimeRange(26.5, 38.1),
                segment_positions=(0, 1),
                confidence=0.8,
                evidence=(
                    "long_uncovered_time_range",
                    "substantial_stable_context",
                ),
            ),
        ),
    )

    audit = transcriber.recognize_omission_candidates(request)[0]

    assert audit.consensus_reached
    assert audit.context_validation_passed
    assert audit.confidence_validation_passed
    assert audit.language_model_validation_passed
    assert audit.morphology_validation_passed
    assert audit.lexical_uncertainty_detected
    assert audit.uncertain_noun_sequences == ("陸豪",)
    assert audit.lexical_uncertainty_reasons == (
        "unlexicalized_sentence_initial_noun_sequence",
    )
    assert "lexical_uncertainty_requires_review" in audit.review_reasons
    assert not audit.validation_passed
    assert not audit.automatic_replacement_allowed


@pytest.mark.parametrize(
    "text",
    (
        "日本は美しいです。",
        "東京大学は有名です。",
        "衣装の状態は問題ありません。",
    ),
)
def test_common_topic_phrases_are_not_marked_as_uncertain_noun_sequences(
    text: str,
) -> None:
    analyzer = SudachiMorphologicalAnalyzer()

    assert not _sentence_initial_uncertain_noun_sequence(text, analyzer)


def _domain_segment(position: int, text: str, start: float, end: float) -> Segment:
    time_range = TimeRange(start, end)
    sentence = Sentence(
        text,
        time_range,
        (Word(text, time_range, 0.9),),
    )
    return Segment(position, text, time_range, (sentence,))


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


def test_internal_gap_retry_accepts_bounded_morphological_repair(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="電車がゆっくりになってなりました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.03,
        words=(
            SimpleNamespace(
                word="電車がゆっくりになって",
                start=0.0,
                end=1.0,
                probability=0.99,
            ),
            SimpleNamespace(
                word="なりました。",
                start=4.0,
                end=6.0,
                probability=0.6,
            ),
        ),
    )
    candidate = SimpleNamespace(
        text="電車がゆっくりになって止まりました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.149,
        words=(
            SimpleNamespace(
                word="電車がゆっくりになって止まりました。",
                start=0.0,
                end=6.0,
                probability=0.794,
            ),
        ),
    )
    model = RetryWhisperModel([(original,), (candidate,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == candidate.text
    assert result.retry_decisions[0].accepted
    assert result.retry_decisions[0].reasons == (
        "accepted_morphological_repair",
    )


def test_internal_gap_retry_does_not_exempt_ordinary_content_change(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="電車がゆっくりになっていました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.03,
        words=(
            SimpleNamespace(
                word="電車がゆっくりになって",
                start=0.0,
                end=1.0,
                probability=0.99,
            ),
            SimpleNamespace(
                word="いました。",
                start=4.0,
                end=6.0,
                probability=0.6,
            ),
        ),
    )
    candidate = SimpleNamespace(
        text="電車がゆっくりになって止まりました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.149,
        words=(
            SimpleNamespace(
                word="電車がゆっくりになって止まりました。",
                start=0.0,
                end=6.0,
                probability=0.794,
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
    assert not result.retry_decisions[0].accepted
    assert "language_model_regression" in result.retry_decisions[0].reasons


def test_internal_gap_retry_selects_supported_safe_candidate(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="電車がゆっくりになってなりました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.1,
        words=(
            SimpleNamespace(
                word="電車がゆっくりになって",
                start=0.0,
                end=1.0,
                probability=0.9,
            ),
            SimpleNamespace(
                word="なりました。",
                start=4.0,
                end=6.0,
                probability=0.5,
            ),
        ),
    )
    unsafe = SimpleNamespace(
        text="駅で昼ご飯を食べました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.01,
        words=(
            SimpleNamespace(
                word="駅で昼ご飯を食べました。",
                start=0.0,
                end=6.0,
                probability=0.99,
            ),
        ),
    )
    repaired = SimpleNamespace(
        text="電車がゆっくりになって止まりました。",
        start=0.0,
        end=6.0,
        avg_logprob=-0.2,
        words=(
            SimpleNamespace(
                word="電車がゆっくりになって止まりました。",
                start=0.0,
                end=6.0,
                probability=0.82,
            ),
        ),
    )
    model = RetryWhisperModel(
        [(original,), (unsafe,), (repaired,), (repaired,)]
    )
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == repaired.text
    assert len(result.retry_decisions) == 2
    selected = next(item for item in result.retry_decisions if item.selected)
    rejected = next(item for item in result.retry_decisions if not item.selected)
    assert selected.passed_validation
    assert selected.candidate_support_count == 2
    assert selected.selection_score is not None
    assert not rejected.passed_validation
    assert rejected.candidate_text == unsafe.text


def test_low_confidence_morphological_chain_triggers_local_repair(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="多分、いいなっても悩まないことなんてないと思います。",
        start=0.0,
        end=10.0,
        avg_logprob=-0.2,
        words=(
            SimpleNamespace(word="多分、", start=0.0, end=2.0, probability=0.95),
            SimpleNamespace(word="いい", start=2.0, end=3.0, probability=0.2),
            SimpleNamespace(
                word="なっても悩まないことなんてないと思います。",
                start=3.0,
                end=10.0,
                probability=0.95,
            ),
        ),
    )
    repaired = SimpleNamespace(
        text="たぶんいくつになっても悩まないことなんてないと思います。",
        start=0.0,
        end=10.0,
        avg_logprob=-0.18,
        words=(
            SimpleNamespace(
                word="たぶんいくつになっても悩まないことなんてないと思います。",
                start=0.0,
                end=10.0,
                probability=0.9,
            ),
        ),
    )
    model = RetryWhisperModel(
        [(original,), (repaired,), (repaired,), (repaired,)]
    )
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == repaired.text
    assert len(model.calls) == 4
    assert result.retry_decisions[0].selected
    assert result.retry_decisions[0].passed_validation
    assert result.retry_decisions[0].candidate_support_count == 3
    assert result.retry_decisions[0].reasons == ("accepted_morphological_repair",)


def test_morphological_retry_preserves_terminal_mark_and_original_time_range(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="何歳になっても悩みます。",
        start=1.0,
        end=8.0,
        avg_logprob=-0.2,
        words=(
            SimpleNamespace(word="何歳に", start=1.0, end=3.0, probability=0.2),
            SimpleNamespace(word="なっても悩みます。", start=3.0, end=8.0, probability=0.9),
        ),
    )
    repaired = SimpleNamespace(
        text="いくつになっても悩みます",
        start=1.1,
        end=7.2,
        avg_logprob=-0.1,
        words=(
            SimpleNamespace(
                word="いくつになっても悩みます",
                start=1.1,
                end=7.2,
                probability=0.99,
            ),
        ),
    )
    model = RetryWhisperModel(
        [(original,), (repaired,), (repaired,), (repaired,)]
    )
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == "いくつになっても悩みます。"
    assert result.segments[0].time_range.start_seconds == 1.0
    assert result.segments[0].time_range.end_seconds == 8.0
    final_word = result.segments[0].sentences[0].words[-1]
    assert final_word.text.endswith("。")
    assert final_word.time_range.end_seconds == 8.0


def test_morphological_retry_rejects_candidate_that_keeps_anomaly(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="多分、いいなっても悩まない。",
        start=0.0,
        end=5.0,
        words=(
            SimpleNamespace(word="多分、", start=0.0, end=1.0, probability=0.9),
            SimpleNamespace(word="いい", start=1.0, end=2.0, probability=0.2),
            SimpleNamespace(
                word="なっても悩まない。",
                start=2.0,
                end=5.0,
                probability=0.9,
            ),
        ),
    )
    unchanged = SimpleNamespace(
        text=original.text,
        start=0.0,
        end=5.0,
        words=(
            SimpleNamespace(word=original.text, start=0.0, end=5.0, probability=0.99),
        ),
    )
    model = RetryWhisperModel(
        [(original,), (unchanged,), (unchanged,), (unchanged,)]
    )
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == original.text
    assert len(result.retry_decisions) == 1
    assert not result.retry_decisions[0].passed_validation
    assert not result.retry_decisions[0].selected


def test_short_high_confidence_response_accepts_consensus_repair(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="なにょ",
        start=1.0,
        end=1.5,
        avg_logprob=-0.3,
        words=(
            SimpleNamespace(word="な", start=1.0, end=1.3, probability=0.90),
            SimpleNamespace(word="にょ", start=1.3, end=1.5, probability=0.91),
        ),
    )
    repaired = SimpleNamespace(
        text="何を",
        start=1.0,
        end=1.5,
        avg_logprob=-0.1,
        words=(
            SimpleNamespace(word="何を", start=1.0, end=1.5, probability=0.97),
        ),
    )
    model = RetryWhisperModel(
        [(original,), (repaired,), (repaired,), (repaired,)]
    )
    transcriber = FasterWhisperTranscriber(retry_max_segments=0)
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == "何を"
    assert len(model.calls) == 4
    assert result.retry_decisions[0].accepted
    assert result.retry_decisions[0].candidate_support_count == 3
    assert result.retry_decisions[0].reasons == ("accepted_morphological_repair",)
    assert len(result.short_anomaly_retry_audits) == 1
    assert result.short_anomaly_retry_audits[0].accepted
    assert result.short_anomaly_retry_audits[0].failure_reasons == ()
    analysis = result.short_utterance_analysis_audits[0]
    assert analysis.original_text == "なにょ"
    assert analysis.morpheme_surfaces == ("な", "にょ")
    assert analysis.structure_penalty == 1
    assert analysis.short_anomaly_detected


def test_short_high_confidence_response_rejects_disagreeing_candidates(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="なにょ",
        start=1.0,
        end=1.5,
        avg_logprob=-0.3,
        words=(
            SimpleNamespace(word="な", start=1.0, end=1.3, probability=0.90),
            SimpleNamespace(word="にょ", start=1.3, end=1.5, probability=0.91),
        ),
    )
    candidates = tuple(
        SimpleNamespace(
            text=text,
            start=1.0,
            end=1.5,
            avg_logprob=-0.1,
            words=(
                SimpleNamespace(word=text, start=1.0, end=1.5, probability=0.97),
            ),
        )
        for text in ("何を", "何の", "何よ")
    )
    model = RetryWhisperModel(
        [(original,), *((candidate,) for candidate in candidates)]
    )
    transcriber = FasterWhisperTranscriber(retry_max_segments=0)
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == original.text
    assert len(result.retry_decisions) == 3
    assert not any(item.accepted for item in result.retry_decisions)
    assert result.short_anomaly_retry_audits[0].failure_reasons == (
        "candidate_consensus_missing",
    )


def test_short_well_formed_response_does_not_trigger_anomaly_retry(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="そうだね",
        start=1.0,
        end=1.5,
        avg_logprob=-0.1,
        words=(
            SimpleNamespace(word="そう", start=1.0, end=1.3, probability=0.95),
            SimpleNamespace(word="だね", start=1.3, end=1.5, probability=0.96),
        ),
    )
    model = RetryWhisperModel([(original,)])
    transcriber = FasterWhisperTranscriber(retry_max_segments=0)
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == original.text
    assert len(model.calls) == 1
    assert result.retry_decisions == ()
    assert len(result.short_utterance_analysis_audits) == 1
    assert not result.short_utterance_analysis_audits[0].short_anomaly_detected


@pytest.mark.parametrize(
    ("text", "expected_penalty"),
    (
        ("てかさ", 0),
        ("ってかさ", 0),
        ("じゃあさ", 0),
        ("だからさ", 0),
        ("そうだね", 0),
        ("なにょ", 1),
    ),
)
def test_short_utterance_penalty_distinguishes_complete_colloquial_markers(
    text: str,
    expected_penalty: int,
) -> None:
    assert _short_utterance_structure_penalty(
        text,
        SudachiMorphologicalAnalyzer(),
    ) == expected_penalty


def test_complete_colloquial_marker_does_not_trigger_short_retry(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="てかさ",
        start=1.0,
        end=1.7,
        avg_logprob=-0.1,
        words=(
            SimpleNamespace(word="て", start=1.0, end=1.2, probability=0.95),
            SimpleNamespace(word="か", start=1.2, end=1.4, probability=0.95),
            SimpleNamespace(word="さ", start=1.4, end=1.7, probability=0.95),
        ),
    )
    model = RetryWhisperModel([(original,)])
    transcriber = FasterWhisperTranscriber(retry_max_segments=0)
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == "てかさ"
    assert len(model.calls) == 1
    assert result.short_anomaly_retry_audits == ()
    assert not result.short_utterance_analysis_audits[0].short_anomaly_detected


def test_low_confidence_retry_rejects_new_short_morphological_anomaly(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    original = SimpleNamespace(
        text="何を",
        start=1.0,
        end=1.7,
        avg_logprob=-0.4,
        words=(
            SimpleNamespace(word="何", start=1.0, end=1.3, probability=0.55),
            SimpleNamespace(word="を", start=1.3, end=1.7, probability=0.60),
        ),
    )
    malformed = SimpleNamespace(
        text="なにょ",
        start=1.0,
        end=1.7,
        avg_logprob=-0.1,
        words=(
            SimpleNamespace(word="な", start=1.0, end=1.3, probability=0.90),
            SimpleNamespace(word="にょ", start=1.3, end=1.7, probability=0.91),
        ),
    )
    model = RetryWhisperModel([(original,), (malformed,)])
    transcriber = FasterWhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe(
        WhisperTranscriptionRequest(source_path, tmp_path / "work", "run-001")
    )

    assert result.segments[0].text == "何を"
    assert len(result.retry_decisions) == 1
    assert not result.retry_decisions[0].accepted
    assert "morphological_structure_degradation" in (
        result.retry_decisions[0].reasons
    )


def test_short_retry_window_excludes_word_from_preceding_segment() -> None:
    original = SimpleNamespace(text="なにょ", start=504.17, end=504.89)
    retry_segments = (
        SimpleNamespace(
            text="一応店員さん伝えとこうか",
            start=502.67,
            end=504.21,
            avg_logprob=-0.4,
            words=(
                SimpleNamespace(
                    word="か", start=504.01, end=504.21, probability=0.98
                ),
            ),
        ),
        SimpleNamespace(
            text="え、なにを?",
            start=504.21,
            end=504.99,
            avg_logprob=-0.4,
            words=(
                SimpleNamespace(
                    word="え、", start=504.21, end=504.57, probability=0.39
                ),
                SimpleNamespace(
                    word="なにを?", start=504.69, end=504.99, probability=0.70
                ),
            ),
        ),
    )

    extracted = _extract_external_retry_window(original, retry_segments)

    assert len(extracted) == 1
    assert extracted[0].text == "え、なにを?"
    assert tuple(word.word for word in extracted[0].words) == ("え、", "なにを?")
    assert extracted[0].start == original.start
    assert extracted[0].end == original.end


def test_short_retry_uses_surrounding_text_anchors() -> None:
    original = SimpleNamespace(text="なにょ", start=504.17, end=504.89)
    left = SimpleNamespace(text="一応店員さん伝えとこうか")
    right = SimpleNamespace(text="なんか今回そのバチバチのやつやらないのは")

    def segment(text: str, start: float, end: float) -> SimpleNamespace:
        return SimpleNamespace(
            text=text,
            start=start,
            end=end,
            avg_logprob=-0.2,
            words=(
                SimpleNamespace(
                    word=text,
                    start=start,
                    end=end,
                    probability=0.9,
                ),
            ),
        )

    retry_segments = (
        segment(left.text, 502.67, 504.21),
        segment("え、なにを?", 504.21, 504.99),
        segment("いや、うん", 505.27, 506.25),
        segment(right.text, 506.60, 509.14),
    )

    extracted = _extract_external_retry_between_anchors(
        original,
        retry_segments,
        left,
        right,
    )

    assert len(extracted) == 1
    assert extracted[0].text == "え、なにを?"


def test_short_retry_accepts_ordered_anchors_inside_merged_segment() -> None:
    left = SimpleNamespace(text="一応店員さん伝えとこうか")
    right = SimpleNamespace(text="なんか今回そのバチバチのやつやらないのは")
    retry_segments = (
        SimpleNamespace(
            text=(
                "一応店員さん伝えとこうかいや"
                "何か今回そのマチバチのやつやらないのは"
            )
        ),
    )

    assert _has_ordered_text_anchors(retry_segments, left, right)

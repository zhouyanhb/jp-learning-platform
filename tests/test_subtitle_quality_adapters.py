from __future__ import annotations

from pathlib import Path

from jp_learning_platform.domain import Segment, Subtitle, TimeRange
from jp_learning_platform.infrastructure import (
    ConservativeSubtitleMerger,
    DomainSubtitleValidator,
    LocalReadabilityOptimizer,
    PassthroughQwenRepairer,
    PassthroughWhisperXAligner,
)
from jp_learning_platform.workflow import (
    QwenRepairRequest,
    ReadabilityOptimizationRequest,
    SubtitleMergeRequest,
    SubtitleValidationRequest,
    WhisperXAlignmentRequest,
)


def _segment() -> Segment:
    return Segment(
        position=0,
        text="日本語です",
        time_range=TimeRange(0.0, 1.0),
    )


def _subtitle(
    index: int,
    text: str,
    start: float,
    end: float,
    speaker_id: str | None = None,
) -> Subtitle:
    return Subtitle(
        index=index,
        text=text,
        time_range=TimeRange(start, end),
        speaker_id=speaker_id,
    )


def test_passthrough_whisperx_aligner_keeps_segments() -> None:
    segment = _segment()

    result = PassthroughWhisperXAligner().align(
        WhisperXAlignmentRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    assert result.segments == (segment,)


def test_passthrough_qwen_repairer_keeps_segments() -> None:
    segment = _segment()

    result = PassthroughQwenRepairer().repair(
        QwenRepairRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(segment,),
        )
    )

    assert result.segments == (segment,)


def test_conservative_subtitle_merger_merges_short_adjacent_cues() -> None:
    subtitles = (
        _subtitle(1, "日本語", 0.0, 0.5),
        _subtitle(2, "です。", 0.6, 1.0),
    )

    result = ConservativeSubtitleMerger().merge(
        SubtitleMergeRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=subtitles,
        )
    )

    assert len(result.subtitles) == 1
    assert result.subtitles[0].text == "日本語です。"
    assert result.subtitles[0].index == 1
    assert result.subtitles[0].time_range == TimeRange(0.0, 1.0)


def test_conservative_subtitle_merger_does_not_merge_different_speakers() -> None:
    subtitles = (
        _subtitle(1, "そう", 0.0, 0.3, speaker_id="speaker-1"),
        _subtitle(2, "はい。", 0.4, 0.8, speaker_id="speaker-2"),
    )

    result = ConservativeSubtitleMerger().merge(
        SubtitleMergeRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=subtitles,
        )
    )

    assert len(result.subtitles) == 2
    assert tuple(subtitle.text for subtitle in result.subtitles) == ("そう", "はい。")
    assert tuple(subtitle.speaker_id for subtitle in result.subtitles) == (
        "speaker-1",
        "speaker-2",
    )


def test_conservative_subtitle_merger_keeps_sentence_final_cues_separate() -> None:
    subtitles = (
        _subtitle(1, "手を挙げてください", 60.49, 63.99),
        _subtitle(2, "いつでもいいです", 63.99, 66.18),
    )

    result = ConservativeSubtitleMerger().merge(
        SubtitleMergeRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=subtitles,
        )
    )

    assert tuple(subtitle.text for subtitle in result.subtitles) == (
        "手を挙げてください",
        "いつでもいいです",
    )


def test_conservative_subtitle_merger_preserves_speaker_when_merging() -> None:
    subtitles = (
        _subtitle(1, "日本語", 0.0, 0.5, speaker_id="speaker-1"),
        _subtitle(2, "です。", 0.6, 1.0, speaker_id="speaker-1"),
    )

    result = ConservativeSubtitleMerger().merge(
        SubtitleMergeRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=subtitles,
        )
    )

    assert len(result.subtitles) == 1
    assert result.subtitles[0].speaker_id == "speaker-1"


def test_local_readability_optimizer_normalizes_punctuation() -> None:
    subtitle = _subtitle(1, " 日本語です..  ", 0.0, 1.0, speaker_id="speaker-1")

    result = LocalReadabilityOptimizer().optimize(
        ReadabilityOptimizationRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=(subtitle,),
        )
    )

    assert result.subtitles[0].text == "日本語です。"
    assert result.subtitles[0].speaker_id == "speaker-1"


def test_local_readability_optimizer_removes_spaces_between_japanese_words() -> None:
    subtitle = _subtitle(1, "最も 良いものを一つ", 0.0, 1.0)

    result = LocalReadabilityOptimizer().optimize(
        ReadabilityOptimizationRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=(subtitle,),
        )
    )

    assert result.subtitles[0].text == "最も良いものを一つ"


def test_local_readability_optimizer_restores_discourse_marker_comma() -> None:
    subtitles = (
        _subtitle(1, "では練習しましょう", 91.95, 93.87),
        _subtitle(2, "それでは始めます", 94.0, 95.0),
    )

    result = LocalReadabilityOptimizer().optimize(
        ReadabilityOptimizationRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=subtitles,
        )
    )

    assert tuple(subtitle.text for subtitle in result.subtitles) == (
        "では、練習しましょう",
        "それでは、始めます",
    )
    assert tuple(subtitle.time_range for subtitle in result.subtitles) == (
        TimeRange(91.95, 93.87),
        TimeRange(94.0, 95.0),
    )


def test_local_readability_optimizer_avoids_non_discourse_dewa() -> None:
    subtitles = (
        _subtitle(1, "日本では人気があります", 0.0, 1.0),
        _subtitle(2, "ではありません", 1.0, 2.0),
        _subtitle(3, "では、練習しましょう", 2.0, 3.0),
    )

    result = LocalReadabilityOptimizer().optimize(
        ReadabilityOptimizationRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=subtitles,
        )
    )

    assert tuple(subtitle.text for subtitle in result.subtitles) == tuple(
        subtitle.text for subtitle in subtitles
    )


def test_domain_subtitle_validator_accepts_valid_subtitles() -> None:
    result = DomainSubtitleValidator().validate(
        SubtitleValidationRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=(_subtitle(1, "日本語です。", 0.0, 1.0),),
        )
    )

    assert result.result.is_valid

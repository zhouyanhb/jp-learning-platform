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


def test_conservative_subtitle_merger_does_not_merge_complete_predicate_without_period() -> None:
    subtitles = (
        _subtitle(1, "明日行きます", 65.879, 66.18),
        _subtitle(2, "次の説明を聞いてください。", 66.3, 70.0),
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
        "明日行きます",
        "次の説明を聞いてください。",
    )


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


def test_local_readability_optimizer_resolves_small_subtitle_overlaps() -> None:
    subtitles = (
        _subtitle(1, "あ、それは大丈夫です。", 229.671, 231.175),
        _subtitle(2, "ただ、テキストと先生が同じでも", 231.08, 233.822),
        _subtitle(3, "クラスによって進む速さが違う場合もありますよね。", 233.8, 237.09),
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

    assert result.subtitles[0].time_range.end_seconds == 231.1275
    assert result.subtitles[1].time_range.start_seconds == 231.1275
    assert result.subtitles[1].time_range.end_seconds == 233.811
    assert result.subtitles[2].time_range.start_seconds == 233.811
    assert DomainSubtitleValidator().validate(
        SubtitleValidationRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=result.subtitles,
        )
    ).result.is_valid


def test_local_readability_optimizer_resolves_contained_subtitle_overlap() -> None:
    subtitles = (
        _subtitle(1, "長い字幕です。", 0.0, 10.0),
        _subtitle(2, "短い字幕です。", 1.0, 2.0),
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

    assert result.subtitles[0].time_range.end_seconds == 2.0
    assert result.subtitles[1].time_range.start_seconds == 2.0
    assert DomainSubtitleValidator().validate(
        SubtitleValidationRequest(
            source_path=Path("audio.mp3"),
            working_directory=Path("work"),
            run_id="run-001",
            segments=(_segment(),),
            subtitles=result.subtitles,
        )
    ).result.is_valid


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

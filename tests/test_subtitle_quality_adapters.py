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


def _subtitle(index: int, text: str, start: float, end: float) -> Subtitle:
    return Subtitle(index=index, text=text, time_range=TimeRange(start, end))


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


def test_local_readability_optimizer_normalizes_punctuation() -> None:
    subtitle = _subtitle(1, " 日本語です..  ", 0.0, 1.0)

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

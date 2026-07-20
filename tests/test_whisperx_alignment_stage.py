from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    Subtitle,
    TimeRange,
    Word,
)
from jp_learning_platform.infrastructure.whisperx_aligner import WhisperXAlignerAdapter
from jp_learning_platform.workflow import (
    InvalidWhisperXAlignerError,
    InvalidWhisperXAlignmentError,
    MissingWhisperSegmentsError,
    StageResult,
    WhisperXAlignment,
    WhisperXAlignmentRequest,
    WhisperXAlignmentStage,
)


@dataclass(slots=True)
class FakeAligner:
    alignment: WhisperXAlignment
    requests: list[WhisperXAlignmentRequest]

    def align(self, request: WhisperXAlignmentRequest) -> WhisperXAlignment:
        self.requests.append(request)
        return self.alignment


@dataclass(frozen=True, slots=True)
class InvalidAlignmentAligner:
    def align(self, request: WhisperXAlignmentRequest) -> object:
        return request


def _raw_segment() -> Segment:
    return Segment(
        position=0,
        text="日本語です",
        time_range=TimeRange(start_seconds=0.0, end_seconds=2.0),
    )


def _aligned_segment() -> Segment:
    words = (
        Word(
            text="日本語",
            time_range=TimeRange(start_seconds=0.0, end_seconds=0.9),
            confidence=0.95,
        ),
        Word(
            text="です",
            time_range=TimeRange(start_seconds=1.0, end_seconds=1.8),
            confidence=0.92,
        ),
    )
    sentence = Sentence(
        text="日本語です",
        time_range=TimeRange(start_seconds=0.0, end_seconds=2.0),
        words=words,
    )
    return Segment(
        position=0,
        text="日本語です",
        time_range=TimeRange(start_seconds=0.0, end_seconds=2.0),
        sentences=(sentence,),
    )


def _context(source_path: Path, segments: tuple[Segment, ...]) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path, segments=segments),
        working_directory=source_path.parent / "work",
    )


def test_whisperx_alignment_stage_aligns_existing_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    raw_segment = _raw_segment()
    aligned_segment = _aligned_segment()
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(aligned_segment,),
        ),
        requests=[],
    )

    result = WhisperXAlignmentStage(aligner=aligner).run(
        _context(source_path, (raw_segment,))
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "whisperx-alignment"
    assert aligner.requests == [
        WhisperXAlignmentRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(raw_segment,),
        )
    ]
    assert result.context.document.source_path == source_path
    assert result.context.document.segments == (aligned_segment,)
    assert result.context.run_id == "run-001"
    assert result.context.working_directory == tmp_path / "work"


def test_whisperx_alignment_stage_preserves_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    subtitle = Subtitle(
        index=1,
        text="字幕",
        time_range=TimeRange(start_seconds=0.0, end_seconds=2.0),
    )
    context = PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=source_path,
            segments=(_raw_segment(),),
            subtitles=(subtitle,),
        ),
        working_directory=tmp_path / "work",
    )
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(_aligned_segment(),),
        ),
        requests=[],
    )

    result = WhisperXAlignmentStage(aligner=aligner).run(context)

    assert result.context.document.subtitles == (subtitle,)


def test_whisperx_alignment_stage_accepts_custom_stage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(_aligned_segment(),),
        ),
        requests=[],
    )
    stage = WhisperXAlignmentStage(aligner=aligner, name="  whisperx  ")

    result = stage.run(_context(source_path, (_raw_segment(),)))

    assert stage.name == "whisperx"
    assert result.stage_name == "whisperx"


def test_whisperx_alignment_stage_rejects_missing_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "input.wav"
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=source_path,
            segments=(_aligned_segment(),),
        ),
        requests=[],
    )
    stage = WhisperXAlignmentStage(aligner=aligner)

    with pytest.raises(MissingWhisperSegmentsError):
        stage.run(_context(source_path, ()))

    assert aligner.requests == []


def test_whisperx_alignment_stage_rejects_invalid_aligner() -> None:
    with pytest.raises(InvalidWhisperXAlignerError):
        WhisperXAlignmentStage(aligner=object())


def test_whisperx_alignment_stage_rejects_invalid_alignment_return(
    tmp_path: Path,
) -> None:
    stage = WhisperXAlignmentStage(aligner=InvalidAlignmentAligner())

    with pytest.raises(InvalidWhisperXAlignmentError, match="WhisperXAlignment"):
        stage.run(_context(tmp_path / "input.wav", (_raw_segment(),)))


def test_whisperx_alignment_stage_rejects_mismatched_source_path(
    tmp_path: Path,
) -> None:
    aligner = FakeAligner(
        alignment=WhisperXAlignment(
            source_path=tmp_path / "other.wav",
            segments=(_aligned_segment(),),
        ),
        requests=[],
    )
    stage = WhisperXAlignmentStage(aligner=aligner)

    with pytest.raises(InvalidWhisperXAlignmentError, match="source path"):
        stage.run(_context(tmp_path / "input.wav", (_raw_segment(),)))


def test_whisperx_alignment_requires_segments() -> None:
    with pytest.raises(ValueError, match="segments"):
        WhisperXAlignment(source_path=Path("input.wav"), segments=())


def test_whisperx_alignment_is_immutable() -> None:
    alignment = WhisperXAlignment(
        source_path=Path("input.wav"),
        segments=(_aligned_segment(),),
    )

    with pytest.raises(FrozenInstanceError):
        alignment.segments = ()


def test_whisperx_adapter_maps_external_speaker_labels() -> None:
    adapter = WhisperXAlignerAdapter(device="cpu")

    segments = adapter._to_domain_segments(
        (
            {
                "text": "そう",
                "start": 0.0,
                "end": 0.4,
                "speaker": "speaker-1",
                "words": (
                    {
                        "word": "そう",
                        "start": 0.0,
                        "end": 0.4,
                        "speaker": "speaker-1",
                    },
                ),
            },
            {
                "text": "はい",
                "start": 0.5,
                "end": 0.9,
                "speaker": "speaker-2",
                "words": (
                    {
                        "word": "はい",
                        "start": 0.5,
                        "end": 0.9,
                        "speaker": "speaker-2",
                    },
                ),
            },
        )
    )

    assert tuple(segment.speaker_id for segment in segments) == (
        "speaker-1",
        "speaker-2",
    )
    assert tuple(segment.sentences[0].speaker_id for segment in segments) == (
        "speaker-1",
        "speaker-2",
    )
    assert tuple(segment.sentences[0].words[0].speaker_id for segment in segments) == (
        "speaker-1",
        "speaker-2",
    )


def test_whisperx_adapter_preserves_source_word_boundaries_for_japanese_text() -> None:
    adapter = WhisperXAlignerAdapter(device="cpu")
    source_words = (
        Word(text="これ", time_range=TimeRange(3.86, 4.3), confidence=0.94),
        Word(text="から", time_range=TimeRange(4.3, 4.56), confidence=0.99),
        Word(text="音", time_range=TimeRange(4.56, 5.18), confidence=0.97),
        Word(text="を", time_range=TimeRange(5.18, 5.44), confidence=0.99),
        Word(text="聞", time_range=TimeRange(5.44, 5.54), confidence=0.97),
        Word(text="いて", time_range=TimeRange(5.54, 5.76), confidence=0.99),
        Word(text="ください", time_range=TimeRange(5.76, 6.16), confidence=0.99),
        Word(text="音", time_range=TimeRange(6.81, 7.67), confidence=0.99),
    )
    source_sentence = Sentence(
        text="これから音を聞いてください 音",
        time_range=TimeRange(3.86, 7.67),
        words=source_words,
    )
    source_segment = Segment(
        position=0,
        text=source_sentence.text,
        time_range=source_sentence.time_range,
        sentences=(source_sentence,),
    )

    segments = adapter._to_domain_segments(
        (
            {
                "text": source_sentence.text,
                "start": 4.2,
                "end": 7.926,
                "words": (
                    {"word": "こ", "start": 4.2, "end": 4.321, "score": 1.0},
                    {"word": "れ", "start": 4.321, "end": 4.461, "score": 1.0},
                    {"word": "か", "start": 4.461, "end": 4.581, "score": 1.0},
                    {"word": "ら", "start": 4.581, "end": 5.102, "score": 1.0},
                    {"word": "音", "start": 5.102, "end": 5.402, "score": 1.0},
                    {"word": "を", "start": 5.402, "end": 5.582, "score": 1.0},
                    {"word": "聞", "start": 5.582, "end": 5.703, "score": 1.0},
                    {"word": "い", "start": 5.703, "end": 5.803, "score": 0.998},
                    {"word": "て", "start": 5.803, "end": 5.923, "score": 1.0},
                    {"word": "く", "start": 5.923, "end": 6.003, "score": 0.999},
                    {"word": "だ", "start": 6.003, "end": 6.223, "score": 1.0},
                    {"word": "さ", "start": 6.223, "end": 6.364, "score": 0.995},
                    {"word": "い", "start": 6.364, "end": 7.625, "score": 1.0},
                    {"word": "音", "start": 7.625, "end": 7.926, "score": 1.0},
                ),
            },
        ),
        (source_segment,),
    )

    words = segments[0].sentences[0].words
    assert tuple(word.text for word in words) == tuple(
        word.text for word in source_words
    )
    assert words[6].text == "ください"
    assert words[6].time_range == TimeRange(5.76, 6.16)
    assert words[7].text == "音"
    assert words[7].time_range == TimeRange(6.81, 7.67)


def test_whisperx_adapter_sentence_range_covers_all_aligned_words() -> None:
    adapter = WhisperXAlignerAdapter(device="cpu")

    segments = adapter._to_domain_segments(
        (
            {
                "text": "abc",
                "start": 10.0,
                "end": 11.0,
                "words": (
                    {"word": "a", "start": 10.0, "end": 10.1},
                    {"word": "b", "start": 9.6, "end": 9.8},
                    {"word": "c", "start": 10.8, "end": 11.0},
                ),
            },
        )
    )

    assert segments[0].time_range == TimeRange(9.6, 11.0)
    assert segments[0].sentences[0].time_range == TimeRange(9.6, 11.0)

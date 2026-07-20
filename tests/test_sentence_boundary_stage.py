from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    SentenceBoundaryCandidate,
    TimeRange,
    Word,
)
from jp_learning_platform.workflow import (
    InvalidSentenceBoundaryDetectionError,
    InvalidSentenceBoundaryDetectorError,
    InvalidSentenceBoundaryResolutionError,
    InvalidSentenceBoundaryResolverError,
    SentenceBoundaryDetection,
    SentenceBoundaryDetectionRequest,
    SentenceBoundaryDetectionStage,
    SentenceBoundaryResolution,
    SentenceBoundaryResolutionRequest,
    SentenceBoundaryResolverStage,
)


@dataclass(slots=True)
class RecordingDetector:
    detection: SentenceBoundaryDetection
    requests: list[SentenceBoundaryDetectionRequest]

    def detect(
        self,
        request: SentenceBoundaryDetectionRequest,
    ) -> SentenceBoundaryDetection:
        self.requests.append(request)
        return self.detection


@dataclass(slots=True)
class RecordingResolver:
    resolution: SentenceBoundaryResolution
    requests: list[SentenceBoundaryResolutionRequest]

    def resolve(
        self,
        request: SentenceBoundaryResolutionRequest,
    ) -> SentenceBoundaryResolution:
        self.requests.append(request)
        return self.resolution


@dataclass(frozen=True, slots=True)
class InvalidDetector:
    def detect(self, request: SentenceBoundaryDetectionRequest) -> object:
        return request


@dataclass(frozen=True, slots=True)
class InvalidResolver:
    def resolve(self, request: SentenceBoundaryResolutionRequest) -> object:
        return request


def _segment() -> Segment:
    words = (
        Word(text="聞いてください", time_range=TimeRange(0.0, 1.0)),
        Word(text="音が", time_range=TimeRange(1.6, 2.0)),
    )
    sentence = Sentence(
        text="聞いてください 音が",
        time_range=TimeRange(0.0, 2.0),
        words=words,
    )
    return Segment(
        position=0,
        text=sentence.text,
        time_range=sentence.time_range,
        sentences=(sentence,),
    )


def _candidate() -> SentenceBoundaryCandidate:
    return SentenceBoundaryCandidate(
        segment_position=0,
        after_word_index=0,
        boundary_time_seconds=1.3,
        pause_time_range=TimeRange(1.0, 1.6),
        acoustic_score=0.95,
        source="torch-energy-vad",
    )


def _context(source_path: Path) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path, segments=(_segment(),)),
        working_directory=source_path.parent / "work",
    )


def test_sentence_boundary_detection_stage_records_candidates(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    candidate = _candidate()
    detector = RecordingDetector(
        detection=SentenceBoundaryDetection(
            source_path=source_path,
            candidates=(candidate,),
        ),
        requests=[],
    )

    result = SentenceBoundaryDetectionStage(detector=detector).run(
        _context(source_path)
    )

    assert detector.requests == [
        SentenceBoundaryDetectionRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(_segment(),),
        )
    ]
    assert result.stage_name == "sentence-boundary-detection"
    assert result.context.document.sentence_boundary_candidates == (candidate,)


def test_sentence_boundary_resolver_stage_applies_resolved_segments(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "audio.mp3"
    candidate = _candidate()
    original_segment = _segment()
    resolved_sentence = Sentence(
        text="聞いてください",
        time_range=TimeRange(0.0, 1.0),
        words=original_segment.sentences[0].words[:1],
    )
    resolved_segment = Segment(
        position=0,
        text=resolved_sentence.text,
        time_range=original_segment.time_range,
        sentences=(resolved_sentence,),
    )
    context = PipelineContext(
        run_id="run-001",
        document=Document(
            source_path=source_path,
            segments=(original_segment,),
            sentence_boundary_candidates=(candidate,),
        ),
        working_directory=tmp_path / "work",
    )
    resolver = RecordingResolver(
        resolution=SentenceBoundaryResolution(
            source_path=source_path,
            segments=(resolved_segment,),
        ),
        requests=[],
    )

    result = SentenceBoundaryResolverStage(resolver=resolver).run(context)

    assert resolver.requests == [
        SentenceBoundaryResolutionRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(original_segment,),
            candidates=(candidate,),
        )
    ]
    assert result.stage_name == "sentence-boundary-resolver"
    assert result.context.document.segments == (resolved_segment,)
    assert result.context.document.sentence_boundary_candidates == (candidate,)


def test_sentence_boundary_detection_stage_rejects_invalid_detector() -> None:
    with pytest.raises(InvalidSentenceBoundaryDetectorError):
        SentenceBoundaryDetectionStage(detector=object())


def test_sentence_boundary_detection_stage_rejects_invalid_result(
    tmp_path: Path,
) -> None:
    stage = SentenceBoundaryDetectionStage(detector=InvalidDetector())

    with pytest.raises(InvalidSentenceBoundaryDetectionError):
        stage.run(_context(tmp_path / "audio.mp3"))


def test_sentence_boundary_resolver_stage_rejects_invalid_resolver() -> None:
    with pytest.raises(InvalidSentenceBoundaryResolverError):
        SentenceBoundaryResolverStage(resolver=object())


def test_sentence_boundary_resolver_stage_rejects_invalid_result(
    tmp_path: Path,
) -> None:
    stage = SentenceBoundaryResolverStage(resolver=InvalidResolver())

    with pytest.raises(InvalidSentenceBoundaryResolutionError):
        stage.run(_context(tmp_path / "audio.mp3"))


def test_sentence_boundary_detection_is_immutable() -> None:
    detection = SentenceBoundaryDetection(
        source_path=Path("audio.mp3"),
        candidates=(_candidate(),),
    )

    with pytest.raises(FrozenInstanceError):
        detection.candidates = ()

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from jp_learning_platform.domain import Document, PipelineContext, Segment, TimeRange
from jp_learning_platform.workflow import (
    InvalidJapaneseWordNormalizationError,
    InvalidJapaneseWordNormalizerError,
    JapaneseWordNormalization,
    JapaneseWordNormalizationRequest,
    JapaneseWordNormalizationStage,
    MissingJapaneseWordSegmentsError,
    StageResult,
)


@dataclass(slots=True)
class FakeWordNormalizer:
    normalization: JapaneseWordNormalization
    requests: list[JapaneseWordNormalizationRequest]

    def normalize(
        self,
        request: JapaneseWordNormalizationRequest,
    ) -> JapaneseWordNormalization:
        self.requests.append(request)
        return self.normalization


@dataclass(frozen=True, slots=True)
class InvalidWordNormalizer:
    def normalize(self, request: JapaneseWordNormalizationRequest) -> object:
        return request


def _segment() -> Segment:
    return Segment(
        position=0,
        text="天気です",
        time_range=TimeRange(0.0, 1.0),
    )


def _context(source_path: Path, segments: tuple[Segment, ...]) -> PipelineContext:
    return PipelineContext(
        run_id="run-001",
        document=Document(source_path=source_path, segments=segments),
        working_directory=source_path.parent / "work",
    )


def test_japanese_word_normalization_stage_replaces_segments(tmp_path: Path) -> None:
    source_path = tmp_path / "lesson.mp3"
    normalized_segment = Segment(
        position=0,
        text="天気です",
        time_range=TimeRange(0.0, 1.2),
    )
    normalizer = FakeWordNormalizer(
        normalization=JapaneseWordNormalization(
            source_path=source_path,
            segments=(normalized_segment,),
        ),
        requests=[],
    )

    result = JapaneseWordNormalizationStage(normalizer=normalizer).run(
        _context(source_path, (_segment(),))
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "japanese-word-normalization"
    assert result.context.document.segments == (normalized_segment,)
    assert normalizer.requests == [
        JapaneseWordNormalizationRequest(
            source_path=source_path,
            working_directory=tmp_path / "work",
            run_id="run-001",
            segments=(_segment(),),
        )
    ]


def test_japanese_word_normalization_stage_rejects_missing_segments(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "lesson.mp3"
    normalizer = FakeWordNormalizer(
        normalization=JapaneseWordNormalization(
            source_path=source_path,
            segments=(_segment(),),
        ),
        requests=[],
    )

    with pytest.raises(MissingJapaneseWordSegmentsError):
        JapaneseWordNormalizationStage(normalizer=normalizer).run(
            _context(source_path, ())
        )

    assert normalizer.requests == []


def test_japanese_word_normalization_stage_rejects_invalid_normalizer() -> None:
    with pytest.raises(InvalidJapaneseWordNormalizerError):
        JapaneseWordNormalizationStage(normalizer=object())


def test_japanese_word_normalization_stage_rejects_invalid_return(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "lesson.mp3"

    with pytest.raises(InvalidJapaneseWordNormalizationError):
        JapaneseWordNormalizationStage(normalizer=InvalidWordNormalizer()).run(
            _context(source_path, (_segment(),))
        )


def test_japanese_word_normalization_rejects_empty_segments() -> None:
    with pytest.raises(ValueError, match="segments"):
        JapaneseWordNormalization(source_path=Path("lesson.mp3"), segments=())


def test_japanese_word_normalization_is_immutable() -> None:
    normalization = JapaneseWordNormalization(
        source_path=Path("lesson.mp3"),
        segments=(_segment(),),
    )

    with pytest.raises(FrozenInstanceError):
        normalization.segments = ()

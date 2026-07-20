"""Japanese word timing normalization workflow stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from jp_learning_platform.domain import Document, PipelineContext, Segment
from jp_learning_platform.workflow.runtime import StageResult

JAPANESE_WORD_NORMALIZATION_STAGE_NAME = "japanese-word-normalization"

T = TypeVar("T")


def _normalize_name(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")

    return normalized


def _tuple_of_type(
    values: Iterable[T],
    item_type: type[T],
    field_name: str,
) -> tuple[T, ...]:
    try:
        tuple_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field_name} must be iterable.") from error

    for value in tuple_values:
        if not isinstance(value, item_type):
            raise TypeError(f"{field_name} must contain {item_type.__name__} values.")

    return tuple_values


@dataclass(frozen=True, slots=True)
class JapaneseWordNormalizationRequest:
    """Input passed to Japanese word timing normalizers."""

    source_path: Path
    working_directory: Path
    run_id: str
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "working_directory", Path(self.working_directory))
        object.__setattr__(self, "run_id", _normalize_name(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "segments",
            _tuple_of_type(self.segments, Segment, "segments"),
        )


@dataclass(frozen=True, slots=True)
class JapaneseWordNormalization:
    """Normalized Japanese word timings for existing transcript segments."""

    source_path: Path
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        segments = _tuple_of_type(self.segments, Segment, "segments")
        if not segments:
            raise ValueError("segments must not be empty.")

        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "segments", segments)


class JapaneseWordNormalizer(Protocol):
    """Contract implemented by Japanese tokenization/timing adapters."""

    def normalize(
        self,
        request: JapaneseWordNormalizationRequest,
    ) -> JapaneseWordNormalization:
        """Merge ASR/aligner pieces into Japanese word-level timings."""


class JapaneseWordStageError(RuntimeError):
    """Base error for Japanese word normalization failures."""


class InvalidJapaneseWordNormalizerError(JapaneseWordStageError):
    """Raised when a normalizer does not satisfy the stage contract."""

    def __init__(self) -> None:
        super().__init__(
            "Japanese word normalizer must define a callable normalize method."
        )


class MissingJapaneseWordSegmentsError(JapaneseWordStageError):
    """Raised when word normalization has no segments to inspect."""

    def __init__(self) -> None:
        super().__init__(
            "Japanese word normalization requires existing document segments."
        )


class InvalidJapaneseWordNormalizationError(JapaneseWordStageError):
    """Raised when a normalizer returns an invalid normalization result."""


@dataclass(frozen=True, slots=True)
class JapaneseWordNormalizationStage:
    """Workflow stage that normalizes Japanese word timing granularity."""

    normalizer: JapaneseWordNormalizer
    name: str = JAPANESE_WORD_NORMALIZATION_STAGE_NAME

    def __post_init__(self) -> None:
        if not callable(getattr(self.normalizer, "normalize", None)):
            raise InvalidJapaneseWordNormalizerError()

        object.__setattr__(self, "name", _normalize_name(self.name, "name"))

    def run(self, context: PipelineContext) -> StageResult:
        if not isinstance(context, PipelineContext):
            raise TypeError("context must be a PipelineContext.")

        if not context.document.segments:
            raise MissingJapaneseWordSegmentsError()

        request = JapaneseWordNormalizationRequest(
            source_path=context.document.source_path,
            working_directory=context.working_directory,
            run_id=context.run_id,
            segments=context.document.segments,
        )
        normalization = self.normalizer.normalize(request)
        if not isinstance(normalization, JapaneseWordNormalization):
            raise InvalidJapaneseWordNormalizationError(
                "Japanese word normalizer must return JapaneseWordNormalization."
            )

        if normalization.source_path != request.source_path:
            raise InvalidJapaneseWordNormalizationError(
                "Japanese word normalization source path must match the request."
            )

        document = Document(
            source_path=context.document.source_path,
            segments=normalization.segments,
            subtitles=context.document.subtitles,
            sentence_boundary_candidates=context.document.sentence_boundary_candidates,
        )
        next_context = PipelineContext(
            run_id=context.run_id,
            document=document,
            working_directory=context.working_directory,
        )
        return StageResult(stage_name=self.name, context=next_context)


__all__ = [
    "InvalidJapaneseWordNormalizationError",
    "InvalidJapaneseWordNormalizerError",
    "JAPANESE_WORD_NORMALIZATION_STAGE_NAME",
    "JapaneseWordNormalization",
    "JapaneseWordNormalizationRequest",
    "JapaneseWordNormalizationStage",
    "JapaneseWordNormalizer",
    "JapaneseWordStageError",
    "MissingJapaneseWordSegmentsError",
]

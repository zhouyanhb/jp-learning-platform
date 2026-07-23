"""Japanese learning-word normalization workflow stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jp_learning_platform.domain import Document, PipelineContext, Segment
from jp_learning_platform.workflow.runtime import StageResult

WORD_NORMALIZATION_STAGE_NAME = "word-normalization"


@dataclass(frozen=True, slots=True)
class WordNormalizationRequest:
    source_path: Path
    segments: tuple[Segment, ...]


@dataclass(frozen=True, slots=True)
class WordNormalization:
    source_path: Path
    segments: tuple[Segment, ...]


class WordNormalizer(Protocol):
    def normalize(self, request: WordNormalizationRequest) -> WordNormalization: ...


@dataclass(frozen=True, slots=True)
class WordNormalizationStage:
    normalizer: WordNormalizer
    name: str = WORD_NORMALIZATION_STAGE_NAME

    def run(self, context: PipelineContext) -> StageResult:
        if not context.document.segments:
            raise ValueError("Word normalization requires existing document segments.")
        request = WordNormalizationRequest(
            source_path=context.document.source_path,
            segments=context.document.segments,
        )
        result = self.normalizer.normalize(request)
        if not isinstance(result, WordNormalization):
            raise TypeError("Word normalizer must return WordNormalization.")
        if result.source_path != request.source_path or not result.segments:
            raise ValueError("Word normalization must preserve the source and segments.")
        document = Document(
            source_path=context.document.source_path,
            segments=result.segments,
            subtitles=context.document.subtitles,
        )
        return StageResult(
            stage_name=self.name,
            context=PipelineContext(
                run_id=context.run_id,
                document=document,
                working_directory=context.working_directory,
            ),
            data=result,
        )


__all__ = [
    "WORD_NORMALIZATION_STAGE_NAME",
    "WordNormalization",
    "WordNormalizationRequest",
    "WordNormalizationStage",
    "WordNormalizer",
]

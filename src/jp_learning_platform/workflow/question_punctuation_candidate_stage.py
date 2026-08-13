"""Generate auditable question-punctuation candidates without editing text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jp_learning_platform.domain import PipelineContext, Sentence, TimeRange
from jp_learning_platform.workflow.runtime import StageResult


QUESTION_PUNCTUATION_CANDIDATE_STAGE_NAME = "question-punctuation-candidates"


@dataclass(frozen=True, slots=True)
class QuestionPunctuationCandidate:
    segment_position: int
    sentence_index: int
    time_range: TimeRange
    text: str
    confidence: float
    evidence: tuple[str, ...]


class QuestionPunctuationCandidateDetector(Protocol):
    def detect(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
        following: Sentence | None,
    ) -> QuestionPunctuationCandidate | None: ...

    def detect_all(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
        following: Sentence | None,
    ) -> tuple[QuestionPunctuationCandidate, ...]: ...


@dataclass(frozen=True, slots=True)
class QuestionPunctuationCandidateStage:
    detector: QuestionPunctuationCandidateDetector
    name: str = QUESTION_PUNCTUATION_CANDIDATE_STAGE_NAME

    def run(self, context: PipelineContext) -> StageResult:
        indexed = tuple(
            (segment.position, sentence_index, sentence)
            for segment in context.document.segments
            for sentence_index, sentence in enumerate(segment.sentences)
        )
        candidates = tuple(
            candidate
            for index, (segment_position, sentence_index, sentence) in enumerate(indexed)
            for candidate in self.detector.detect_all(
                    segment_position,
                    sentence_index,
                    sentence,
                    indexed[index + 1][2] if index + 1 < len(indexed) else None,
                )
        )
        return StageResult(
            stage_name=self.name,
            context=context,
            data={"candidates": candidates},
        )


__all__ = [
    "QUESTION_PUNCTUATION_CANDIDATE_STAGE_NAME",
    "QuestionPunctuationCandidate",
    "QuestionPunctuationCandidateDetector",
    "QuestionPunctuationCandidateStage",
]

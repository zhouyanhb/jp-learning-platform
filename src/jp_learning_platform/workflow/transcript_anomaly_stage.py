"""Non-destructive transcript anomaly analysis before sentence resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jp_learning_platform.domain import PipelineContext, Segment, TimeRange
from jp_learning_platform.workflow.runtime import StageResult

TRANSCRIPT_ANOMALY_STAGE_NAME = "transcript-anomaly-analysis"


@dataclass(frozen=True, slots=True)
class TranscriptAnomalyCandidate:
    kind: str
    time_range: TimeRange
    segment_positions: tuple[int, ...]
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranscriptAnomalyRequest:
    source_path: Path
    segments: tuple[Segment, ...]


class TranscriptAnomalyDetector(Protocol):
    def detect(
        self,
        request: TranscriptAnomalyRequest,
    ) -> tuple[TranscriptAnomalyCandidate, ...]: ...


@dataclass(frozen=True, slots=True)
class TranscriptAnomalyAnalysisStage:
    detector: TranscriptAnomalyDetector
    name: str = TRANSCRIPT_ANOMALY_STAGE_NAME

    def run(self, context: PipelineContext) -> StageResult:
        candidates = self.detector.detect(
            TranscriptAnomalyRequest(
                source_path=context.document.source_path,
                segments=context.document.segments,
            )
        )
        return StageResult(
            stage_name=self.name,
            context=context,
            data={"candidates": candidates},
        )

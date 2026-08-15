"""Non-destructive transcript anomaly analysis before sentence resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dataclasses import replace

from jp_learning_platform.domain import Document, PipelineContext, Segment, TimeRange
from jp_learning_platform.workflow.runtime import StageResult

TRANSCRIPT_ANOMALY_STAGE_NAME = "transcript-anomaly-analysis"


@dataclass(frozen=True, slots=True)
class TranscriptAnomalyCandidate:
    kind: str
    time_range: TimeRange
    segment_positions: tuple[int, ...]
    confidence: float
    evidence: tuple[str, ...]
    sentence_indexes: tuple[int, ...] = ()


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


TRANSCRIPT_ANOMALY_ISOLATION_STAGE_NAME = "transcript-anomaly-isolation"
ISOLATED_CONTENT_ANOMALY_KINDS = frozenset(
    {
        "possible_alignment_failure",
        "possible_background_sound",
        "possible_repeated_laughter",
        "possible_repeated_vocalization",
    }
)


@dataclass(frozen=True, slots=True)
class TranscriptAnomalyIsolationStage:
    detector: TranscriptAnomalyDetector
    name: str = TRANSCRIPT_ANOMALY_ISOLATION_STAGE_NAME

    def run(self, context: PipelineContext) -> StageResult:
        candidates = self.detector.detect(
            TranscriptAnomalyRequest(
                source_path=context.document.source_path,
                segments=context.document.segments,
            )
        )
        by_sentence: dict[tuple[int, int], list[str]] = {}
        for candidate in candidates:
            if candidate.kind not in ISOLATED_CONTENT_ANOMALY_KINDS:
                continue
            for position in candidate.segment_positions:
                segment = next(
                    (
                        item
                        for item in context.document.segments
                        if item.position == position
                    ),
                    None,
                )
                if segment is None:
                    continue
                indexes = candidate.sentence_indexes or tuple(
                    range(len(segment.sentences))
                )
                for sentence_index in indexes:
                    if sentence_index < len(segment.sentences):
                        by_sentence.setdefault((position, sentence_index), []).append(
                            candidate.kind
                        )

        segments = tuple(
            replace(
                segment,
                sentences=tuple(
                    _isolate_sentence(
                        sentence,
                        by_sentence.get((segment.position, sentence_index), ()),
                    )
                    for sentence_index, sentence in enumerate(segment.sentences)
                ),
            )
            for segment in context.document.segments
        )
        document = Document(
            source_path=context.document.source_path,
            segments=segments,
            subtitles=context.document.subtitles,
        )
        return StageResult(
            stage_name=self.name,
            context=replace(context, document=document),
            data={
                "candidates": candidates,
                "isolated_sentence_count": len(by_sentence),
            },
        )


def _isolate_sentence(sentence, anomaly_kinds):
    kinds = tuple(dict.fromkeys((*sentence.anomaly_kinds, *anomaly_kinds)))
    if not kinds:
        return sentence
    return replace(
        sentence,
        anomaly_kinds=kinds,
        excluded_from_language_evaluation=True,
        learning_words=(),
        learning_words_suppressed=True,
    )

"""Non-destructive local ASR retries for high-evidence omission candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jp_learning_platform.domain import Segment, TimeRange
from jp_learning_platform.workflow.runtime import StageResult
from jp_learning_platform.workflow.transcript_anomaly_stage import (
    TranscriptAnomalyCandidate,
    TranscriptAnomalyDetector,
    TranscriptAnomalyRequest,
)


TRANSCRIPT_OMISSION_SHADOW_STAGE_NAME = "transcript-omission-shadow"
OMISSION_SHADOW_EVIDENCE = frozenset(
    {"long_uncovered_time_range", "substantial_stable_context"}
)


@dataclass(frozen=True, slots=True)
class TranscriptOmissionShadowRequest:
    source_path: Path
    segments: tuple[Segment, ...]
    candidates: tuple[TranscriptAnomalyCandidate, ...]


@dataclass(frozen=True, slots=True)
class TranscriptOmissionCandidateDisagreement:
    candidate_indexes: tuple[int, int]
    left_fragment: str
    right_fragment: str
    operation: str


@dataclass(frozen=True, slots=True)
class TranscriptOmissionShadowAudit:
    time_range: TimeRange
    segment_positions: tuple[int, ...]
    retry_attempted: bool
    raw_candidate_texts: tuple[str, ...] = ()
    extracted_candidate_texts: tuple[str, ...] = ()
    recovered_time_coverage: tuple[float, ...] = ()
    candidate_consensus_text: str = ""
    candidate_consensus_count: int = 0
    candidate_count: int = 0
    consensus_reached: bool = False
    normalized_candidate_texts: tuple[str, ...] = ()
    core_character_consensus: str = ""
    core_character_coverage: tuple[float, ...] = ()
    core_morpheme_consensus: tuple[str, ...] = ()
    candidate_disagreements: tuple[
        TranscriptOmissionCandidateDisagreement, ...
    ] = ()
    candidate_confidences: tuple[float | None, ...] = ()
    candidate_language_model_scores: tuple[float | None, ...] = ()
    candidate_morphology_penalties: tuple[int, ...] = ()
    context_validation_passed: bool = False
    confidence_validation_passed: bool = False
    language_model_validation_passed: bool = False
    morphology_validation_passed: bool = False
    lexical_uncertainty_detected: bool = False
    lexical_uncertainty_reasons: tuple[str, ...] = ()
    uncertain_noun_sequences: tuple[str, ...] = ()
    validation_passed: bool = False
    automatic_replacement_allowed: bool = False
    review_reasons: tuple[str, ...] = ()


class TranscriptOmissionShadowRecognizer(Protocol):
    def recognize_omission_candidates(
        self,
        request: TranscriptOmissionShadowRequest,
    ) -> tuple[TranscriptOmissionShadowAudit, ...]: ...


@dataclass(frozen=True, slots=True)
class TranscriptOmissionShadowStage:
    detector: TranscriptAnomalyDetector
    recognizer: TranscriptOmissionShadowRecognizer
    name: str = TRANSCRIPT_OMISSION_SHADOW_STAGE_NAME

    def run(self, context) -> StageResult:
        candidates = tuple(
            candidate
            for candidate in self.detector.detect(
                TranscriptAnomalyRequest(
                    source_path=context.document.source_path,
                    segments=context.document.segments,
                )
            )
            if candidate.kind == "possible_asr_omission"
            and OMISSION_SHADOW_EVIDENCE.issubset(candidate.evidence)
        )
        audits = self.recognizer.recognize_omission_candidates(
            TranscriptOmissionShadowRequest(
                source_path=context.document.source_path,
                segments=context.document.segments,
                candidates=candidates,
            )
        )
        return StageResult(
            stage_name=self.name,
            context=context,
            data={
                "eligible_candidate_count": len(candidates),
                "audits": audits,
                "shadow_only": True,
            },
        )

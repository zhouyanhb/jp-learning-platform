from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    TimeRange,
    Word,
)
from jp_learning_platform.workflow.transcript_anomaly_stage import (
    TranscriptAnomalyCandidate,
    TranscriptAnomalyRequest,
)
from jp_learning_platform.workflow.transcript_omission_shadow_stage import (
    TranscriptOmissionShadowAudit,
    TranscriptOmissionShadowRequest,
    TranscriptOmissionShadowStage,
)


@dataclass
class _Detector:
    candidates: tuple[TranscriptAnomalyCandidate, ...]

    def detect(self, request: TranscriptAnomalyRequest):
        del request
        return self.candidates


@dataclass
class _Recognizer:
    requests: list[TranscriptOmissionShadowRequest]

    def recognize_omission_candidates(self, request):
        self.requests.append(request)
        return tuple(
            TranscriptOmissionShadowAudit(
                time_range=candidate.time_range,
                segment_positions=candidate.segment_positions,
                retry_attempted=True,
            )
            for candidate in request.candidates
        )


def test_shadow_stage_only_retries_high_evidence_omissions(tmp_path: Path) -> None:
    segments = (_segment(0, 0.0, 4.0), _segment(1, 14.0, 18.0))
    eligible = _candidate(
        ("long_uncovered_time_range", "substantial_stable_context")
    )
    ordinary = _candidate(("uncovered_time_range", "low_confidence_edges"))
    recognizer = _Recognizer([])
    context = PipelineContext(
        "run",
        Document(Path("news.m4a"), segments),
        tmp_path,
    )

    result = TranscriptOmissionShadowStage(
        _Detector((eligible, ordinary)),
        recognizer,
    ).run(context)

    assert result.context is context
    assert result.data["shadow_only"] is True
    assert result.data["eligible_candidate_count"] == 1
    assert recognizer.requests[0].candidates == (eligible,)


def _segment(position: int, start: float, end: float) -> Segment:
    time_range = TimeRange(start, end)
    text = "十分に長い安定した文脈です。"
    sentence = Sentence(text, time_range, (Word(text, time_range, 0.9),))
    return Segment(position, text, time_range, (sentence,))


def _candidate(evidence: tuple[str, ...]) -> TranscriptAnomalyCandidate:
    return TranscriptAnomalyCandidate(
        kind="possible_asr_omission",
        time_range=TimeRange(4.0, 14.0),
        segment_positions=(0, 1),
        confidence=0.8,
        evidence=evidence,
    )
